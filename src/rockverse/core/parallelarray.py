"""
Provides the ParallelArray class for distributed management of chunked
`Zarr <https://zarr.readthedocs.io>`_ arrays.
"""

import h5py
import zarr
from itertools import product
from rockverse import _assert
from rockverse.errors import collective_only_rank0_runs, collective_raise, CustomCollectiveException
from rockverse.core.attributes import Attributes

from mpi4py import MPI
from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

class ParallelArray:
    """
    A high-level interface for distributed management of chunked arrays.

    This class is designed to work seamlessly in MPI environments.
    It builds on top of
    `Zarr arrays <https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#>`_
    and supports efficient parallel read, write, and mathematical operations on
    numeric or boolean Zarr arrays across multiple MPI processes.

    The class maps array chunks to MPI ranks to balance workload distribution,
    supports chunk-wise data access and modification with automatic
    synchronization, and ensures data consistency through collective MPI
    communication.

    .. note::
        This class should not be directly instantiated. Use the
        :ref:`creation functions <core module creation functions>`  instead.

    Parameters
    ----------
    zarray: zarr.core.array.Array
        An existing Zarr array to be managed in parallel.
    """

    def __init__(self, zarray):
        _assert.zarr_array('zarray', zarray)

        # Numeric or boolean only
        if zarray.dtype.kind not in 'cfuib':
            collective_raise(TypeError("expected numeric or boolean array for zarray."))

        self._zarray = zarray
        self._zarray.attrs['_ROCKVERSE_DATATYPE'] = 'ParallelArray'

        ranges = [range(m) for m in self._zarray.cdata_shape]
        self._chunk_block_index_to_id = {combo: k for k, combo in enumerate(product(*ranges))}
        self._chunk_id_to_block_index = {v: k for k, v in self._chunk_block_index_to_id.items()}

    @property
    def zarray(self):
        """
        The underlying Zarr array managed by this ParallelArray instance on each MPI process.
        """
        return self._zarray

    @property
    def chunk_process_map(self):
        """
        Mapping from chunk IDs to their chunk grid indices.
        Returns a dictionary where each key is a chunk ID (integer) and the value is the
        corresponding chunk's multi-dimensional block index (tuple of integers) within
        the Zarr array's chunk grid.
        This mapping facilitates identifying the chunk location and is used to distribute
        chunks among MPI processes for parallel operations.

        Returns
        -------
        dict[int, tuple[int, ...]]
            Dictionary mapping chunk IDs to chunk grid indices.
        """
        ranges = [range(m) for m in self._zarray.cdata_shape]
        return {k: combo for k, combo in enumerate(product(*ranges))}

    def chunk_slice_index(self, chunk_id):
        """
        Calculate the slice objects corresponding to the specified chunk ID.
        This method returns a tuple of slice objects that define the portion of the array
        covered by the chunk identified by `chunk_id`. These slices can be used to index
        the underlying local Zarr array to access or modify data within that chunk.

        Parameters
        ----------
        chunk_id : int
            The identifier of the chunk within the array's chunk grid.

        Returns
        -------
        tuple of slice
            Slices defining the region of the array covered by the chunk.

        Raises
        ------
        IndexError
            If the given chunk_id is not valid for this array.
        """
        chunk_process_map = self.chunk_process_map
        if chunk_id not in chunk_process_map.keys():
            collective_raise(IndexError(f'invalid chunk_id={chunk_id} for array nchunks={self._zarray.nchunks}.'))
        block_index = chunk_process_map[chunk_id]
        chunk_shape = self._zarray.chunks
        array_shape = self._zarray.shape
        ind = tuple((bi*ci, min(bi*ci+ci, si)) for bi, ci, si in zip(block_index, chunk_shape, array_shape))
        return tuple(slice(i[0], i[1], 1) for i in ind)

    def clean_chunks(self):
        """
        Reset all chunks not owned by the current MPI process to the array's fill value.
        This method iterates over all chunks of the Zarr array and clears those chunks
        whose assigned chunk ID modulo the total number of MPI processes does not match
        the current MPI rank. This helps ensure that each MPI rank only maintains data
        for its assigned chunks, which can prevent data inconsistencies in parallel operations.
        """
        chunk_process_map = self.chunk_process_map
        for block_id, block_index in chunk_process_map.items():
            if block_id % mpi_nprocs != mpi_rank:
                self.zarray[block_index] = self.array.fill_value

    def __getitem__(self, index):
        temp = zarr.zeros_like(self._zarray, store={})
        selection = temp[index]
        for block_id, block_index in self.chunk_process_map.items():
            if block_id % mpi_nprocs == mpi_rank:
                temp = zarr.zeros_like(self._zarray, store={})
                temp.blocks[block_index] = self._zarray.blocks[block_index]
                selection += temp[index]
        return comm.allreduce(selection, op=MPI.SUM)

    def __setitem__(self, index, array):
        temp = zarr.zeros_like(self._zarray, store={})
        for block_id, block_index in self.chunk_process_map.items():
            if block_id % mpi_nprocs == mpi_rank:
                temp.blocks[block_index] = self.zarray.blocks[block_index].copy()
                temp[index] = array
                self.zarray.blocks[block_index] = temp.blocks[block_index].copy()
        comm.barrier()


    def h5_dump(self, filename, path, mode='a', **kwargs):
        """
        Export the array data and its attributes to an HDF5 file.

        This method writes the contents of the parallel array into an HDF5 dataset at
        the specified path within the given file. The operation is performed serially
        by all MPI processes in rank order to ensure compatibility with
        non-MPI-enabled HDF5 libraries.

        Parameters
        ----------
        filename : str
            The name of the HDF5 file to write to.
        path : str
            The internal path within the HDF5 file where the dataset will be stored.
        mode : str, optional
            File open mode. Options include:

            - 'r' : Readonly, file must exist.
            - 'r+' : Read/write, file must exist.
            - 'w' : Create file, truncate if exists.
            - 'w-' or 'x' : Create file, fail if exists.
            - 'a' : Read/write if exists, create otherwise (default).
        **kwargs
            Additional keyword arguments to pass to `h5py.File`.

        Example
        -------
        Dump the contents of the array into the '/my/awesome/array' location in an HDF5 file:

        .. code-block:: python

            import rockverse as rv
            array_instance = rv.create_array(...)  # Create your array...
            array_instance.h5dump('filename.h5', path='/my/awesome/array')

        """

        # Serial writing. HDF5 installation may not have MPI enabled...

        for rank in range(mpi_nprocs):
            error_msg = ''
            if rank == mpi_rank:
                try:
                    mode_ = mode if rank == 0 else 'a'
                    with h5py.File(filename, mode_, **kwargs) as fobj:
                        if rank == 0:
                            # Rank 0 creates array
                            h5array = fobj.create_dataset(
                                path,
                                shape=self._zarray.shape,
                                chunks=self._zarray.chunks,
                                dtype=self._zarray.dtype,
                                fillvalue=self._zarray.fill_value)
                            # Rank 0 writes attributes
                            for k, v in self._zarray.attrs.items():
                                h5array.attrs[k] = v
                        else:
                            h5array = fobj[path]
                        # write chunks
                        for block_id, block_index in self.chunk_process_map.items():
                            block_slice = self.chunk_slice_index(block_id)
                            if block_id % mpi_nprocs == mpi_rank:
                                h5array[block_slice] = self._zarray[block_slice]
                except Exception as e:
                    error_msg = f"{e.__class__.__name__}: {e}"
            error_msg = comm.bcast(error_msg, root=rank)
            if error_msg:
                name, msg = error_msg.split(':')[0].strip(), ''.join(error_msg.split(':')[1:]).strip()
                msg = f"Error exporting array to {path} in {filename}. {msg}"
                collective_raise(CustomCollectiveException(name, msg))


def create_array(shape,
                 dtype,
                 chunk_shape=None,
                 store=None,
                 path=None,
                 overwrite=False,
                 **kwargs):
    """
    Create empty parallel array.

    Parameters
    ----------
    shape : tuple
        Desired array shape.
    dtype : string or dtype
        Numpy dtype. Type must be numeric (unsigned integer, integer, float, complex)
        or boolean. Ex: ``dtype=int``, ``dtype='u2'``, ``dtype='f4'``, ``dtype=np.complex128``.
    chunk_shape : iterable of ints | None, optional
        If iterable of integers, define the chunk shape. If `None`, `False`, empty tuple or
        any other object that makes ``not chunk_shape`` True, chunk shape will be set to the
        array shape, i.e., single chunk for the whole array.
    store :  str | zarr.storage.StoreLike | None, optional
        A string with the file path in the local file disk,
        or any valid `Zarr store <https://zarr.readthedocs.io/en/stable/user-guide/storage.html>`_,
        or ``None`` to use Memory store. Default is None.
    path : str or None, optional
        The path of the array within the store. If path is None, the array will be
        located at the root of the store.
    overwrite : bool, optional
        If True, delete all pre-existing data in the store at the specified path
        before creating the new image. Default value is False.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        `Zarr.create_array <https://zarr.readthedocs.io/en/stable/api/zarr/index.html#zarr.create_array>`_ function.

    Returns
    -------
    ParallelArray
        The created array object.
    """
    # Check for valid shape ---------------------
    _assert.iterable.ordered_integers_positive('shape', shape)

    # Check for valid dtype ---------------------
    _assert.condition.numeric_or_boolean('dtype', dtype)

    # Check for valid chunk_shape ---------------
    if not chunk_shape:
        _chunk_shape = shape
    else:
        _chunk_shape = chunk_shape
    _assert.iterable.ordered_numbers_positive('chunk_shape', _chunk_shape)

    # Check for valid overwrite -----------------
    _assert.instance('overwrite', overwrite, 'boolean', (bool,))

    # Check for valid path ----------------------
    if path is not None:
        _assert.instance('path', path, 'string', (str,))

    kwargs['shape'] = shape
    kwargs['dtype'] = dtype
    kwargs['chunks'] = _chunk_shape
    kwargs['store'] = store
    kwargs['overwrite'] = overwrite
    kwargs['path'] = path
    kwargs['zarr_format'] = 3
    kwargs['chunk_key_encoding'] = {"name": "default", "separator": "/"}

    # Attributes will go only to rank 0
    if 'attributes' in kwargs:
        attributes = kwargs.pop('attributes')
    else:
        attributes = {}
    attributes['_ROCKVERSE_DATATYPE'] = 'ParallelArray'

    if not store or isinstance(store, zarr.storage.MemoryStore):
        z = zarr.create(**kwargs)
    else: #Only rank 0 writes metadata to disk
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                z = zarr.create(**kwargs)
        for k in range(mpi_nprocs):
            if k == mpi_rank:
                z = zarr.open(store=store, path=kwargs['name'], mode='r+')
            comm.barrier()

    if mpi_rank == 0:
        z.attrs.update(**attributes)

    comm.barrier()
    return ParallelArray(z)


def array(data, chunk_shape=None, store=None, path=None, overwrite=False, **kwargs):
    """
    Create a parallel array and populate it with values from `data`.
    This function creates a new parallel array with the same shape and data type as the
    provided data array, using the specified chunk shape and storage options. After creation,
    it fills the array with the values from data, distributing the data across MPI processes
    for parallel management.

    Parameters
    ----------
    data : array-like
        The source array whose data will populate the new parallel array.
    chunk_shape : iterable of ints or None, optional
        Defines the chunk shape for the parallel array. If None or falsy, the chunk shape
        defaults to the shape of the entire array (single chunk).
    store : str, zarr.storage.StoreLike, or None, optional
        Storage location for the array. Can be a file path, a Zarr-compatible store object,
        or None for in-memory storage.
    path : str or None, optional
        The path within the store where the array will be located. If None, the root path is used.
    overwrite : bool, optional
        If True, any existing data at the target location will be overwritten. Default is False.
    **kwargs
        Additional keyword arguments passed to the underlying create_array function.

    Returns
    -------
    ParallelArray
        The newly created and populated parallel array.
    """
    _assert.array_like('data', data)
    new_array = create_array(shape=data.shape,
                             dtype=data.dtype,
                             chunk_shape=chunk_shape,
                             store=store,
                             path=path,
                             overwrite=overwrite,
                             **kwargs)
    new_array[...] = data
    return new_array


if __name__ == "__main__":
    import numpy as np
    import h5py

    shape=(5,2,8)
    a = create_array(shape=shape, dtype=np.float32, chunk_shape=(2,2,2))
    a[...] = np.random.rand(*shape)
    a.zarray[...]
    a[...]

    filename = r"C:\Users\GOB7\Downloads\test.h5"
    path='/my/awesome/array'
    self=a
    self.h5_dump(filename, path='/my/awesome/array', mode='w')

    with h5py.File(filename, 'r') as fobj:
        b = fobj['/my/awesome/array'][...]
        c = {k: v for k, v in fobj['/my/awesome/array'].attrs.items()}
    b == a[...]
    c
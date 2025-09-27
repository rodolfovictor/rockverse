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

    This class builds on top of
    `Zarr arrays <https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#>`_
    and is designed to work seamlessly in MPI environments. It supports
    efficient parallel read, write, and mathematical operations on
    numeric or boolean Zarr arrays across multiple MPI processes.

    The class maps array chunks to MPI ranks to balance memory and workload
    distribution, supports chunk-wise data access and modification with
    automatic MPI synchronization, and ensures data consistency through
    collective MPI communication.

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
        self._attrs = Attributes(zarray)
        self._attrs['_ROCKVERSE_DATATYPE'] = 'ParallelArray'

    @property
    def zarray(self):
        """
        The underlying Zarr array managed by this array instance on each MPI process.
        """
        return self._zarray

    @property
    def attrs(self):
        """
        The collective metadata attributes associated with this array as an
        :class:`Attributes` object.
        """
        return self._attrs

    @property
    def name(self):
        """
        Gets or sets the array name.
        """
        return self.attrs.get('name')

    @property
    def unit(self):
        """
        Gets or sets the array data unit.
        """
        return self.attrs.get('unit')

    @property
    def latex_name(self):
        """
        Gets or sets the LaTeX representation of the array name.
        """
        return self.attrs.get('latex_name')

    @property
    def latex_unit(self):
        """
        Gets or sets the LaTeX representation of the array data unit.
        """
        return self.attrs.get('latex_unit')

    @property
    def description(self):
        """
        Gets or sets the array description.
        """
        return self.attrs.get('description')

    @name.setter
    def name(self, value):
        _assert.string('name', value)
        self.attrs['name'] = value

    @unit.setter
    def unit(self, value):
        _assert.string('unit', value)
        self.attrs['unit'] = value

    @latex_name.setter
    def latex_name(self, value):
        _assert.string('latex_name', value)
        self.attrs['latex_name'] = value

    @latex_unit.setter
    def latex_unit(self, value):
        _assert.string('latex_unit', value)
        self.attrs['latex_unit'] = value

    @description.setter
    def description(self, value):
        _assert.string('description', value)
        self.attrs['description'] = value

    def get_plot_label(self, unit=True, latex=True):
        """
        Build a string 'name (unit)' based on the array attributes.

        Parameters
        ----------
        unit : bool, optional
            If True, use unit in the resulting string -> `'name (unit)'`.
            If False, discard unit -> `'name'`.
            Default is True.
        latex : bool, optional
            If True, use the LaTeX reprentations instead of the plain ones.
            Fall back to plain labels in case of absent LaTeX versions.
            Default is True.

        Returns
        -------
        str
            The resulting string to be used as a plot label.
        """
        if self.latex_name and latex:
            name_str = self.latex_name
        elif self.name:
            name_str = self.name
        else:
            return ''

        if unit and self.latex_unit and latex:
            unit_str = f" ({self.latex_unit})"
        elif unit and self.unit:
            unit_str = f" ({self.unit})"
        else:
            unit_str = ''

        return f"{name_str}{unit_str}"


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
        chunks = self._zarray.chunks
        array_shape = self._zarray.shape
        ind = tuple((bi*ci, min(bi*ci+ci, si)) for bi, ci, si in zip(block_index, chunks, array_shape))
        return tuple(slice(i[0], i[1], 1) for i in ind)

    def clean_chunks(self):
        """
        Reset all chunks not owned by the current MPI process to the array's fill
        value. This helps ensure that each MPI rank only maintains data for its
        assigned chunks, which can prevent data inconsistencies in parallel operations.
        """
        chunk_process_map = self.chunk_process_map
        for block_id, block_index in chunk_process_map.items():
            if block_id % mpi_nprocs != mpi_rank:
                self.zarray.blocks[block_index] = self.zarray.fill_value

    def __getitem__(self, index, /):
        """
        Return `self[index]`.

        This is a collective operation across all MPI ranks. Each rank
        contributes by accessing the chunks it owns, and the results are
        combined via an MPI all-reduce operation to produce the complete data
        selection.

        Parameters
        ----------
        index : int, slice, tuple, or array-like
            The indexing expression specifying the portion of the array to modify.
        """
        temp = zarr.zeros_like(self._zarray, store={})
        selection = temp[index]
        for block_id, block_index in self.chunk_process_map.items():
            if block_id % mpi_nprocs == mpi_rank:
                temp = zarr.zeros_like(self._zarray, store={})
                temp.blocks[block_index] = self._zarray.blocks[block_index]
                selection += temp[index]
        return comm.allreduce(selection, op=MPI.SUM)

    def __setitem__(self, index, array):
        """
        Set the values of `self[index]`.

        This is a collective operation across all MPI ranks. Each rank updates
        the chunks of the array it owns, modifying the corresponding portions
        of the underlying Zarr array. Synchronization barriers ensure that all
        ranks complete their updates before proceeding.

        Parameters
        ----------
        index : int, slice, tuple, or array-like
            The indexing expression specifying the portion of the array to modify.
        array : array-like
            The data to assign to the specified portion of the array.
        """
        temp = zarr.zeros_like(self._zarray, store={})
        for block_id, block_index in self.chunk_process_map.items():
            if block_id % mpi_nprocs == mpi_rank:
                temp.blocks[block_index] = self.zarray.blocks[block_index].copy()
                temp[index] = array
                self.zarray.blocks[block_index] = temp.blocks[block_index].copy()
        comm.barrier()

    @property
    def shape(self):
        """
        Array shape.
        """
        return self._zarray.shape

    @property
    def chunks(self):
        """
        Chunk shape.
        """
        return self._zarray.chunks

    @property
    def nchunks(self):
        """
        Total number of chunks.
        """
        return self._zarray.nchunks

    @property
    def dtype(self):
        """
        Numpy data type for the array.
        """
        return self._zarray.dtype

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
            array_instance.h5_dump('filename.h5', path='/my/awesome/array')

        """

        _assert.string('filename', filename)
        _assert.string('path', path)

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
                 chunks=None,
                 store=None,
                 path=None,
                 overwrite=False,
                 name=None,
                 unit=None,
                 description=None,
                 latex_name=None,
                 latex_unit=None,
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
    chunks : iterable of ints | None, optional
        If iterable of integers, define the chunk shape. If `None`, `False`, empty tuple or
        any other object that makes ``not chunks`` True, chunk shape will be set to the
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
    name : str, optional
        The array name.
    unit : str, optional
        The data unit of the array values.
    description : str, optional
        A description of the array.
    latex_name : str, optional
        The LaTeX representation of the array name.
    latex_unit : str, optional
        The LaTeX representation of the array data unit.
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

    # Check for valid chunks --------------------
    if not chunks:
        _chunks = shape
    else:
        _chunks = chunks
    _assert.iterable.ordered_numbers_positive('chunks', _chunks)

    # Check for valid overwrite -----------------
    _assert.instance('overwrite', overwrite, 'boolean', (bool,))

    # Check for valid path ----------------------
    if path is not None:
        _assert.instance('path', path, 'string', (str,))

    # Attributes will go through the proper class
    if 'attributes' in kwargs:
        attributes = kwargs.pop('attributes')
    else:
        attributes = {}

    # Check for valid default attributes --------
    for key, value in zip(('name', 'unit', 'description', 'latex_name', 'latex_unit'),
                          (name, unit, description, latex_name, latex_unit)):
        if value is not None:
            _assert.string(key, value)
            attributes[key] = value

    kwargs['shape'] = shape
    kwargs['dtype'] = dtype
    kwargs['chunks'] = _chunks
    kwargs['store'] = store
    kwargs['overwrite'] = overwrite
    kwargs['path'] = path
    kwargs['zarr_format'] = 3
    kwargs['chunk_key_encoding'] = {"name": "default", "separator": "/"}

    if not store or isinstance(store, zarr.storage.MemoryStore):
        z = zarr.create(**kwargs)
    else: #Only rank 0 writes metadata to disk
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                z = zarr.create(**kwargs)
        for k in range(mpi_nprocs):
            if k == mpi_rank:
                z = zarr.open(store=store, path=kwargs['path'], mode='r+')
            comm.barrier()
    comm.barrier()

    new_array = ParallelArray(z)
    new_array.attrs.update(attributes)

    return new_array


def array(data, chunks=None, store=None, path=None, overwrite=False, **kwargs):
    """
    Create a parallel array and populate it with values from `data`. This
    function creates a new parallel array with the same shape and data type as
    the provided data array, using the specified chunk shape and storage options.
    After creation, it fills the array with the values from data, distributing
    the data across MPI processes for parallel management.

    Parameters
    ----------
    data : array-like
        The source array whose data will populate the new parallel array.
    chunks : iterable of ints or None, optional
        Defines the chunk shape for the parallel array. If None or falsy, the
        chunk shape defaults to the shape of the entire array (single chunk).
    store : str, zarr.storage.StoreLike, or None, optional
        Storage location for the array. Can be a file path, a Zarr-compatible
        store object, or None for in-memory storage.
    path : str or None, optional
        The path within the store where the array will be located. If None,
        the root path is used.
    overwrite : bool, optional
        If True, any existing data at the target location will be overwritten.
        Default is False.
    **kwargs
        Additional keyword arguments passed to the underlying create_array
        function.

    Returns
    -------
    ParallelArray
        The newly created and populated parallel array.
    """
    _assert.array_like('data', data)
    if 'dtype' not in kwargs:
        kwargs['dtype'] = data.dtype
    new_array = create_array(shape=data.shape,
                             chunks=chunks,
                             store=store,
                             path=path,
                             overwrite=overwrite,
                             **kwargs)
    new_array[...] = data
    return new_array

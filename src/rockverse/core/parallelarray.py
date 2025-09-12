"""
Provides the ParallelArray class for distributed management of chunked
`Zarr <https://zarr.readthedocs.io>`_ arrays.
"""

import zarr
from itertools import product
from rockverse import _assert
from rockverse.errors import collective_raise

from mpi4py import MPI
from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

class ParallelArray:
    # TODO review create function. Should not be a call to a zarr array because of memory
    # TODO PREVENT RECHUNK OR MAKE self._chunk_block_index_to_id and self._chunk_id_to_block_indexinto functions
    # problem because dict is not sorted

    """
    A high-level interface for distributed management of chunked arrays.

    This class is designed to work seamlessly in MPI environments.
    It builds on top of
    `Zarr arrays <https://zarr.readthedocs.io/en/stable/user-guide/arrays.html#>`_
    and supports efficient parallel read, write, and mathematical operations on
    numeric or boolean Zarr arrays across multiple MPI processes.

    The class maps array chunks to MPI ranks to balance workload distribution,
    supports chunk-wise data access and modification with automatic
    synchronization, and ensures data consistency through collective
    communication.

    >>>>>>>>>>>>> REVISAR <<<<<<<<<<<<<<<<<<

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

        ranges = [range(m) for m in zarray.cdata_shape]
        self._chunk_block_index_to_id = {combo: k for k, combo in enumerate(product(*ranges))}
        self._chunk_id_to_block_index = {v: k for k, v in self._chunk_block_index_to_id.items()}


    def chunk_slice_index(self, chunk_id, return_indices=False):
        # TODO REVISAR DOCSTRING

        """
        Calculate the slice indices for a given Zarr chunk.

        This method computes the first and last+1 indices for a specific Zarr chunk
        within the array, based on the block's ID. Useful for working with chunked
        data in parallel processing.

        Parameters
        ----------
        chunk_id : int
            The ID of the block for which to calculate the slice indices.

        Returns
        -------
        tuple
            If ``return_indices`` is True, a tuple containing six integers:
            (box, bex, boy, bey, boz, bez). These represent the start and end
            indices for the block in the x, y, and z directions, respectively.
            If ``return_indices`` is False, a tuple containing the three slices:
            (slice(box, bex), slice(boy, bey), slice(boz, bez)).
        """
        if chunk_id not in self._chunk_id_to_block_index:
            collective_raise(IndexError(f'invalid chunk_id={chunk_id} for array nchunks={self._zarray.nchunks}.'))
        block_index = self._chunk_id_to_block_index[chunk_id]
        chunk_shape = self._zarray.chunks
        array_shape = self._zarray.shape

        ind = tuple((bi*ci, min(bi*ci+ci, si)) for bi, ci, si in zip(block_index, chunk_shape, array_shape))

        if return_indices:
            return ind
        return tuple(slice(i[0], i[1], 1) for i in ind)


    @property
    def zarray(self):
        """
        The underlying Zarr array managed by this ParallelArray instance.
        """
        return self._zarray

    def clean_chunks(self):
        for block_index, block_id in self._chunk_block_index_to_id.items():
            if block_id % mpi_nprocs != mpi_rank:
                self.zarray[block_index] = self.array.fill_value

    def chunk_id(self, index):
        if index in self._chunk_block_index_to_id:
            return self._chunk_block_index_to_id[index]
        collective_raise(IndexError(f"invalid block index {index} for chunk grid shape {self.zarray.cdata_shape}."))

    def __getitem__(self, index):
        temp = zarr.zeros_like(self._zarray, store={})
        selection = temp[index]
        for block_index, block_id in self._chunk_block_index_to_id.items():
            if block_id % mpi_nprocs == mpi_rank:
                temp = zarr.zeros_like(self.zarray, store={})
                temp.blocks[block_index] = self.zarray.blocks[block_index]
                selection += temp[index]
        return comm.reduce(selection, op=MPI.SUM)

    def __setitem__(self, index, array):
        temp = zarr.zeros_like(self._zarray, store={})
        for block_index, block_id in self._chunk_block_index_to_id.items():
            if block_id % mpi_nprocs == mpi_rank:
                temp.blocks[block_index] = self.zarray.blocks[block_index].copy()
                temp[index] = array
                self.zarray.blocks[block_index] = temp.blocks[block_index].copy()
        comm.barrier()


if __name__ == "__main__":
    import numpy as np
    zarray = zarr.array(np.random.rand(5,2,8)+1j*np.random.rand(5,2,8), chunk_shape=(2,2,2))
    a = ParallelArray(zarray)
    a.zarray[...]
    a[...]

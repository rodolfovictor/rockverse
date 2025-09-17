"""
Provides the Attributes class for parallel management of Zarr array and group attributes.
"""

import zarr
from itertools import product
from rockverse import _assert
from rockverse.errors import collective_only_rank0_runs, collective_raise, CustomCollectiveException

from mpi4py import MPI
from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

# TODO: implement other dict-like functions
# TODO: iter, keys, values, items, pop

class Attributes:
    """
    A high-level interface for attribute handling in RockVerse arrays or groups.

    This class builds on top of
    `Zarr Attributes <https://zarr.readthedocs.io/en/stable/user-guide/attributes.html>`_
    and offers a high-level interface to access and modify attributes of Zarr arrays or
    groups in MPI environments.

    It ensures that attribute reads and writes are coordinated among MPI processes, with
    only the rank 0 process performing actual attribute operations while others participate
    in collective synchronization. This design maintains consistency and avoids race
    conditions when working with attributes in distributed settings.

    .. note::
        This class should not be directly instantiated.
        It will be handled by the
        :ref:`creation functions <core module creation functions>`.


    Parameters
    ----------
    zobj: zarr.core.array.Array, zarr.core.group.Group
        An existing Zarr array or group to be managed in parallel.
    """

    def __init__(self, zobj):
        _assert.zarr_array_or_group('zobj', zobj)
        self._zobj = zobj

    @property
    def zobj(self):
        return _zobj

    def __getitem__(self, key):
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = self._zobj.attrs[key]
        value = comm.bcast(value, root=0)
        return value

    def __setitem__(self, key, value):
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                self._zobj.attrs[key] = value

    def __contains__(self, key):
        value = False
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = key in self._zobj.attrs
        value = comm.bcast(value, root=0)
        return value

    def __len__(self):
        value = 0
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = len(self._zobj.attrs)
        value = comm.bcast(value, root=0)
        return value

    def asdict(self):
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = {k: v for k, v in self._zobj.attrs}
        value = comm.bcast(value, root=0)
        return value

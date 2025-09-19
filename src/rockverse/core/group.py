import zarr
from rockverse import _assert
from rockverse.errors import collective_raise, collective_only_rank0_runs
from rockverse.core.attributes import Attributes
from rockverse.core.parallelarray import create_array, ParallelArray

from rockverse.configure import config
mpi_comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

class Group():
    """
    A high-level interface for managing Zarr groups in a parallel MPI
    environment. This class wraps a Zarr group and provides synchronized access
    and manipulation of group datasets, subgroups, arrays, and attributes
    across MPI processes, ensuring consistency and avoiding race conditions in
    parallel workflows.

    .. note::
        This class should not be directly instantiated. Use the
        :ref:`creation functions <core module creation functions>` instead.

    Parameters
    ----------
    zgroup : zarr.group.Group
        An existing Zarr group to be managed in parallel.
    """

    def __init__(self, zgroup):
        _assert.zarr_group('zgroup', zgroup)
        self._zgroup = zgroup
        self._attrs = Attributes(zgroup)

    @property
    def zgroup(self):
        """
        The underlying Zarr group managed by this instance.
        """
        return self._zgroup

    @property
    def attrs(self):
        """
        The collective metadata attributes associated with this group as an
        :class:`Attributes` object.
        """
        return self._attrs

    def create_group(self, path, overwrite=False, **kwargs):
        """
        Create a subgroup within this group at the specified path.

        Parameters
        ----------
        path : str
            The path for the new subgroup within the group path.
        overwrite : bool, optional
            If True, existing data at the specified path will be overwritten. Default is False.
        kwargs
            Additional keyword arguments passed to :func:`create_group <rockverse.create_group>`.

        Returns
        -------
        Group
            The new subgroup instance.
        """
        kwargs['store'] = self.zgroup.store
        kwargs['path'] = f"{self.zgroup.path}/{path}"
        kwargs['overwrite'] = overwrite
        new_group = create_group(**kwargs)
        return new_group

    def create_array(self, path, overwrite=False, **kwargs):
        """
        Create a new parallel array within this group with the specified name.

        Parameters
        ----------
        path : str
            The path for the new array within the group path.
        overwrite : bool, optional
            If True, existing data at the specified path will be overwritten. Default is False.
        kwargs
            Additional keyword arguments passed to :func:`create_array <rockverse.create_array>`.
        """
        kwargs['store'] = self.zgroup.store
        kwargs['path'] = f"{self.zgroup.path}/{path}"
        kwargs['overwrite'] = overwrite
        new_array = create_array(**kwargs)
        return new_array

    def __getitem__(self, key, /):
        """
        Return self[key].
        """
        if key not in self.zgroup:
            collective_raise(KeyError(key))
        rvdtype = None
        if mpi_rank == 0:
            rvdtype = self.zgroup[key].attrs.get('_ROCKVERSE_DATATYPE')
        rvdtype = mpi_comm.bcast(rvdtype, root=0)

        if rvdtype == 'Group':
            return Group(self.zgroup[key])
        if rvdtype == 'ParallelArray':
            return ParallelArray(self.zgroup[key])

        collective_raise(TypeError(f"Data in {key} is not a valid RockVerse data type."))


def create_group(store, path, overwrite=False, **kwargs):
    """
    Create a RockVerse group at the specified storage location and path.

    Parameters
    ----------
    store :  str | zarr.storage.StoreLike | None, optional
        A string with the file path in the local file disk,
        or any valid `Zarr store <https://zarr.readthedocs.io/en/stable/user-guide/storage.html>`_,
        or ``None`` to use Memory store. Default is None.
    path : str
        The internal path within the store to the group.
    overwrite : bool, optional
        If True, existing data at the specified path will be overwritten. Default is False.
    **kwargs
        Additional keyword arguments passed to `zarr.create_group`.

    Returns
    -------
    Group
        The created Group instance.
    """

    kwargs['store'] = store
    kwargs['overwrite'] = overwrite
    kwargs['path'] = path
    kwargs['zarr_format'] = 3

    if 'attributes' in kwargs:
        attrs = kwargs.pop('attributes')
    else:
        attrs = {}

    if '_ROCKVERSE_DATATYPE' not in attrs:
        attrs['_ROCKVERSE_DATATYPE'] = 'Group'

    if not store or isinstance(store, zarr.storage.MemoryStore):
        zgroup = zarr.create_group(**kwargs)
    else: #Only rank 0 writes metadata to disk
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                zgroup = zarr.create_group(**kwargs)
        for k in range(mpi_nprocs):
            if k == mpi_rank:
                zgroup = zarr.open(store=store, path=kwargs['path'], mode='r+')
            mpi_comm.barrier()

    if mpi_rank == 0:
        zgroup.attrs.update(**attrs)
    mpi_comm.barrier()

    return Group(zgroup)

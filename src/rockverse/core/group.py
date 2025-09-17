import os
import zarr
from rockverse import _assert
from rockverse._assert import collective_raise
from rockverse.configure import config, config_context
from rockverse.errors import collective_raise, collective_only_rank0_runs


from rockverse.configure import config
mpi_comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

class Group():

    def __init__(self, zgroup):
        _assert.zarr_group('zgroup', zgroup)
        self._zgroup = zgroup

    @property
    def zgroup(self):
        return self._zgroup

    def create_group(self, path, overwrite=False):
        zgroup = zarr.group(store=self.zgroup.store, path=path, overwrite=overwrite)
        zgroup.attrs['_ROCKVERSE_DATATYPE'] = 'Group'
        return Group(zgroup)

    def __getitem__(self, key):
        if key not in self.zgroup:
            collective_raise(KeyError(f"{key} not found in store."))
        else:
            rv_data_type = None
            if mpi_rank == 0:
                if '_ROCKVERSE_DATATYPE' in self.zgroup.attrs:
                    rv_data_type = self.zgroup.attrs['_ROCKVERSE_DATATYPE']
            rv_data_type = mpi_comm.bcast(rv_data_type, root=0)

            if rv_data_type == ('Group'):
                return Group(self.zgroup[key])

            if rv_data_type in _DATA_CLASS_MAP:
                return _DATA_CLASS_MAP[rv_data_type](self.zgroup[key])

            return self.zgroup[key]


def create_group(store, path, overwrite=False, **kwargs):

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

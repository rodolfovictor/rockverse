import os
import zarr
from rockverse import _assert
from rockverse._assert import collective_raise
from rockverse.configure import config, config_context
mpi_comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

from rockverse.voxel_image import VoxelImage
from rockverse.dect import DECTGroup
from rockverse.seismic import SeismicData, import_segy


_DATA_CLASS_MAP = {'VoxelImage': VoxelImage,
                   'DECTGroup': DECTGroup,
                   'DualEnergyCTGroup': DECTGroup,
                   'SeismicData': SeismicData}


class Group():

    def __init__(self, zgroup):
        _assert.zarr_group('zgroup', zgroup)
        self.zgroup = zgroup

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

    def import_segy(self, filename, path, **kwargs):
        return import_segy(filename=filename,
                           store=self.zgroup.store,
                           path=os.path.join(self.zgroup.path, path),
                           **kwargs)


def create_group(store, overwrite=False):
    zgroup = zarr.group(store, overwrite=overwrite)
    zgroup.attrs['_ROCKVERSE_DATATYPE'] = 'Group'
    return Group(zgroup)

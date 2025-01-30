__version__ = "0.3.5"

from rockverse._utils.logo import make_logo

# Expose RcParams as a library-wide instance
from rockverse.rc import rcparams

# Expose Config as a library-wide instance
from rockverse.config import config

# Define the public API
__all__ = [
    "__version__",
    "config",
    "rcparams",
    "make_logo",
    "open",
    "voxel_image",
    "region",
    "OrthogonalViewer",
    "dualenergyct",
]

from rockverse import voxel_image
from rockverse import region
from rockverse.viz import OrthogonalViewer
from rockverse import dualenergyct

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

import zarr
from rockverse._assert import collective_raise as _collective_raise
def open(store, **kwargs):
    """
    Opens a RockVerse data store and returns the appropriate object.

    Parameters:
    -----------
    store : str or zarr.storage.BaseStore
        Path or zarr store object containing RockVerse data.
    **kwargs : dict
        Additional arguments passed to zarr.open.

    Returns:
    --------
    object
        An instance of the corresponding RockVerse data class.

    Raises:
    -------
    ValueError
        If the store does not contain valid RockVerse data or an unsupported data type.
    """
    status = 'OK'
    if mpi_rank == 0:
        try:
            z = zarr.open(store, **kwargs)
            rv_data_type = z.attrs['_ROCKVERSE_DATATYPE']
        except Exception as e:
            status = str(e)
            pass
    status = comm.bcast(status, root=0)
    if status != 'OK':
        _collective_raise(ValueError(f"{store} does not contain valid RockVerse data."))

    z = zarr.open(store, **kwargs)
    rv_data_type = z.attrs['_ROCKVERSE_DATATYPE']

    if rv_data_type == 'VoxelImage':
        return voxel_image.VoxelImage(store=z.store)

    if rv_data_type == 'DualEnergyCTGroup':
        return dualenergyct.DualEnergyCTGroup(store=z.store)

    _collective_raise(ValueError(f"{store} does not contain valid RockVerse data."))

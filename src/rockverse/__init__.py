import sys

__version__ = "1.2.1d"

#----------------------------------------------------------------------
# Make sure the main depencies are present with the supported versions.
#----------------------------------------------------------------------
def _extract_version(mod, n=2):
    return tuple(map(int, mod.__version__.split('.')[:n]))

#Python --------------------------
PYVERSION = sys.version_info[:2]
if PYVERSION < (3, 11) or PYVERSION >= (3, 13):
    msg = ("RockVerse needs Python version >=3.11 and <3.13. Got Python "
            f"{PYVERSION[0]}.{PYVERSION[1]}.")
    raise ImportError(msg)

#Zarr ----------------------------
import zarr
zarr_version = _extract_version(zarr)
if zarr_version < (3, 0) or zarr_version >= (4, 0):
    msg = ("RockVerse needs Zarr version >=3.0 and >4.0. Got Zarr "
           f"{zarr.__version__}.")
    raise ImportError(msg)

import scipy
scipy_version = _extract_version(scipy, n=3)
if scipy_version > (1, 13, 1):
    msg = ("RockVerse requires SciPy version <=1.13.1. Got SciPy "
           f"{scipy.__version__}.")
    raise ImportError(msg)

import matplotlib
matplotlib_version = _extract_version(matplotlib)
if matplotlib_version < (3, 10):
    msg = ("RockVerse requires Matplotlib version >=3.10. Got Matplotlib "
           f"{matplotlib.__version__}.")
    raise ImportError(msg)


#----------------------------------------------------------------------
# Now build RockVerse
#----------------------------------------------------------------------
from rockverse._utils.logo import make_logo

# Expose Config as a library-wide instance
from rockverse.configure import config, config_context

# Expose MPI parameters library-wide variables
mpi_comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

# Define the public API
__all__ = [
    "__version__",
    "config",
    "make_logo",
    "open",
    "voxel_image",
    "region",
    "OrthogonalViewer",
    "dect",
]

from rockverse import voxel_image
from rockverse import region
from rockverse.viz import OrthogonalViewer
from rockverse import dect
from rockverse.errors import collective_only_rank0_runs, collective_raise

def open(store, *, path=None, **kwargs):
    """
    Opens RockVerse data.

    Parameters:
    -----------
    store : str or zarr.storage.BaseStore
        Path or zarr store object containing RockVerse data.
    path : str | None, optional
        The path within the store to open.
    kwargs : dict
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
    with collective_only_rank0_runs():
        if mpi_rank == 0:
            try:
                z = zarr.open(store=store, path=path, **kwargs)
                rv_data_type = z.attrs['_ROCKVERSE_DATATYPE']
            except:
                raise ValueError(f"{store} does not contain valid RockVerse data.")

    with zarr.config.set({'array.order': 'C'}):
        for k in range(mpi_nprocs):
            if k == mpi_rank:
                z = zarr.open(store=store, path=path, **kwargs)
            mpi_comm.barrier()

    rv_data_type = z.attrs['_ROCKVERSE_DATATYPE']

    if rv_data_type == 'VoxelImage':
        return voxel_image.VoxelImage(z)

    if rv_data_type in ('DECTGroup', 'DualEnergyCTGroup'):
        return dect.DECTGroup(z)

    collective_raise(ValueError(f"{store} does not contain valid RockVerse data."))

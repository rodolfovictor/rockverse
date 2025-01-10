__version__ = "0.2.1"

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
    #"dualenergyct",
]

from rockverse import voxel_image
from rockverse import region
from rockverse.viz import OrthogonalViewer
#from rockverse import dualenergyct


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
        An instance of the corresponding RockVerse data class (e.g., `VoxelImage`).

    Raises:
    -------
    ValueError
        If the store does not contain valid RockVerse data or an unsupported data type.
    """
    try:
        z = zarr.open(store, **kwargs)
        rv_data_type = z.attrs['_ROCKVERSE_DATATYPE']
    except Exception:
        _collective_raise(ValueError(f"{store} does not contain valid RockVerse data."))

    if rv_data_type == 'VoxelImage':
        return voxel_image.VoxelImage(store=z.store)

    _collective_raise(ValueError(f"{store} does not contain valid RockVerse data."))

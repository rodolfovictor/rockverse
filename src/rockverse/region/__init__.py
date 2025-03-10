"""
Provides classes to define abstract Regions of Interest (ROIs) within
:class:`VoxelImage <rockverse.voxel_image.VoxelImage>` objects.
ROIs are used to focus computational operations on specific
subsections of the voxel data, allowing for more efficient and targeted analysis. By
defining ROIs, you can limit processing to only the voxels inside these regions,
which is particularly useful for large-scale digital rock simulations and petrophysical
analyses.
"""

from rockverse.region.region import Region
from rockverse.region.sphere import Sphere
from rockverse.region.cylinder import Cylinder

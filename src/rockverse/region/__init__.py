"""
This module provides classes to define abstract Regions of Interest (ROIs) within
``VoxelImage`` objects. ROIs are used to focus computational operations on specific
subsections of the voxel data, allowing for more efficient and targeted analysis. By
defining ROIs, you can limit processing to only the voxels inside these regions,
which is particularly useful for large-scale digital rock simulations and petrophysical
analyses.

Key Features:
-------------
- Define ROIs using geometric shapes such as spheres and cylinders.
- Use ROIs to restrict computational operations, improving memory usage and performance.
- Seamlessly integrate with other classes and functions within the ``rockverse.digitalrock`` module.
"""

from rockverse.region.region import Region
from rockverse.region.sphere import Sphere
from rockverse.region.cylinder import Cylinder

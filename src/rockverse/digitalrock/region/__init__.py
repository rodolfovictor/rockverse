"""
This module provides classes for defining abstract regions of interest in
``VoxelImage`` objects. Regions can be used is several ``rockverse.digitalrock``
classes and functions and will limit the operations to the voxels inside the
defined regions of interest.
"""

from rockverse.digitalrock.region.sphere import Sphere
from rockverse.digitalrock.region.cylinder import Cylinder

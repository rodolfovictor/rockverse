"""
Visualization Module
====================

The ``viz`` module provides tools for visualizing various types of data and images.

Currently, this module includes the `OrthogonalViewer` class, which allows users
to visualize orthogonal slices (XY, XZ, ZY planes) of voxel images along with
their histograms. The `OrthogonalViewer` supports overlays for masks and
segmentations, enabling interactive exploration and analysis of the data.

Future extensions to this module may include additional visualization tools
to handle different types of data representations and enhance the overall
visualization capabilities.
"""

from rockverse.viz.orthogonal_slices import OrthogonalViewer

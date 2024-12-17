"""
Overview
--------

This module defines the basic class for RockVerse Digital Rock Petrophysics,
the VoxelImage class, intended to contain voxelized images and scalar fields in
general. The VoxelImage class builds upon
`Zarr arrays <https://zarr.readthedocs.io/en/stable/_autoapi/zarr.core.Array.html#zarr.core.Array>`_
by adding attributes and methods specifically designed for digital rock
petrophysics in a high-performance, parallel computing environment.

Since it is derived from Zarr arrays, it can efficiently handle large images by
leveraging Zarr's chunked storage. Methods in this class also take advantage of
Zarr's chunked storage for MPI-based parallel processing, optimized for both CPUs
and GPUs using `Numba <https://numba.pydata.org/>`_ 'just-in-time' compilation,
providing flexibility and high-performance computing (HPC) on cluster computers.

The VoxelImage class is designed for simplicity, handling complex computational
abstractions under the hood. This makes it accessible and user-friendly for non-HPC
specialists through high-level functions.

"""

from rockverse.digitalrock.voxel_image.voxel_image import VoxelImage
from rockverse.digitalrock.voxel_image.creation import (
    create,
    empty,
    zeros,
    ones,
    full,
    empty_like,
    zeros_like,
    ones_like,
    full_like,
    sphere_pack,
    import_raw
    )
#from rockverse.digitalrock.voxel_image.histogram import Histogram
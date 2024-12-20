"""
Digital Rock Module
===================

The ``rockverse.digitalrock`` module provides a collection of tools for creating,
importing, processing, analyzing, and visualizing high-resolution digital rock images.
It is designed to support computational petrophysics workflows, with a focus on
handling large datasets that may exceed available memory. This module includes functions
and classes for image manipulation, segmentation, simulation, and petrophysical property
analysis, all optimized for high-performance computing environments.

Key Features:
-------------
- **Voxel Image Manipulation**: Efficiently load and process high-resolution voxel data,
  with the ability to handle large volumes of data using memory-mapping techniques.
- **Segmentation and Feature Extraction**: Provides tools for segmenting rock images into
  different phases and extracting features such as porosity, permeability, and mineral
  distribution.
- **Region of Interest (ROI) Handling**: Defines regions of interest within voxel images
  to focus computational efforts on specific areas of the rock sample.
- **Advanced Numerical Methods**: Implements various numerical methods for analyzing the
  physical properties of rocks, including flow simulations, percolation theory, and
  diffusion models.
- **Integration with HPC**: Optimized for high-performance computing (HPC) environments,
  leveraging parallel computing frameworks such as MPI for distributed computing across
  clusters and GPUs.

"""

import rockverse.digitalrock.voxel_image as voxel_image
import rockverse.digitalrock.region as region
from rockverse.digitalrock.orthogonal_slices import OrthogonalViewer

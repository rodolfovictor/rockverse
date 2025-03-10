.. _tutorials voxel images:

============================
Introduction to Voxel Images
============================

Voxel images are the heart of Digital Rock Petrophysics workflows in RockVerse.
The :class:`VoxelImage <rockverse.voxel_image.VoxelImage>` class builds upon
`Zarr arrays <https://zarr.readthedocs.io/en/stable/user-guide/arrays.html>`_
by adding attributes and methods specifically optimized for digital rock petrophysics workflows in
a high performance parallel computing environment, seamlessly enabling advanced computational
capabilities in a user-friendly API.

If you're unfamiliar with Zarr, we recommend exploring the
`Zarr user guide <https://zarr.readthedocs.io/en/stable/user-guide/index.html>`_
to gain a deeper understanding of its fundamentals.

Chunked storage
===============

Chunked storage is a powerful feature in Zarr arrays.
VoxelImage leverages the power of Zarr arrays for storing voxel data in a chunked format.
This means that the large 3D image is divided into smaller, manageable chunks for efficient
storage, retrieval, and parallel processing.
It also supports compression within chunks, significantly reducing
file size without compromising data integrity. This is especially useful for the massive
datasets usually encountered in digital rock petrophysics.

Key advantages of chunked data:

- Efficient storage: Chunks can be compressed individually, reducing file size.
- Random access: Specific chunks can be accessed and processed without loading the entire dataset.
- Parallel processing: Different chunks can be processed simultaneously, speeding up computations.

.. grid:: 2
  :gutter: 0

  .. grid-item-card::
    :columns: 4
    :shadow: none

    .. image:: ../../_static/chunked-array.png
        :align: center
        :width: 200
        :alt: Chunked array

  .. grid-item-card::
    :columns: 8
    :shadow: none

    Sketch of a 6x6x6 voxel image divided into a 3x3x3 grid of 2x2x2 shaped chunks
    (`original image here <https://www.unidata.ucar.edu/software/netcdf/workshops/2012/nc4chunking/WhatIsChunking.html>`_).


.. toctree::
    :hidden:

    voxel_image/creating.ipynb

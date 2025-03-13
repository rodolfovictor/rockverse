.. _VoxelImage class:

rockverse.voxel_image.VoxelImage
================================

.. currentmodule:: rockverse.voxel_image

.. autoclass:: VoxelImage

Attributes
----------

.. autosummary::
    :toctree: _autogen

    ~VoxelImage.dtype
    ~VoxelImage.description
    ~VoxelImage.field_name
    ~VoxelImage.field_unit
    ~VoxelImage.nx
    ~VoxelImage.ny
    ~VoxelImage.nz
    ~VoxelImage.shape
    ~VoxelImage.ox
    ~VoxelImage.oy
    ~VoxelImage.oz
    ~VoxelImage.voxel_origin
    ~VoxelImage.hx
    ~VoxelImage.hy
    ~VoxelImage.hz
    ~VoxelImage.voxel_length
    ~VoxelImage.h_unit
    ~VoxelImage.voxel_unit
    ~VoxelImage.dimensions
    ~VoxelImage.bounding_box
    ~VoxelImage.meta_data_as_dict
    ~VoxelImage.ndim
    ~VoxelImage.chunks
    ~VoxelImage.nchunks
    ~VoxelImage.zarray

Basic methods
-------------

.. autosummary::
    :toctree: _autogen

    ~VoxelImage.chunk_slice_indices
    ~VoxelImage.get_voxel_coordinates
    ~VoxelImage.get_closest_voxel_index
    ~VoxelImage.check_mask_and_segmentation
    ~VoxelImage.create_mask_from_region

Parallel math
-------------

.. autosummary::
    :toctree: _autogen

    ~VoxelImage.math
    ~VoxelImage.combine

Saving to disk
--------------

.. autosummary::
    :toctree: _autogen

    ~VoxelImage.copy
    ~VoxelImage.export_raw

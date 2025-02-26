.. _VoxelImage class:

The ``VoxelImage`` class
------------------------

.. currentmodule:: rockverse.voxel_image

.. autoclass:: VoxelImage

   .. rubric:: Attributes

   .. autosummary::
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


   .. rubric:: Basic methods

   .. autosummary::
      ~VoxelImage.chunk_slice_indices
      ~VoxelImage.get_voxel_coordinates
      ~VoxelImage.get_closest_voxel_index
      ~VoxelImage.check_mask_and_segmentation
      ~VoxelImage.create_mask_from_region

   .. rubric:: Saving to disk

   .. autosummary::
      ~VoxelImage.copy
      ~VoxelImage.export_raw


   .. rubric:: Attribute documentation

   .. autoattribute:: VoxelImage.dtype
   .. autoattribute:: VoxelImage.description
   .. autoattribute:: VoxelImage.field_name
   .. autoattribute:: VoxelImage.field_unit
   .. autoattribute:: VoxelImage.nx
   .. autoattribute:: VoxelImage.ny
   .. autoattribute:: VoxelImage.nz
   .. autoattribute:: VoxelImage.shape
   .. autoattribute:: VoxelImage.ox
   .. autoattribute:: VoxelImage.oy
   .. autoattribute:: VoxelImage.oz
   .. autoattribute:: VoxelImage.voxel_origin
   .. autoattribute:: VoxelImage.hx
   .. autoattribute:: VoxelImage.hy
   .. autoattribute:: VoxelImage.hz
   .. autoattribute:: VoxelImage.voxel_length
   .. autoattribute:: VoxelImage.h_unit
   .. autoattribute:: VoxelImage.voxel_unit
   .. autoattribute:: VoxelImage.dimensions
   .. autoattribute:: VoxelImage.bounding_box
   .. autoattribute:: VoxelImage.meta_data_as_dict
   .. autoattribute:: VoxelImage.ndim
   .. autoattribute:: VoxelImage.chunks
   .. autoattribute:: VoxelImage.nchunks
   .. autoattribute:: VoxelImage.zarray

   .. rubric:: Basic methods documentation

   .. automethod:: chunk_slice_indices
   .. automethod:: get_voxel_coordinates
   .. automethod:: get_closest_voxel_index
   .. automethod:: check_mask_and_segmentation
   .. automethod:: create_mask_from_region
   .. automethod:: copy
   .. automethod:: export_raw

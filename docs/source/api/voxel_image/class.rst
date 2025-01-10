.. _VoxelImage class:

The VoxelImage class
------------------------

.. currentmodule:: rockverse.voxel_image

.. autoclass:: VoxelImage

   .. rubric:: Digital rock attributes summary

   .. autosummary::
      ~VoxelImage.description
      ~VoxelImage.field_name
      ~VoxelImage.field_unit
      ~VoxelImage.nx
      ~VoxelImage.ny
      ~VoxelImage.nz
      ~VoxelImage.shape
      ~VoxelImage.hx
      ~VoxelImage.hy
      ~VoxelImage.hz
      ~VoxelImage.voxel_length
      ~VoxelImage.dimensions
      ~VoxelImage.h_unit
      ~VoxelImage.voxel_unit
      ~VoxelImage.ox
      ~VoxelImage.oy
      ~VoxelImage.oz
      ~VoxelImage.voxel_origin
      ~VoxelImage.bounding_box
      ~VoxelImage.meta_data_as_dict

   .. rubric:: Basic methods summary

   .. autosummary::
      ~VoxelImage.chunk_slice_indices
      ~VoxelImage.get_voxel_coordinates
      ~VoxelImage.get_closest_voxel_index
      ~VoxelImage.check_mask_and_segmentation
      ~VoxelImage.create_mask_from_region

   .. rubric:: Inherited Zarr attributes summary

   .. autosummary::

      ~VoxelImage.attrs
      ~VoxelImage.basename
      ~VoxelImage.blocks
      ~VoxelImage.cdata_shape
      ~VoxelImage.chunk_store
      ~VoxelImage.chunks
      ~VoxelImage.compressor
      ~VoxelImage.dtype
      ~VoxelImage.fill_value
      ~VoxelImage.filters
      ~VoxelImage.info
      ~VoxelImage.initialized
      ~VoxelImage.is_view
      ~VoxelImage.itemsize
      ~VoxelImage.meta_array
      ~VoxelImage.name
      ~VoxelImage.nbytes
      ~VoxelImage.nbytes_stored
      ~VoxelImage.nchunks
      ~VoxelImage.nchunks_initialized
      ~VoxelImage.ndim
      ~VoxelImage.oindex
      ~VoxelImage.order
      ~VoxelImage.path
      ~VoxelImage.read_only
      ~VoxelImage.size
      ~VoxelImage.store
      ~VoxelImage.synchronizer
      ~VoxelImage.vindex
      ~VoxelImage.write_empty_chunks




   .. rubric:: Digital rock attributes documentation

   .. autoattribute:: description
   .. autoattribute:: field_name
   .. autoattribute:: field_unit
   .. autoattribute:: nx
   .. autoattribute:: ny
   .. autoattribute:: nz
   .. autoattribute:: shape
   .. autoattribute:: hx
   .. autoattribute:: hy
   .. autoattribute:: hz
   .. autoattribute:: voxel_length
   .. autoattribute:: dimensions
   .. autoattribute:: h_unit
   .. autoattribute:: voxel_unit
   .. autoattribute:: ox
   .. autoattribute:: oy
   .. autoattribute:: oz
   .. autoattribute:: voxel_origin
   .. autoattribute:: bounding_box
   .. autoattribute:: meta_data_as_dict


   .. rubric:: Basic methods documentation

   .. automethod:: chunk_slice_indices
   .. automethod:: get_voxel_coordinates
   .. automethod:: get_closest_voxel_index
   .. automethod:: check_mask_and_segmentation
   .. automethod:: create_mask_from_region



   .. rubric:: Inherited Zarr attributes documentation

   .. autoattribute:: attrs
   .. autoattribute:: basename
   .. autoattribute:: blocks
   .. autoattribute:: cdata_shape
   .. autoattribute:: chunk_store
   .. autoattribute:: chunks
   .. autoattribute:: compressor
   .. autoattribute:: dtype
   .. autoattribute:: fill_value
   .. autoattribute:: filters
   .. autoattribute:: info
   .. autoattribute:: initialized
   .. autoattribute:: is_view
   .. autoattribute:: itemsize
   .. autoattribute:: meta_array
   .. autoattribute:: name
   .. autoattribute:: nbytes
   .. autoattribute:: nbytes_stored
   .. autoattribute:: nchunks
   .. autoattribute:: nchunks_initialized
   .. autoattribute:: ndim
   .. autoattribute:: oindex
   .. autoattribute:: order
   .. autoattribute:: path
   .. autoattribute:: read_only
   .. autoattribute:: size
   .. autoattribute:: store
   .. autoattribute:: synchronizer
   .. autoattribute:: vindex
   .. autoattribute:: write_empty_chunks

# Backlog

NOW MIGRATING TO ZARR 3
   - Finished reviewing VoxelImage functions
   - Ensure VoxelImage voxel order C
   - Pin VoxelImage 3D
   - Assert VoxelImage Numeric or boolean
   - Write a collective __setitem__
      - COLLECTIVE SET SEND DATA TO MPI PROCESSOS
      - COLLECTIVE SET ONLY AFTER FIXIN ALL ORITINAL SETITEM TO FAST VOXELIMAGE._ARRAY[...]=...
    - SPEED UP VOXELIMAGE SAVE WHEN NOT RECHUNKING

CHECK GENERAL ROCKVERSE OPEN NOW IN ZARR 3

MIGRATE DUAL ENERGY TO ZARR 3


## Tasks to be completed

- Fine-tune threads per block GPU
    - Adjust GPU thread configurations for optimized performance in parallel computing tasks.
    - Investigate the best thread-per-block configuration for varying workloads.

- Support for 128-bit data in importing raw data
    - Extend the raw data import functionality to support 128-bit data types.
    - Ensure proper handling and conversion of 128-bit data during the import process.

- Add documentation for dependencies
    - Create a section in the documentation listing all project dependencies.

- VoxelImage
    - Implement other in place operators -=, *=, etc
    - Guarantee segmentation and masks with different chunk size when store is not in local file system.

- Orthogonal slices:
    - Add scalebar
    - Add calibrationbar
    - Add contour lines to scalar fields
    - Add chunk limit lines
    - Add "over black", "over white" options for segmentation colors
    - Histogram legend with segmentation name instead of phase number
    - Enforce 3D images
    - Generalize allowed colormaps
    - Allow ref_voxel as center of referenced block (chunk)
    - Choose which phases to be shown in histogram
    - Check and raise ValueError if complex dtype


- Histogram
    - Add histogram documentation to sphinx
    - Change to fast algorithm if equal size bins

- tqdm
    - Check printing to stdout instead of stderr

- Dual Energy
    - Implement run(block_id) to check inversion by running only on one chunk.
    - Bug accumulating results on second run
    - Bug freezing with GPU

- Zarr
    - Update to zarr 3
    - Test block indexing for 'F' or 'C'
    - put dimension_names in voxel image

> Note: These tasks are subject to change based on project needs and priorities.

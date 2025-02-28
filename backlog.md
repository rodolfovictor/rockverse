# Backlog

NOW MIGRATING TO ZARR 3

ACABEI DE ORGANIZAR O DUALENERGY
  - segmentation and mask should not be hashed
  - Tirar o collective call nos get/set property
  - Execute o tutorial
    - link the copy_image method documentation in segmentation building
  - put inversion parameters as input parameters or dictionary
  - Realização dos CT values estranha

Numba clean GPU memory

REVIEW PRINTS TO COLLECTIVE PRINT IN CONFIG

MIGRATING HISTOGRAM
  - PUT MDF OPTION (maybe discrete=True)

MIGRATING ORTHOGONALVIEWER
  - Check for interactivity
  - AINDA QUERO FAZER SÓ O RANK 0 colocar imagem
  - implement set method to set many properties at once with one refresh at the end

VOXELIMAGE
  - Revisado API
  - PUT DATA IN VOXELIMAGE CREATE
  - DEPRECATE from_array
  - EXPORT RAW COMPLEX IN REAL AND IMAGINARY
  - DETAIL TEST ON ALL VOXELIMAGE FUNCTIONS AND METHODS
  - GPT DOCUMENTATION
  - COPY change to faste chunk by chunk method if same chunk shape
  - change array to zarray
  - implement hash

DEVELOP CONFIG whith methods

DEPRECATE ASSERT drpdtype

REVISAR TUTORIAIS

REVISAR GALLERY

## Tasks to be completed

- Fine-tune threads per block GPU
    - Adjust GPU thread configurations for optimized performance in parallel computing tasks.
    - Investigate the best thread-per-block configuration for varying workloads.
    - Check for NumbaPerformanceWarning

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

- Dual Energy
    - Implement run(block_id) to check inversion by running only on one chunk.
    - Bug accumulating results on second run
    - Bug freezing with GPU

- Zarr
    - Update to zarr 3
    - Test block indexing for 'F' or 'C'
    - put dimension_names in voxel image

> Note: These tasks are subject to change based on project needs and priorities.

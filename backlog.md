# Backlog

## Tasks to be completed

- Fine-tune threads per block GPU
    - Adjust GPU thread configurations for optimized performance in parallel computing tasks.
    - Investigate the best thread-per-block configuration for varying workloads.

- Autoset MPI, GPU
    - Implement automatic detection and configuration of MPI (Message Passing Interface), and GPU usage based on system capabilities.
    - Ensure compatibility with different computing environments.

- Support for 128-bit data in importing raw data
    - Extend the raw data import functionality to support 128-bit data types.
    - Ensure proper handling and conversion of 128-bit data during the import process.

- Add documentation for dependencies
    - Create a section in the documentation listing all project dependencies.
    - Include installation instructions and version specifications for each dependency.

- Activate GPU in voxel_image modulus
    - math
    - sphere pack

- VoxelImage operators
    - Implement +, +=, -, -=, etc

- Orthogonal slices:
    - Add scalebar
    - Add calibrationbar
    - Add contour lines to scalar fields
    - Add "over black", "over white" options for segmentation colors
    - Histogram legend with segmentation name instead of phase number

- Histogram
    - Add histogram documentation to sphinx
    - Change to fast algorithm if equal size bins

- Zarr
    - Update to zarr 3

- tqdm
    - Check printing to stdout instead of stderr

> Note: These tasks are subject to change based on project needs and priorities.

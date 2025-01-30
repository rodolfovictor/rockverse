# Backlog

## Tasks to be completed

- Fine-tune threads per block GPU
    - Adjust GPU thread configurations for optimized performance in parallel computing tasks.
    - Investigate the best thread-per-block configuration for varying workloads.

- Support for 128-bit data in importing raw data
    - Extend the raw data import functionality to support 128-bit data types.
    - Ensure proper handling and conversion of 128-bit data during the import process.

- Add documentation for dependencies
    - Create a section in the documentation listing all project dependencies.

- VoxelImage operators
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

> Note: These tasks are subject to change based on project needs and priorities.

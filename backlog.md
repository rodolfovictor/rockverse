# Backlog

## Tasks to be completed

- Introduce well logs

- Fine-tune threads per block GPU
  - Adjust GPU thread configurations for optimized performance in parallel computing tasks.
  - Investigate the best thread-per-block configuration for varying workloads.
  - Check for NumbaPerformanceWarning.

- Support for 128-bit data in importing raw data
  - Extend the raw data import functionality to support 128-bit data types.
  - Ensure proper handling and conversion of 128-bit data during the import process.

- Add documentation for dependencies
  - Create a section in the documentation listing all project dependencies.

- VoxelImage
  - Implement in-place operators (e.g., -=, *=, etc.).
  - Guarantee segmentation and masks with different chunk sizes when the store is not in the local file system.
  - Deprecate from_array: integrate data into VoxelImage creation.
  - Export raw complex data into real and imaginary components.
  - Modify the copy function to implement a fast chunk-by-chunk method if the chunk shapes are the same.
  - Implement hashing.
  - Import raw boolean
  - Export raw boolean to Fiji as 0-255 uint8

- Orthogonal slices
  - Add scale bar.
  - Add calibration bar (MAtplotlib AnchoredSizeBar)
  - Add contour lines to scalar fields.
  - Add chunk limit lines.
  - Introduce "over black" and "over white" options for segmentation colors.
  - Display histogram legend with segmentation names instead of phase numbers.
  - Enforce 3D images.
  - Generalize allowed colormaps.
  - Allow ref_voxel as the center of the referenced block (chunk).
  - Choose which phases to display in the histogram.
  - Raise ValueError if complex dtype is detected.
  - Expand to multi-energy CT.
  - Expand to vector fields.
  - Check for interactivity.
  - Ensure only rank0 creates the figure while all handle collective calls.
  - Implement a set method to set multiple properties at once with a single refresh at the end.

- Histogram
  - Add histogram documentation to Sphinx.
  - Switch to a fast algorithm if using equal-size bins.
  - Introduce MDF option (maybe discrete=True).

- Dual Energy
  - Implement run(block_id) to check inversion by running only on one chunk.
  - Fix bug accumulating results on the second run.
  - Fix bug causing freezing with GPU.
  - Ensure segmentation and mask are not hashed.
  - Remove collective calls in get/set property.
  - Pass inversion parameters as input parameters or as a dictionary.
  - Check for unusual CT Monte Carlo drawings.
  - Implement re-hash to ignore stored hashes and update dependencies

- Zarr
  - Test block indexing for 'F' or 'C' formats.
  - Include dimension_names in the voxel image.

- Assert
  - Deprecate drpdtype

- Config attribute
  - Develop get and set methods to prevent the addition of new keys.

- Miscellaneous
  - Review print statements for collective print.
  - Replace "raise Exception"s for more specific exceptions
  - tqdm remove special characters when printing to file

- Documentation
  - install/troubleshooting  conda git

> Note: These tasks are subject to change based on project needs and priorities.

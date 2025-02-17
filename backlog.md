# Backlog

NOW MIGRATING TO ZARR 3

Resolvendo dual energy
ACABEI DE REORGANIZAR O CALIBRATION MATERIAL class
    - REVISAR AS FUNÇÕES REMANESCENTES
    - Avisos collective MPI calls no docstring


Execute o tutorial

REVIEW PRINTS TO COLLECTIVE PRINT IN CONFIG

MIGRATING HISTOGRAM
  - PUT MDF OPTION (maybe discrete=True)

MIGRATING ORTHOGONALVIEWER
  - Check for interactivity
  - AINDA QUERO FAZER SÓ O RANK 0 colocar imagem

VOXELIMAGE
  - PUT DATA IN VOXELIMAGE CREATE
  - DEPRECATE from_array
  - EXPORT RAW COMPLEX IN REAL AND IMAGINARY
  - DETAIL TEST ON ALL VOXELIMAGE FUNCTIONS AND METHODS
  - GPT DOCUMENTATION
  - COPY change to faste chunk by chunk method if same chunk shape
  - change array to zarray


MIGRATE DUAL ENERGY TO ZARR 3
  - DUALENERGYCTGROUP
      - reviewed __init__
  - --------------> Não está gerando todos os coeficientes <------------
  - GENERALIZE TO ARBITRARY PDFs
  - GUARANTEE SAME CHUNK SIZE FOR ALL ARRAYS
  - lowECT etc to snake_case
  - In the tutorial
    - put rechunk
    - raise number of bins at the beginning
  - put inversion parameters as input parameters or dictionary
  - name or path?
  - POSSIBILITY TO INVERT ONLY CHUNKS OR ONLY SEGMENTATION PHASES
  * ``'description'`` - string describing `calibration_material0`. Will be
    used in the default plots.



DEVELOP GENERAL HISTOGRAM

DEVELOP CONFIG whith methods like dual energy standard materials

DEPRECATE ASSERT drpdtype


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

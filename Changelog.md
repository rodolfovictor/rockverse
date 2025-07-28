# Release notes

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-07-28

### Added

- Dual energy CT inversion built-in Gaussian models for calibration materials attenuation coefficients.
- Dual energy inversion coefficient visualization.

### Fixed

- Bug on hashing dual energy group coefficient matrices.

## [1.1.2] - 2025-04-10

### Fixed

- Bug fix in MPI communicator for general ``open`` function.
- Updated to use Matplotlib version >= 3.10

## [1.1.1] - 2025-03-26

### Changed

- ``dualenergyct`` module renamed to ``dect``.
- ``rc`` module merged into ``configure`` module.
- Added ``config_context`` context manager for runtime configurations.

## [1.0.0] - 2025-03-10

**Major change**: RockVerse data migrated to Zarr 3.0.

### Added

- Version checks for main dependencies at module import time.
- VoxelImage class
    - Now can handle complex floating-point data.
    - ``VoxelImage.__setitem__`` implemented and is always MPI collective.
    - General function to execute in MPI rank 0 and broadcast exceptions.
- Dual energy inversion
    - Handles arbitrary PDFs for standard materials, not only Gaussian.
    - Segmentation array is now optional.

### Changed

- VoxelImage class
  - No longer directly inherits from Zarr array. Zarr array attributes and
  methods in the VoxelImage class are now only accessible through the
  class attribute ``VoxelImage.zarray``. Old Python scripts will crash if
  not updated.
  - ``VoxelImage.collective_getitem`` is deprecated and has been removed.
  ``VoxelImage.__getitem__`` is now always MPI collective. For non-collective
  get item, access ``VoxelImage.zarray`` instead. For non-collective set item,
  set ``VoxelImage.zarray`` instead.
  - ``VoxelImage.save`` is now ``VoxelImage.copy`` and can save a re-chunked
  copy to any Zarr store.
  - Restricted to 3D arrays for performance; other ndims coming soon.
- tqdm bars now output to stdout by default.
- Dual energy CT processing
  - Fully remodeled to use Zarr 3. No backward compatibility with RockVerse
  versions < 1.0.
  - Removed mandatory Gaussian distributions for the CT attenuation at
  standard materials. Now accepts arbitrary probability density functions.

## [0.3.5] - 2025-01-30

### Added

- GPU general runtime config class.
- GPU support for sphere pack generator.
- GPU support for parallel voxel image math.

### Changed

- VoxelImage ``copy_from_array`` method changed to ``from_array``.

## [0.3.3] - 2025-01-11

### Fixed

- Pinning Zarr and Matplotlib versions.
- Minor bugs in voxel image orthogonal viewer.

## [0.3.0] - 2025-01-10

### Added
- Monte Carlo Dual Energy CT

## [0.2.0] - 2025-01-09

### Added
- Digital rock features
  - Region module
    - Region class
    - Sphere class
    - Cylinder class
  - VoxelImage histogram
  - Orthogonal Viewer

### Fixed
- Zarr parameters not being passed to VoxelImage create function.

## [0.1.0] - 2024-12-08

### Added

- Initial documentation
- Gallery/miscellaneous: Using the RockVerse logo
- Assertion methods
- Digital rock features:
    - VoxelImage class
    - Creation functions
    - Sphere pack
    - Import/export raw data
    - Parallel math

### Changed
- Proportions in logo horizontal mode.

## [0.0.1] - 2024-09-15

### Added

- Initial test release with just the library logo.

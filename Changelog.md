# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-02-??
### Added
- Version checks for main dependencies at module import time.
- VoxelImage class
  - Now can handle complex floating-point
  - ``VoxelImage.__setitem__`` implemented and is always MPI collective.
- General function to execute in MPI rank 0 and broadcast exception
- Dual energy inversion
  - Handles arbitrary pdfs for standard materials, not only Gaussian.
  - Segmentation array is now optional

### Changed
- **Major change**: RockVerse data migrated to Zarr 3.0.
- VoxelImage class
  - Not directly inheritance of Zarr array anymore.
    Zarr array attributes and methods in VoxelImage class are now only
    accessible through class attribute ``VoxelImage.zarray``.
    Old python scripts will crash if not changed.
  - ``VoxelImage.collective_getitem`` is deprecated and was removed.
    ``VoxelImage.__getitem__`` now is always MPI collective. For non-collective
    getitem, get from ``VoxelImage.zarray`` instead.
    For non-collective setitem, set ``VoxelImage.zarray`` instead.
  - ``VoxelImage.save`` is now ``VoxelImage.copy`` and can save a re-chunked
    copy to any Zarr store.
  - Restricted to 3D arrays for performance. Other ``ndims`` comming soon.
- tqdm bars now outputs to stdout by default
- Dual energy CT processing
  - Fully remodeled to use Zarr 3. No back-compatibility to RockVerse versions < 1.0
  - Removed mandatory Gaussian distributions for the CT attenuation at standard
    materials. Now accept arbitrary probability density functions.

### Fixed
- ???

## [0.3.5] - 2025-01-30
### Added
- GPU general runtime config class
- GPU enabled for sphere pack generator
- GPU enabled for parallel voxel image math

### Changed
- VoxelImage ``copy_from_array`` method changed to ``from_array``.

## [0.3.3] - 2025-01-11

### Fixed
- Pinning Zarr and Matplotlib versions
- Minor bugs in voxel image orthogonal viewer

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
- Digital rock features
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

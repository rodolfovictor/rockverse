# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.5] - 2025-01-30
### Added
- GPU general runtime config class
- GPU enabled for sphere pack generator
- GPU enabled for parallel voxel image math

### Changed
- VoxelImage ``copy_from_array`` method changed to ``from_array``.

### Fixed

## [0.3.3] - 2025-01-11

### Fixed
- Pinning zarr and Matplotlib versions
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

### Changed
- None

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

### Fixed
- None

## [0.0.1] - 2024-09-15
### Added
- Initial test release with just the library logo.

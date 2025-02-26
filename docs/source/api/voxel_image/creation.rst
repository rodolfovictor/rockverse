.. _voxel image creation functions:

Creation functions
==================

Creation functions adapt the Zarr creation functions
to efficiently create ``VoxelImage`` objects in a
parallel MPI execution environment across multiple
CPU cores and GPUs.


.. currentmodule:: rockverse.voxel_image


.. rubric:: Basic creation functions

.. autosummary::
    ~create
    ~empty
    ~zeros
    ~ones
    ~full
    ~empty_like
    ~zeros_like
    ~ones_like
    ~full_like
    ~from_array

.. rubric:: Create from other formats

.. autosummary::
    ~sphere_pack
    ~import_raw

.. rubric:: Function documentation

.. autofunction:: create
.. autofunction:: empty
.. autofunction:: zeros
.. autofunction:: ones
.. autofunction:: full
.. autofunction:: empty_like
.. autofunction:: zeros_like
.. autofunction:: ones_like
.. autofunction:: full_like
.. autofunction:: from_array
.. autofunction:: sphere_pack
.. autofunction:: import_raw

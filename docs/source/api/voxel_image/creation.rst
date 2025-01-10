.. _voxel image creation functions:

Creation functions
------------------

Creation functions adapt the
`Zarr creation functions <https://zarr.readthedocs.io/en/stable/api/creation.html>`_
to efficiently create ``VoxelImage`` objects in a parallel MPI execution
environment across multiple CPU cores and GPUs.

.. currentmodule:: rockverse.voxel_image
.. autofunction:: create
.. autofunction:: empty
.. autofunction:: zeros
.. autofunction:: ones
.. autofunction:: full
.. autofunction:: empty_like
.. autofunction:: zeros_like
.. autofunction:: ones_like
.. autofunction:: full_like
.. autofunction:: sphere_pack
.. autofunction:: copy_from_array
.. autofunction:: import_raw

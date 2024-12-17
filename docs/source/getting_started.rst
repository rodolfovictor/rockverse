.. _rockverse_docs_gettingstarted:

###############
Getting started
###############

RockVerse is a Python library designed for high-performance computational in
geosciences, enabling efficient numerical computations, scalable parallel
processing, and optimized data handling. It integrates state-of-the-art Python
libraries to deliver performance and scalability for scientific workflows:

- `Numba <https://numba.pydata.org/>`_: Provides just-in-time compilation for efficient numerical operations
  on CPUs and CUDA-enabled GPUs.
- `mpi4py <https://mpi4py.readthedocs.io>`_: Enables distributed parallel processing via the Message Passing
  Interface (MPI), supporting multi-node and multi-processor workflows.
- `Zarr <https://zarr.readthedocs.io/en/stable/>`_: Optimizes storage and access for large, chunked, compressed N-dimensional
  arrays, ideal for handling geoscience data.
- `NumPy <https://numpy.org/>`_: Offers core numerical array operations essential for scientific computing.
- `SciPy <https://scipy.org/>`_: Supplies advanced mathematical functions and algorithms for numerical analysis.
- `Pandas <https://pandas.pydata.org/>`_: Simplifies data manipulation and analysis through powerful tabular data structures.

With this foundation, RockVerse facilitates scalable, high-performance computations on a
variety of hardware, from local machines to high-performance computing (HPC) clusters.

This section will guide you through the installation process, environment setup, and basic
usage examples to get started with RockVerse.


.. toctree::
    :maxdepth: 1
    :hidden:

    getting_started/install.rst

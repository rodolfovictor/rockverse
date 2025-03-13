.. _rockverse_docs_api:

=============
API reference
=============

**Version:** |version|


This reference manual details functions, modules, and objects included in RockVerse.
Make sure you also check our
:ref:`tutorials <rockverse_docs_tutorials>`
and the
:ref:`example gallery <rockverse_docs_gallery>`.


Library-Wide
============

.. list-table::
   :header-rows: 0
   :widths: auto

   * - ``rockverse.config``
     - An instance of the :class:`configuration class <rockverse.configure.Config>`
       for the RockVerse library, containing various configuration settings
       and parameters used throughout the library.
   * - ``rockverse.comm``
     - The runtime Message Passing Interface (MPI) communicator.
   * - ``rockverse.mpi_rank``
     - The rank of the calling process in the MPI communicator.
   * - ``rockverse.mpi_nprocs``
     - The total number of processes in the MPI communicator.


Digital Rock Modules
====================

.. autosummary::

  rockverse.voxel_image
  rockverse.region
  rockverse.dect
  rockverse.histogram


Visualization Modules
=====================

.. autosummary::

  rockverse.viz


Runtime configuration
=====================

.. autosummary::

  rockverse.configure



.. toctree::
  :hidden:
  :maxdepth: 2

  api/voxel_image
  api/region
  api/dect
  api/viz
  api/config
  api/histogram

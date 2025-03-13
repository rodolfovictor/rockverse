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

Objects
-------

.. list-table::
   :header-rows: 0
   :widths: auto

   * - ``rockverse.mpi_comm``
     - The runtime Message Passing Interface (MPI) communicator.
   * - ``rockverse.mpi_rank``
     - The rank of the calling process in the MPI communicator.
   * - ``rockverse.mpi_nprocs``
     - The total number of processes in the MPI communicator.
   * - ``rockverse.config``
     - The lib-wide instance of the :class:`configuration class <rockverse.configure.Config>`
       containing the configuration settings and parameters.

Functions
---------

.. autosummary::
  :toctree: _autogen

  rockverse.open

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

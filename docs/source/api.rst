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


.. _core module data types:

Basic data types
================

.. autosummary::

  ~rockverse.Attributes
  ~rockverse.ParallelArray
  ~rockverse.Coordinate
  ~rockverse.CoordinateSet
  ~rockverse.ScalarField
  ~rockverse.Group
  ~rockverse.FieldGroup

.. toctree::
  :hidden:
  :maxdepth: 2

  api/core/attributes
  api/core/paralellarray
  api/core/coordinate
  api/core/coordinateset
  api/core/scalarfield
  api/core/group
  api/core/fieldgroup


.. _core module creation functions:

Data creation functions
=======================

Create new data
---------------

.. autosummary::
  :toctree: _autogen

  ~rockverse.create_array
  ~rockverse.array
  ~rockverse.coordinate
  ~rockverse.create_group
  ~rockverse.create_fieldgroup
  ~rockverse.open

Importing from other formats
----------------------------

.. autosummary::
  :toctree: _autogen

  ~rockverse.read_las

Sample data
-----------

LAS files
^^^^^^^^^

.. autosummary::
  :toctree: _autogen

  ~rockverse.cwls_las_sample



Library-Wide Objects
====================

.. list-table::
   :header-rows: 0
   :widths: auto

   * - ``mpi_comm``
     - The runtime Message Passing Interface (MPI) communicator.
   * - ``mpi_rank``
     - The rank of the calling process in the MPI communicator.
   * - ``mpi_nprocs``
     - The total number of processes in the MPI communicator.
   * - ``config``
     - The lib-wide instance of the :class:`configuration class <rockverse.configure.Config>`
       containing the configuration settings and parameters.



Well data Modules
=================

.. autosummary::
  ~rockverse.las

Digital Rock Modules
====================

.. autosummary::

  ~rockverse.voxel_image
  ~rockverse.region
  ~rockverse.dect


Seismic Module
==============

.. autosummary::

  ~rockverse.seismic


Visualization Modules
=====================

.. autosummary::

  ~rockverse.viz


Runtime configuration
=====================

.. autosummary::

  ~rockverse.configure



.. toctree::
  :maxdepth: 2

  api/las
  api/voxel_image
  api/region
  api/dect
  api/seismic
  api/viz
  api/config

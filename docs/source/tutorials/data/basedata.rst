.. _tutorials understanding data:

==============
Base datatypes
==============

RockVerse data types are designed to operate seamlessly from personal computers to
large-scale, high-performance parallel computing systems, providing flexibility
across diverse hardware configurations. The high-level API integrates Message
Passing Interface (MPI) protocols to ensure robust communication and synchronization
in complex computational workflows, while keeping the MPI hard work transparent to
the user.

Base data classes build upon `Zarr <https://zarr.readthedocs.io>`_ arrays and groups,
by adding attributes and methods optimized for workflows in computational petrophysics,
enabling advanced computational capabilities across multiple CPUs and GPUs within a
user-friendly API. The chunked data structure of Zarr arrays is particularly well
suited for distribution and management in MPI environments, dividing large datasets
into smaller, manageable chunks for efficient storage, retrieval, and parallel processing.
It also supports compression within chunks, significantly reducing file size without
compromising data integrity. This is especially useful for the massive datasets usually
encountered in computational petrophysics, especially in digital rock workflows.

.. grid:: 2
  :gutter: 0

  .. grid-item-card::
    :columns: 4
    :shadow: none

    .. image:: ../../_static/chunked-array.png
        :align: center
        :width: 200
        :alt: Chunked array

  .. grid-item-card::
    :columns: 8
    :shadow: none

    Sketch of a $6 \\times 6 \\times 6$ three-dimensional array divided into a
    $3 \\times 3 \\times 3$ grid of $2 \\times 2 \\times 2$ shaped chunks.
    RockVerse distributes these chunks across the available MPI processes.
    (`original image here <https://www.unidata.ucar.edu/software/netcdf/workshops/2012/nc4chunking/WhatIsChunking.html>`_).



If you're unfamiliar with Zarr, we recommend exploring the
`Zarr user guide <https://zarr.readthedocs.io/en/stable/user-guide/index.html>`_
to gain a deeper understanding of its fundamentals.





.. toctree::
    :hidden:

    basedata/parallelarray.ipynb
    basedata/coordinates.ipynb
    basedata/scalarfields.ipynb

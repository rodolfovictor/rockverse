
.. _hpc_fine_tuning:

HPC Fine-Tuning
===============

The ``drp.dualenergyct`` module is designed to leverage High-Performance Computing (HPC) resources for efficient Dual Energy Computed Tomography (DECT) processing. This section provides guidance on fine-tuning the module's performance in HPC environments.

MPI Parallelization
-------------------

The module uses MPI (Message Passing Interface) for distributed computing across multiple nodes or cores. To optimize MPI performance:

1. Adjust the number of MPI processes based on your hardware and data size.
2. Use a process-to-core binding strategy appropriate for your cluster architecture.

Example MPI run command:

.. code-block:: bash

   mpirun -np 4 python your_dect_script.py

GPU Acceleration
----------------

When GPUs are available, the module can utilize CUDA for accelerated computations. To enable and optimize GPU usage:

1. Set ``use_gpu=True`` when calling the ``run()`` method.
2. Adjust the ``threads_per_block`` parameter to match your GPU architecture.

.. code-block:: python

   dect_group.threads_per_block = 256  # Adjust based on your GPU
   dect_group.run(use_gpu=True)

Memory Management
-----------------

Efficient memory usage is crucial for large datasets. Consider the following:

1. Adjust the ``chunk_size`` of your Zarr arrays to balance I/O performance and memory usage.
2. Use the ``hash_buffer_size`` parameter to control memory consumption during hash calculations.

.. code-block:: python

   dect_group.hash_buffer_size = 200  # Adjust based on available memory

Workload Distribution
---------------------

For optimal performance across heterogeneous computing resources:

1. Use the ``maximum_iterations`` parameter to control the Monte Carlo simulation workload.
2. Adjust the ``required_iterations`` parameter based on the desired statistical accuracy and available compute time.

.. code-block:: python

   dect_group.maximum_iterations = 100000
   dect_group.required_iterations = 10000

I/O Optimization
----------------

To minimize I/O bottlenecks:

1. Use a high-performance parallel file system if available.
2. Adjust Zarr compression settings to balance storage space and read/write performance.

Profiling and Monitoring
------------------------

To identify performance bottlenecks:

1. Use MPI profiling tools to analyze communication patterns and load balancing.
2. Monitor GPU utilization and memory usage with tools like ``nvidia-smi``.
3. Use Python profiling tools to identify CPU-bound operations that may benefit from further optimization.

.. note::
   Always test and benchmark your specific use case to find the optimal configuration for your HPC environment.

.. _rockverse_docs_install:

Installation
============

.. contents:: Table of Contents
   :depth: 2

Prerequisites
-------------

- **Python** 3.7 or later
- Basic familiarity with command-line operations
- Basic familiarity with virtual environments (Conda or pip)

Installing
----------

To avoid conflicts with existing Python setups, install RockVerse in a dedicated virtual environment.

If you're unfamiliar with virtual environments, refer to:

- `Python virtual environments <https://docs.python.org/3/tutorial/venv.html>`_
- `Conda environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_

In this guide, the virtual environment is named ``rockverse-env``.

.. note::
   Choose either **Conda** or **pip** for the installation process. Using **pip** requires a working MPI implementation with headers and a C compiler. Ensure ``mpicc`` and ``mpiexec`` or ``mpirun`` point to the desired MPI installation. If MPI is unavailable, **Conda** is recommended.

1. Create a Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

   .. tab-item:: Conda

      Update Conda (optional but recommended). Ensure you're using Conda 23.10 or later for improved speed:

      .. code-block:: sh

         conda update -n base conda

      Create and activate the environment:

      .. code-block:: sh

         conda create --name rockverse-env
         conda activate rockverse-env

   .. tab-item:: Pip

      Create and activate a virtual environment:

      **On Windows**:

      .. code-block:: sh

         cd path\to\my\environment
         python -m venv rockverse-env
         .\rockverse-env\Scripts\activate

      **On Linux/macOS** (bash example):

      .. code-block:: sh

         cd path/to/my/environment
         python -m venv rockverse-env
         source ./rockverse-env/bin/activate

2. Configure MPI
~~~~~~~~~~~~~~~~

If you lack a system MPI installation, use one of the MPI implementations available on **conda-forge**. This step requires a Conda environment.

.. tab-set::

   .. tab-item:: OpenMPI (Linux/macOS)

      Installs `Open MPI <https://www.open-mpi.org/>`_
      with command-line executable ``mpirun``:

      .. code-block:: sh

         conda install -c conda-forge openmpi

   .. tab-item:: MPICH (Linux/macOS)

      Installs `MPICH <https://www.mpich.org/>`_
      with command-line executable ``mpiexec``:

      .. code-block:: sh

         conda install -c conda-forge mpich

   .. tab-item:: Intel MPI (Linux/Windows)

      Installs `Intel MPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html>`_
      with both command-line executables ``mpirun`` and ``mpiexec``:

      .. code-block:: sh

         conda install -c conda-forge impi_rt

   .. tab-item:: Microsoft MPI (Windows)

      Installs `Microsoft MPI <https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_
      and command-line executable ``mpiexec``:

      .. code-block:: sh

         conda install -c conda-forge msmpi

If you prefer a system-installed MPI (e.g., on cluster computers), ensure ``mpicc`` and ``mpirun`` or ``mpiexec`` point to the correct installation.

Now install **mpi4py**:

.. tab-set::

   .. tab-item:: Conda

      .. code-block:: sh

         conda install -c conda-forge mpi4py

   .. tab-item:: Pip

      .. code-block:: sh

         pip install --no-cache-dir mpi4py

Test your MPI installation:

.. tab-set::

   .. tab-item:: mpirun

      .. code-block:: sh

         mpirun -n 5 python -m mpi4py.bench helloworld

   .. tab-item:: mpiexec

      .. code-block:: sh

         mpiexec -n 5 python -m mpi4py.bench helloworld

You should get an output similar to this
('localhost' will be the hostname in your machine):

.. code-block:: sh

    Hello, World! I am process 0 of 5 on localhost.
    Hello, World! I am process 1 of 5 on localhost.
    Hello, World! I am process 2 of 5 on localhost.
    Hello, World! I am process 3 of 5 on localhost.
    Hello, World! I am process 4 of 5 on localhost.

3. Install RockVerse
~~~~~~~~~~~~~~~~~~~~

Install RockVerse and its dependencies (this might take a while...)

.. tab-set::

   .. tab-item:: Conda (Stable)

      .. code-block:: sh

         conda install -c conda-forge rockverse

   .. tab-item:: Pip (Stable)

      .. code-block:: sh

         pip install rockverse

   .. tab-item:: Pip (Nightly)

      .. code-block:: sh

         pip install git+https://github.com/rodolfovictor/rockverse.git

   .. tab-item:: Development Mode

      .. code-block:: sh

         git clone https://github.com/rodolfovictor/rockverse.git
         cd rockverse
         pip install -e .


4. (Optional) Configure Numba access to GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RockVerse supports accelerated computations on CUDA-enabled GPUs using Numba. Multiple GPUs
can be utilized simultaneously through RockVerse's MPI-based distribution strategies.
By default, RockVerse prioritizes GPU devices when available.
To enable GPU support, ensure that:

1. **You have CUDA-capable hardware and drivers installed**.
   Refer to Numba's `CUDA documentation <https://numba.readthedocs.io/en/stable/cuda/overview.html#supported-gpus>`_ for hardware compatibility.

2. **Install the appropriate CUDA toolkit**:

.. tab-set::

   .. tab-item:: Conda (CUDA 12)

      Install CUDA 12 support:

      .. code-block:: sh

         conda install -c conda-forge cuda-nvcc cuda-nvrtc

   .. tab-item:: Conda (CUDA 11)

      Install CUDA 11 toolkit:

      .. code-block:: sh

         conda install -c conda-forge cudatoolkit

   .. tab-item:: Pip

      Install NVIDIA bindings:

      .. code-block:: sh

         pip install cuda-python

      Set environment variables:

      .. code-block:: sh

         export NUMBA_CUDA_USE_NVIDIA_BINDING="1"  # Linux
         set NUMBA_CUDA_USE_NVIDIA_BINDING="1"     # Windows

Test Numba's GPU detection:

.. code-block:: sh

   python -c "from numba import cuda; print(cuda.is_available())"

If the output is ``True``, you can list the devices running

.. code-block:: sh

   python -c "from numba import cuda; print([d.name for d in cuda.gpus])"


Updating RockVerse
------------------

.. tab-set::

   .. tab-item:: Conda (Stable)

      .. code-block:: sh

         conda update -c conda-forge rockverse

   .. tab-item:: Pip (Stable)

      .. code-block:: sh

         pip install --upgrade rockverse

   .. tab-item:: Pip (Nightly)

      .. code-block:: sh

         pip install --upgrade git+https://github.com/rodolfovictor/rockverse.git

   .. tab-item:: Development Mode

      Just pull the last updates from Github:

      .. code-block:: sh

         cd /path/to/local/installation
         git pull

Troubleshooting
---------------

If you encounter build issues with Pandoc, try reinstalling from Conda:

.. code-block:: sh

   pip uninstall pandoc
   conda install pandoc

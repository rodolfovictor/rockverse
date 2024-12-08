Installation
============

.. contents:: Table of Contents
   :depth: 2


Introduction
------------

RockVerse is developed to work in high-performance taking advantage of two
key libraries: `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ leverages
MPI (Message Passing Interface), enabling distributed parallel computations
across multiple processors or nodes and `Numba <https://numba.pydata.org/>`_
enables high-performance just-in-time compilation for numerical operations on
both CPUs and GPUs. It is generally best to install these two packages first
to better handle potential errors during the installation process.


Prerequisites
-------------

- Python 3.7 or later

- Basic familiarity with command-line operations

- Basic familiarity with conda and pip


1. Create a virtual environment and install Numba
-------------------------------------------------

RockVerse uses several third-party libraries, and conflicts could potentially
break your existing Python installation. To prevent breaking any working Python
installation, install RockVerse in a dedicated virtual environment.
If you are new to Python, you may want to learn more about
`Python virtual environments <https://docs.python.org/3/tutorial/venv.html>`_
or
`Conda environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
In this tutorial we'll name the virtual environment ``rockverse-env``.

.. note::
    When following this tutorial, choose either conda or pip to manage the
    installation throughout. Using pip requires a C compiler and a working
    MPI implementation with development headers and libraries. Make
    sure command-line ``mpicc`` and ``mpirun`` point to the desired MPI
    installation. If you don't have a working MPI installation in your system,
    then you must use Conda.


Using Conda
~~~~~~~~~~~~

Update Conda (optional but recommended):
if you do not have conda 23.10 or later,
update it to take advantage of the faster
`conda-libmamba-solver plugin <https://conda.github.io/conda-libmamba-solver/user-guide/>`_
and speed up your installation:

.. code-block:: sh

    $ conda update -n base conda

Create your conda environment with Numba and activate:

.. code-block:: sh

    $ conda create --name rockverse-env numba
    $ conda activate rockverse-env


Using Pip
~~~~~~~~~~

Create and activate a virtual environment:

If using Windows:

.. code-block:: sh

    $ cd path\to\my\environment
    $ python -m venv rockverse-env
    $ .\rockverse-env\Scripts\activate

For Linux (example for bash shell):

.. code-block:: sh

    $ cd path/to/my/environment
    $ python -m venv rockverse-env
    $ source ./rockverse-env/bin/activate

Test Numba
~~~~~~~~~~

Test if Numba is working.
Run the following Python code:

.. code-block:: python

    from numba import config, get_thread_id, get_num_threads, njit, prange, threading_layer
    config.THREADING_LAYER = 'workqueue'

    @njit(parallel=True)
    def print_procs():
        print(f'Starting {get_num_threads()} processes...')
        for k in prange(get_num_threads()):
            print(f'proc {get_thread_id()} working!')

    print_procs()
    print("Threading layer chosen: %s" % threading_layer())

You should see something similar to this (number of processes depend on your machine):

.. code-block:: sh

    Starting 12 processes...
    proc 7 working!
    proc 4 working!
    proc 5 working!
    proc 0 working!
    proc 1 working!
    proc 8 working!
    proc 9 working!
    proc 10 working!
    proc 3 working!
    proc 11 working!
    proc 6 working!
    proc 2 working!
    Threading layer chosen: workqueue


2. Install Numba support for GPUs (optional)
--------------------------------------------

RockVerse calculations can be greatly enhanced using GPUs.
If you have GPUs available, take a look at
`Numba's documentation for supported GPUs <https://numba.readthedocs.io/en/stable/cuda/overview.html#supported-gpus>`_.


Using Conda
~~~~~~~~~~~~

For CUDA 12, install ``cuda-nvcc`` and ``cuda-nvrtc``:

.. code-block:: sh

    $ conda install -c conda-forge cuda-nvcc cuda-nvrtc

For CUDA 11, ``cudatoolkit`` is required:

.. code-block:: sh

    $ conda install -c conda-forge cudatoolkit


Using Pip
~~~~~~~~~~

Install the NVIDIA bindings with

.. code-block:: sh

    $ pip install cuda-python

Set the environment variable for Numba:

.. code-block:: sh

    export NUMBA_CUDA_USE_NVIDIA_BINDING="1"  # For Linux
    set NUMBA_CUDA_USE_NVIDIA_BINDING="1"  # For Windows

If using specific CUDA versions, set also CUDA_HOME:

.. code-block:: sh

    export CUDA_HOME=/path/to/cuda  # For Linux
    set CUDA_HOME=C:\path\to\cuda  # For Windows


Test Numba access to GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following Python code:

.. code-block:: python

    from numba import cuda
    print(list(cuda.gpus))

You should see a list of available devices (machine in this example has 8 GPUs):

.. code-block:: sh

    [<numba.cuda.cudadrv.devices._DeviceContextManager object at 0x7f2324fc94c0>,
    <numba.cuda.cudadrv.devices._DeviceContextManager object at 0x7f2324fc81d0>,
    <numba.cuda.cudadrv.devices._DeviceContextManager object at 0x7f2324fc87d0>,
    <numba.cuda.cudadrv.devices._DeviceContextManager object at 0x7f2324fcb080>,
    <numba.cuda.cudadrv.devices._DeviceContextManager object at 0x7f2324fc94f0>,
    <numba.cuda.cudadrv.devices._DeviceContextManager object at 0x7f2324fcb320>,
    <numba.cuda.cudadrv.devices._DeviceContextManager object at 0x7f2324fcb0b0>,
    <numba.cuda.cudadrv.devices._DeviceContextManager object at 0x7f2324fcb1d0>]

If you get empty list or errors Numba cannot access your GPU devices.









3. Configure MPI
----------------

If you have a working MPI installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure command-line ``mpicc`` and ``mpirun`` point to the right MPI installation
(such as in cluster computers through `environment modules <https://modules.sourceforge.net/>`_).
If using conda, run

.. code-block:: sh

    $ conda install -c conda-forge mpi4py

If using pip, run

.. code-block:: sh

    $ pip install --no-cache-dir mpi4py


If you do not have a working MPI installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case you must use conda. There are four MPI implementations available on conda-forge:

1. ``openmpi``: installs `Open MPI <https://www.open-mpi.org/>`_  (Linux and macOS);
2. ``mpich``: installs `MPICH <https://www.mpich.org/>`_  (Linux and macOS);
3. ``impi_rt``: installs `Intel MPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html>`_ (Linux and Windows);
4. ``msmpi``: installs `Microsoft MPI <https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_ (Windows).

Pick your favorite and run ONE of the following commands:

.. code-block:: sh

    $ # Choose one!
    $ conda install -c conda-forge mpi4py openmpi
    $ conda install -c conda-forge mpi4py mpich
    $ conda install -c conda-forge mpi4py impi_rt
    $ conda install -c conda-forge mpi4py msmpi


Test if MPI is working
~~~~~~~~~~~~~~~~~~~~~~

Test the MPI installation:

.. code-block:: sh

    $ mpiexec -n 5 python -m mpi4py.bench helloworld

or

.. code-block:: sh

    $ mpirun -n 5 python -m mpi4py.bench helloworld

depending on your installation. You should get an output similar to this
('localhost' will be the hostname in your machine):

.. code-block:: sh

    Hello, World! I am process 0 of 5 on localhost.
    Hello, World! I am process 1 of 5 on localhost.
    Hello, World! I am process 2 of 5 on localhost.
    Hello, World! I am process 3 of 5 on localhost.
    Hello, World! I am process 4 of 5 on localhost.




4. Install RockVerse
--------------------

If Numba and MPI are working in your virtual environment, install RockVerse:

Install the latest stable version from PyPI:

.. code-block:: sh

    pip install rockverse

or via conda:

.. code-block:: sh

    conda install -c conda-forge rockverse

To install the latest development version of RockVerse, you can use pip with the latest GitHub main:

$ pip install git+https://github.com/rodolfovictor/rockverse.git


To work with RockVerse source code in development, ``cd`` to the path
where you want to clone the repository and install from GitHub:

.. code-block:: sh

    $ git clone https://github.com/rodolfovictor/rockverse.git
    $ cd rockverse
    $ pip install -e .

Now run a quick test:

.. code-block:: sh

    $ python -c "import rockverse; print(f'RockVerse {rockverse.__version__} successfully installed!')"

If you see "RockVerse X.X.X successfully installed!" printed, we are good to go!


5. Updating RockVerse
---------------------

To ensure that you are using the latest version of RockVerse with new features,
improvements, and bug fixes, you can easily update it using either pip or conda.

Using Pip
~~~~~~~~~

.. code-block:: sh

    $ pip install --upgrade rockverse

This command will check for any available updates on PyPI and install them.

Using Conda
~~~~~~~~~~~

If you installed RockVerse using Conda, you can update it by running:

.. code-block:: sh

    $ conda update -c conda-forge rockverse

This command will update RockVerse to the latest version available in the
conda-forge channel.
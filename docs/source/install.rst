Installation
============

RockVerse uses several third-party libraries, and conflicts could potentially break your
existing Python installation. I strongly advise you to create a dedicated virtual
environment. If you are new to Python, you may want to learn more about
`Python virtual environments <https://docs.python.org/3/tutorial/venv.html>`_
or
`Conda environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Installation typically comprehends the following steps:

1. Prepare your virtual environment: this will prevent breaking any working Python intallation already in the system.

2. Install Numba:
   `Numba <https://numba.pydata.org/>`_ is a critical dependency for RockVerse, enabling high-performance
   just-in-time compilation for numerical operations. It optimizes computation on both CPUs and GPUs,
   ensuring RockVerse operates efficiently.

3. (Optional) Install Numba support for GPUs: if you have GPUs available, take a look at
   `Numba's documentationfor supported GPUs <https://numba.readthedocs.io/en/stable/cuda/overview.html#supported-gpus>`_.

4. Install MPI:
   RockVerse leverages MPI (Message Passing Interface) through
   `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_,
   enabling distributed parallel computations across multiple processors or nodes.

Below are the steps to get RockVerse to work on your system using either conda or pip.










Pre-installation using conda
----------------------------

.. note::
    Conda is the only option if you don't have a working MPI installation on your system.

Optional (but strongly recommended!): if you do not have a recent conda (23.10 or later),
update it to take advantage of the faster
`conda-libmamba-solver plugin <https://conda.github.io/conda-libmamba-solver/user-guide/>`_
and speed up your installation:

.. code-block:: sh

    $ conda update -n base conda

Create and activate your conda environment (let's name this environment rockverse-env):

.. code-block:: sh

    $ conda create --name rockverse-env
    $ conda activate rockverse-env

**If you have a working MPI installation**, (for example in cluster computers through
`environment modules <https://modules.sourceforge.net/>`_),
make sure ``mpicc`` and ``mpirun`` point to the right MPI installation and just run:

.. code-block:: sh

    $ conda install -c conda-forge numba mpi4py

**If you do not have a working MPI installation**,
there are four MPI implementations available on conda-forge that can be installed
with conda:

1. ``openmpi``: installs `Open MPI <https://www.open-mpi.org/>`_  (Linux and macOS);
2. ``mpich``: installs `MPICH <https://www.mpich.org/>`_  (Linux and macOS);
3. ``impi_rt``: installs `Intel MPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html>`_ (Linux and Windows);
4. ``msmpi``: installs `Microsoft MPI <https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_ (Windows).

Pick your favorite (say ``openmpi``) and create the new conda environment with ``numba`` and ``mpi4py``:

.. code-block:: sh

    $ conda install -c conda-forge numba mpi4py openmpi

(Optional) Enable Numba access to GPUs:
For CUDA 12, install ``cuda-nvcc`` and ``cuda-nvrtc``:

.. code-block:: sh

    $ conda install -c conda-forge cuda-nvcc cuda-nvrtc

For CUDA 11, ``cudatoolkit`` is required:

.. code-block:: sh

    $ conda install -c conda-forge cudatoolkit










Pre-installation using pip
--------------------------

.. note::
    Unlike conda, installing ``mpi4py`` using pip requires a C compiler and a working MPI
    implementation with development headers and libraries. Make sure ``mpicc`` and ``mpirun``
    point to the desired MPI installation.

Create and activate the virtual environment (again, let's call it rockverse-env):

If using Windows,

.. code-block:: sh

    $ cd path\to\my\environment
    $ python -m venv rockverse-env
    $ .\rockverse-env\Scripts\activate

for Linux (example for bash shell):

.. code-block:: sh

    $ cd path/to/my/environment
    python -m venv rockverse-env
    $ source ./rockverse-env/bin/activate

Install numba and mpi4py:

.. code-block:: sh

    $ pip install --no-cache-dir numba mpi4py

(Optional) enable Numba access to GPUs:
Install the NVIDIA bindings with

.. code-block:: sh

    $ pip install cuda-python

You'll need to set the environment variable ``NUMBA_CUDA_USE_NVIDIA_BINDING`` to ``"1"``.
If you want to use specific CUDA versions, set also the environment variable
``CUDA_HOME`` to the directory of the installed CUDA toolkit (e.g. ``/home/user/cuda-12``).










Test your pre-installation
--------------------------

Numba parallelization
^^^^^^^^^^^^^^^^^^^^^

Test if Numba shared memory parallelization is working.
Run the following Python code:

.. code-block:: python

    from numba import config, get_thread_id, get_num_threads, njit, prange, threading_layer
    config.THREADING_LAYER = 'threadsafe'

    @njit(parallel=True)
    def print_procs():
        print(f'Starting {get_num_threads()} processes...')
        for k in prange(get_num_threads()):
            print(f'proc {get_thread_id()} working!')

    print_procs()
    print("Threading layer chosen: %s" % threading_layer())

You should see something similar to

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
    Threading layer chosen: omp


Numba access to GPUs (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the following Python code:

.. code-block:: python

    from numba import cuda
    print(list(cuda.gpus))

Everything should be working if you get a list of sevices similar to this:

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


MPI configuration
^^^^^^^^^^^^^^^^^

Quickly test the MPI installation:

.. code-block:: sh

    $ mpiexec -n 5 python -m mpi4py.bench helloworld

or

.. code-block:: sh

    $ mpirun -n 5 python -m mpi4py.bench helloworld

depending on your installation. You should get an output similar to this:

.. code-block:: sh

    Hello, World! I am process 0 of 5 on localhost.
    Hello, World! I am process 1 of 5 on localhost.
    Hello, World! I am process 2 of 5 on localhost.
    Hello, World! I am process 3 of 5 on localhost.
    Hello, World! I am process 4 of 5 on localhost.

If you encounter errors during MPI installation or execution:

- Ensure the MPI implementation (Open MPI or MPICH) is installed and correctly added to your system's PATH
  (if you used one of the above conda options it should be automatically done).
- Verify that the `mpi4py` library is installed in your current Python environment.
- Check for conflicts between your MPI implementation and `mpi4py`. Some versions of `mpi4py` may require specific versions of MPI.










Install RockVerse
-----------------

If the tests above were successful, install RockVerse (it may take a while):

.. code-block:: sh

    pip install rockverse

Run a quick test

.. code-block:: sh

    $ python -c "import rockverse; print('RockVerse installed successfully!')"

If you see "RockVerse installed successfully!" printed, we are good to go!

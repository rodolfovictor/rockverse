"Manages runtime settings for the RockVerse library."

import copy
from mpi4py import MPI
from numba import cuda
from contextlib import contextmanager
from rockverse import _assert
from rockverse.errors import collective_raise, CustomCollectiveException
from rockverse.configure.orthogonal_viewer import ORTHOGONAL_VIEWER as _ORTHOGONAL_VIEWER
from rockverse.configure.latex_strings import LATEX_STRINGS as _LATEX_STRINGS

def _split_key(key):
    return key.split('.')


class Config():
    """
    Manages runtime settings.
    """

    def __init__(self):

        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_rank = MPI.COMM_WORLD.Get_rank()
        self._mpi_nprocs = MPI.COMM_WORLD.Get_size()
        self._processor_name = MPI.Get_processor_name()
        self._dict = {}

        if not cuda.is_available():
            self._gpus = []
            self._dict['selected_gpus'] = []
        else:
            self._gpus = cuda.gpus
            self._dict['selected_gpus'] = list(range(len(self._gpus)))

        self._dict['latex.strings'] = copy.deepcopy(_LATEX_STRINGS)
        self._dict['orthogonal_viewer'] = copy.deepcopy(_ORTHOGONAL_VIEWER)


    def reset(self):
        self.__init__()


    def __getitem__(self, item):
        return self._dict.__getitem__(item)

    def __setitem__(self, key, value):

        if key == 'selected_gpus':
            _assert.iterable.any_iterable_non_negative_integers('device selection', value)
            if any(k not in range(len(self._gpus)) for k in value):
                if len(self._gpus) == 0:
                    collective_raise(RuntimeError(f'GPU devices are not available.'))
                collective_raise(RuntimeError(
                    f'GPU device indices must be less than {len(self._gpus)}.'))
            self._dict['selected_gpus'] = sorted(set(value))

        else:
            raise ValueError('NO')

    @property
    def mpi_rank(self):
        """
        Returns the rank of the calling process in the MPI communicator.
        """
        return self._mpi_rank

    @property
    def mpi_nprocs(self):
        """
        Returns the total number of processes in the MPI communicator.
        """
        return self._mpi_nprocs

    @property
    def processor_name(self):
        """
        Returns the name of the processor running MPI.
        """
        return self._processor_name

    @property
    def mpi_comm(self):
        """
        Returns the MPI communicator object.
        """
        return self._mpi_comm

    @property
    def available_gpus(self):
        """
        Returns the list of all GPUs available to the processor in ``processor_name``.
        """
        return self._gpus


    def print_available_gpus(self):
        """Prints the list of available GPUs.

        Mimics the output of the `nvidia-smi -L` terminal command.
        If no GPUs are available, a message indicating this is printed.
        """
        if not self._gpus:
            print("GPUs not available.")
        else:
            for g, gpu in enumerate(self._gpus):
                print(f'GPU {g}: {gpu.name.decode()} (UUID: {gpu._device.uuid})')

    def print_selected_gpus(self):
        """
        Prints the list of user-selected GPUs.

        Mimics the output of the `nvidia-smi -L` terminal command.
        If no GPUs are available, a message indicating this is printed.
        """
        if not self._gpus:
            print("GPUs not available.")
        else:
            for g, gpu in enumerate(self._gpus):
                if g in self['selected_gpus']:
                    print(f'GPU {g}: {gpu.name.decode()} (UUID: {gpu._device.uuid})')

    def rank_select_gpu(self):
        """
        Selects the GPU to be used by the current MPI process at runtime.
        The selection is based on the user-select GPUs.

        Returns:
            The index of the selected GPU, or None if no GPUs are available.
        """
        if not self['selected_gpus']:
            return None
        ind = self._mpi_rank % len(self['selected_gpus'])
        gpu_ind = self['selected_gpus'][ind]
        return gpu_ind

    def print_rank_selected_gpu(self):
        """
        Prints the GPU selected for use by the current MPI process.

        Mimics the output of the `nvidia-smi -L` terminal command.
        If no GPUs are available, a message indicating this is printed.
        """
        if not self._gpus:
            print("GPU not available.")
        gpu_ind = self.rank_select_gpu()
        gpu = self._gpus[gpu_ind]
        print(f'GPU {gpu_ind}: {gpu.name.decode()} (UUID: {gpu._device.uuid})')

config = Config()




class config_context():
    """
    Context manager for temporarily modifying the configuration settings.

    This context manager allows for a temporary reassignment of configuration parameters.
    Changes made within the context manager are reverted back to the original settings once
    the context is exited.

    Parameters:
    -----------
    params : dict
        A dictionary of configuration parameters to update.

    Example:
    --------

    Temporarily change the list of allowed GPU devices:

    >>> with config_context({'selected_gpus': [2, 4, 6]}):
    >>>     # Do stuff using these 3 devices
    >>>     my_awesome_function_that_uses_gpu()
    >>>     .
    >>>     .
    >>>     .
    >>> # After the with block, the original configuration is restored.
    """

    def __init__(self, params):
        self.backup_dict = {}
        self.update_dict = {}
        self.update_dict.update(**params)

    def __enter__(self):

        #Backup current configuration
        self.backup_dict = copy.deepcopy(config._dict)
        for k, v in self.update_dict.items():
            config[k] = v

    def __exit__(self, exc_type, exc_value, exc_tb):
        config._dict = copy.deepcopy(self.backup_dict)
        if exc_type is not None:
            collective_raise(CustomCollectiveException(exc_type, exc_value))

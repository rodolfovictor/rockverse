from mpi4py import MPI
from numba import cuda
from rockverse import _assert

class Config():
    """
    Configuration class for managing runtime MPI and GPU settings.

    This class encapsulates the configuration parameters related to
    MPI (Message Passing Interface) and GPU (Graphics Processing Unit)
    availability. It provides properties to access the current MPI rank,
    total number of processes, processor name, available GPUs, and selected GPUs.
    """

    def __init__(self):

        self._mpi_comm = MPI.COMM_WORLD
        self._mpi_rank = MPI.COMM_WORLD.Get_rank()
        self._mpi_nprocs = MPI.COMM_WORLD.Get_size()
        self._processor_name = MPI.Get_processor_name()

        if not cuda.is_available():
            self._gpus = []
            self._selected_gpus = []
        else:
            self._gpus = cuda.gpus
            self._selected_gpus = list(range(len(self._gpus)))

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

    @property
    def selected_gpus(self):
        """Get or set the list of selected GPU indices.

        These indices set the devices that are allowed to be used by
        RockVerse during runtime.

        You can set this list at runtime by providing a list of integers in
        range('total number of available GPUs').

        Examples
        --------
            >>> # Get the currently selected GPU indices
            >>> current_selected = config.selected_gpus

            >>> # Set the selected GPUs to the first and second GPUs available
            >>> config.selected_gpus = [0, 1]

            >>> # You can use any iterable
            >>> config.selected_gpus = (0, 1, 2)
            >>> config.selected_gpus = {0, 1, 2}
            >>> config.selected_gpus = range(2)


            >>> # Attempting to set selected GPUs to an invalid index will raise an error
            >>> try:
            >>>     config.selected_gpus = [0, 5]  # Assuming only 3 GPUs are available
            >>> except RuntimeError as e:
            >>>    print(e)  # Output: GPU device indices must be less than 3.

            >>> # Setting selected GPUs to an empty list means no GPU will be used
            >>> config.selected_gpus = []
        """
        return self._selected_gpus

    @selected_gpus.setter
    def selected_gpus(self, v):
        _assert.iterable.any_iterable_non_negative_integers('device selection', v)
        if any(k not in range(len(self._gpus)) for k in v):
            _assert.collective_raise(RuntimeError(
                f'GPU device indices must be less than {len(self._gpus)}.'))
        self._selected_gpus = sorted(v)

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
                if g in self._selected_gpus:
                    print(f'GPU {g}: {gpu.name.decode()} (UUID: {gpu._device.uuid})')

    def rank_select_gpu(self):
        """
        Selects the GPU to be used by the current MPI process at runtime.
        The selection is based on the user-select GPUs.

        Returns:
            The index of the selected GPU, or None if no GPUs are available.
        """
        if not self._selected_gpus:
            return None
        ind = self._mpi_rank % len(self._selected_gpus)
        gpu_ind = self._selected_gpus[ind]
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
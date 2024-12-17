
from mpi4py import MPI
from numba import cuda
from numba import config as numba_config
from rockverse import _assert

class Config(dict):

    def __init__(self):

        self['MPI'] = {
            'mpi_comm': MPI.COMM_WORLD,
            'mpi_rank': MPI.COMM_WORLD.Get_rank(),
            'mpi_nprocs': MPI.COMM_WORLD.Get_size(),
            'processor_name': MPI.Get_processor_name()
            }

        if not cuda.is_available():
            self['GPU'] = None
        else:
            self['GPU'] = {'available_devices': {}}
            for g, gpu in enumerate(cuda.gpus):
                self['GPU']['available_devices'][g] = gpu.name.decode()

        self['collective_getitem'] = False

    @property
    def mpi_rank(self):
        return self['MPI']['mpi_rank']

    @property
    def mpi_nprocs(self):
        return self['MPI']['mpi_nprocs']

    @property
    def processor_name(self):
        return self['MPI']['processor_name']

    @property
    def mpi_comm(self):
        return self['MPI']['mpi_comm']

    @property
    def collective_getitem(self):
        return self['collective_getitem']

    @collective_getitem.setter
    def collective_getitem(self, v):
        if v not in (True, False):
            _assert.collective_raise(ValueError("Expected boolean for collective_getitem."))
        self['collective_getitem'] = v

    def exec_mode(self):
        if self['MPI']['mpi_nprocs'] > 1:
            run_mode = 'MPI'
        else:
            run_mode = 'OMP'

        if self['GPU'] is None or len(self['GPU']['available_devices']) == 0:
            device_mode = 'CPU'
        else:
            device_mode = 'GPU'

        return run_mode, device_mode

config = Config()
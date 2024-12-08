__version__ = "0.0.7d"

from rockverse._utils.logo import make_logo

import rockverse.digitalrock as digitalrock

from rockverse.rc import RcParams
rcParams = RcParams()

from mpi4py import MPI
from numba import cuda
_launch_params = {
    'MPI': {
        'comm': MPI.COMM_WORLD,
        'mpi_rank': MPI.COMM_WORLD.Get_rank(),
        'mpi_nprocs': MPI.COMM_WORLD.Get_size(),
        'processor_name': MPI.Get_processor_name()
        },
    'gpus': {},
    }
if cuda.is_available():
    for g, gpu in enumerate(cuda.gpus):
        _launch_params['gpus'][g] = gpu.name.decode()
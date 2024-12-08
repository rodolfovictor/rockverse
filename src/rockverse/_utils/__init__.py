from datetime import datetime
from tqdm import tqdm

from mpi4py import MPI


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

def datetimenow():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def rvtqdm(*args, **kwargs):
    if 'disable' not in kwargs:
        kwargs['disable'] = mpi_rank!=0
    if 'ascii' not in kwargs:
        kwargs['ascii'] = ' >'
    datestr = datetimenow()
    if 'desc' in kwargs:
        kwargs['desc'] = f"{datestr} {kwargs['desc']}"
    else:
        kwargs['desc'] = f"{datestr} "
    return tqdm(*args, **kwargs)


# decorator that copies docstrings
def copy_docstring(func_to_copy_from):
    def decorator(func):
        func.__doc__ = func_to_copy_from.__doc__
        return func
    return decorator
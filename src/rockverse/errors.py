'''
errors module cannot call rockverse config due to circular imports.
Call directly MPI from mpi4py
'''

import sys
from contextlib import contextmanager
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

def collective_raise(e):
    if mpi_rank == 0:
        raise e
    else:
        sys.exit(1)

class CustomCollectiveException(Exception):
    def __init__(self, name, message):
        super().__init__(message)
        self.__class__.__name__ = name

@contextmanager
def collective_only_rank0_runs(id=''):
    """
    Allows only rank 0 to run the code block, capturing exception and
    calling collective_raise in case of errors.

    How to use
    ----------

    >>> with rank0_runs():
    >>>     if mpi_rank == 0: #<-- THIS IS VERY IMPORTANT!
                <block to be run only by rank0>
    >>> <continue the code>
    """
    error_msg = ''
    try:
        yield
    except Exception as e:
        error_msg = f"{e.__class__.__name__}: {e}"
    error_msg = comm.bcast(error_msg, root=0)
    if error_msg:
        name, msg = error_msg.split(':')[0].strip(), ''.join(error_msg.split(':')[1:]).strip()
        collective_raise(CustomCollectiveException(name, msg))

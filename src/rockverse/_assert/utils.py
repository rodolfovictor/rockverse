import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

def collective_raise(e):
    if mpi_rank == 0:
        raise e
    sys.exit(1)

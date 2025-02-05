'''
errors module cannot call rockverse config due to circular imports.
Call directly MPI from mpi4py
'''
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

def collective_raise(e):
    if mpi_rank == 0:
        raise e
    else:
        sys.exit(1)

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


def resolve_index(self, index):
    """
    Resolve index by expanding ellipses and converting boolean masks.
    """
    if not isinstance(index, tuple):
        index = (index,)  # Ensure the index is always a tuple

    # Step 1: Expand ellipsis (...) into slices
    index = expand_ellipsis(self, index)

    # Step 2: Convert boolean masks into integer indices
    resolved_index = []
    for _, idx in enumerate(index):
        if isinstance(idx, (np.ndarray, list)) and np.issubdtype(np.array(idx).dtype, np.bool_):
            # Ensure shape compatibility
            if np.array(idx).shape != self.shape:
                raise IndexError(f"Boolean index shape {idx.shape} does not match array shape {self.shape}")
            # Convert boolean mask to integer indices
            boolean_indices = np.nonzero(idx)
            resolved_index.extend(boolean_indices[i] for i in range(len(boolean_indices)))
        else:
            resolved_index.append(idx)

    return tuple(resolved_index)


def expand_ellipsis(self, index):
    """
    Replace Ellipsis (...) in the index with the correct number of slices.
    """
    result = []
    ellipsis_count = index.count(Ellipsis)

    if ellipsis_count > 1:
        raise IndexError("An index can only have a single ellipsis ('...').")

    ndim = self.ndim  # Total dimensions of the array
    for item in index:
        if item is Ellipsis:
            # Calculate how many slices are needed to fill the ellipsis
            num_missing_slices = ndim - len(result) - (len(index) - index.index(Ellipsis) - 1)
            result.extend([slice(None)] * num_missing_slices)
        else:
            result.append(item)

    # Fill remaining dimensions with slices
    while len(result) < ndim:
        result.append(slice(None))

    return tuple(result)


def collective_print(msg, print_time=True):
    datestr = datetimenow() + ' ' if print_time else ''
    if mpi_rank == 0:
        print(f"{datestr}{msg}", flush=True)
    comm.barrier()
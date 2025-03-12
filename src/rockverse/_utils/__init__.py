import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
from rockverse.errors import collective_raise
from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

def datetimenow():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def rvtqdm(*args, **kwargs):
    if 'disable' not in kwargs:
        kwargs['disable'] = mpi_rank!=0
    if 'ascii' not in kwargs:
        kwargs['ascii'] = ' >'
    if 'file' not in kwargs:
        kwargs['file'] = sys.stdout

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


def expand_ellipsis(array_like, index):
    """
    Replace Ellipsis (...) in the index with the correct number of slices.
    """
    result = []
    ellipsis_count = index.count(Ellipsis)

    if ellipsis_count > 1:
        raise IndexError("An index can only have a single ellipsis ('...').")

    ndim = array_like.ndim  # Total dimensions of the array
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


def resolve_index(array_like, index):
    """
    Resolve index by expanding ellipses and converting boolean masks.
    """
    if not isinstance(index, tuple):
        index = (index,)  # Ensure the index is always a tuple

    # Step 1: Expand ellipsis (...) into slices
    index = expand_ellipsis(array_like, index)

    # Step 2: Convert boolean masks into integer indices
    resolved_index = []
    for _, idx in enumerate(index):
        if isinstance(idx, (np.ndarray, list)) and np.issubdtype(np.array(idx).dtype, np.bool_):
            # Ensure shape compatibility
            if np.array(idx).shape != array_like.shape:
                raise IndexError(f"Boolean index shape {idx.shape} does not match array shape {array_like.shape}")
            # Convert boolean mask to integer indices
            boolean_indices = np.nonzero(idx)
            resolved_index.extend(boolean_indices[i] for i in range(len(boolean_indices)))
        else:
            resolved_index.append(idx)

    return tuple(resolved_index)


def index_bounding_box(array_like, index):
    resolved_index = resolve_index(array_like, index)
    bbox = []
    shape = array_like.shape
    for i, sl in enumerate(resolved_index):
        if np.dtype(type(sl)).kind in 'ui':
            bbox.append((sl, sl+1))
        elif isinstance(sl, slice):
            if sl.step is not None and sl.step <= 0:
                collective_raise(IndexError('Slice step must be positive.'))
            aux = range(sl.start if sl.start is not None else 0,
                        sl.stop if sl.stop is not None else shape[i],
                        sl.step if sl.step is not None else 1)
            bbox.append((min(aux), max(aux)+1))
        elif isinstance(sl, (list, tuple)):
            bbox.append((min(sl), max(sl)+1))
        else:
            collective_raise(IndexError('Invalid index.'))
    return bbox


def collective_print(msg, print_time=True):
    datestr = datetimenow() + ' ' if print_time else ''
    if mpi_rank == 0:
        print(f"{datestr}{msg}", flush=True)
    comm.barrier()


def auto_chunk_3d(shape, n):

    cdata_shape = np.zeros((n**3, 4), dtype='int64')
    ind = 0
    for i in range(1, n+1):
        for j in range(1, n+1):
            for k in range(1, n+1):
                cdata_shape[ind, :] = [i, j, k, i*j*k]
                ind += 1
    cdata_shape =  cdata_shape[cdata_shape[:, 3]==n, :]

    for i in range(cdata_shape.shape[0]):
        cdata_shape[i, 3] = (abs(cdata_shape[i, 0]-cdata_shape[i, 1])
                             +abs(cdata_shape[i, 0]-cdata_shape[i, 2])
                             +abs(cdata_shape[i, 1]-cdata_shape[i, 2]))
    ind = np.argmin(cdata_shape[:, 3])

    chunks = tuple(int(i) for i in np.ceil(np.array(shape).astype(float)/cdata_shape[ind, :3].astype(float)))

    return chunks

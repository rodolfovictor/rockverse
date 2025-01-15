"""
These are methods for High-Performance Computing (HPC) operations for
voxel by voxel image manipulation. They leverage both CPU and GPU
parallelization techniques to perform a wide range of mathematical
operations such as element-wise arithmetic and logical operations.
"""

import numpy as np
from rockverse._utils import rvtqdm
from numba import njit, cuda
import rockverse._assert as _assert
from rockverse.config import config

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()


def _define_grid(chunk_size):
    gpu = cuda.get_current_device()
    sumthreads = int(np.log(gpu.MAX_THREADS_PER_BLOCK)/np.log(2))
    threadsperblock = [1 for _ in range(len(chunk_size))]
    fill = np.argsort((gpu.MAX_BLOCK_DIM_X, gpu.MAX_BLOCK_DIM_Y, gpu.MAX_BLOCK_DIM_Z))[::-1]
    i = 0
    while np.sum(threadsperblock) < sumthreads:
        threadsperblock[fill[i]] += 1
        i = (i+1) % len(chunk_size)
    threadsperblock = tuple(2**k for k in threadsperblock)
    blockspergrid = tuple(int(np.ceil(N/n)) for N, n in zip(chunk_size, threadsperblock))
    return threadsperblock, blockspergrid


@njit()
def _apply_mask_cpu(skip, mask):
    nx, ny, nz = skip.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    skip[i, j, k] = True


@cuda.jit()
def _apply_mask_gpu(skip, mask):
    nx, ny, nz = skip.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if mask[i, j, k]:
        skip[i, j, k] = True


@njit()
def _apply_segmentation_cpu(skip, segm, phases):
    nx, ny, nz = skip.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if segm[i, j, k] not in phases:
                    skip[i, j, k] = True


@cuda.jit()
def _apply_segmentation_gpu(skip, segm, phases):
    nx, ny, nz = skip.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    not_in_segm = True
    for p in phases:
        if p == segm[i, j, k]:
            not_in_segm = False
            break
    if not_in_segm:
        skip[i, j, k] = True


@njit()
def _copy_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array2[i, j, k]


@cuda.jit()
def _copy_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array2[i, j, k]


@njit()
def _add_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] + array2[i, j, k]


@cuda.jit()
def _add_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] + array2[i, j, k]


@njit()
def _subtract_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] - array2[i, j, k]


@cuda.jit()
def _subtract_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] - array2[i, j, k]


@njit()
def _multiply_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] * array2[i, j, k]


@cuda.jit()
def _multiply_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] * array2[i, j, k]


@njit()
def _divide_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] / array2[i, j, k]


@cuda.jit()
def _divide_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] / array2[i, j, k]


@njit()
def _logical_and_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] and array2[i, j, k]


@cuda.jit()
def _logical_and_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] and array2[i, j, k]


@njit()
def _logical_or_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] or array2[i, j, k]


@cuda.jit()
def _logical_or_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] or array2[i, j, k]


@njit()
def _logical_xor_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] != array2[i, j, k]


@cuda.jit()
def _logical_xor_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] != array2[i, j, k]


@njit()
def _min_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = min(array1[i, j, k], array2[i, j, k])


@cuda.jit()
def _min_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = min(array1[i, j, k], array2[i, j, k])


@njit()
def _max_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = max(array1[i, j, k], array2[i, j, k])


@cuda.jit()
def _max_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = max(array1[i, j, k], array2[i, j, k])


@njit()
def _avg_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = 0.5*(array1[i, j, k] + array2[i, j, k])


@cuda.jit()
def _avg_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = 0.5*(array1[i, j, k] + array2[i, j, k])


@njit()
def _absdiff_array_cpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = abs(array1[i, j, k] - array2[i, j, k])


@cuda.jit()
def _absdiff_array_gpu(array1, array2, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = abs(array1[i, j, k] - array2[i, j, k])


@njit()
def _set_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = value


@cuda.jit()
def _set_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = value


@njit()
def _add_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] += value


@cuda.jit()
def _add_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] += value


@njit()
def _subtract_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] -= value


@cuda.jit()
def _subtract_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] -= value


@njit()
def _multiply_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] *= value


@cuda.jit()
def _multiply_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] *= value


@njit()
def _divide_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] /= value


@cuda.jit()
def _divide_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] /= value


@njit()
def _and_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] and value


@cuda.jit()
def _and_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] and value


@njit()
def _or_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] or value


@cuda.jit()
def _or_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] or value


@njit()
def _xor_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = array1[i, j, k] != value


@cuda.jit()
def _xor_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = array1[i, j, k] != value


@njit()
def _min_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = min(array1[i, j, k], value)


@cuda.jit()
def _min_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = min(array1[i, j, k], value)


@njit()
def _max_value_cpu(array1, value, skip):
    nx, ny, nz = array1.shape
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if skip[i, j, k]:
                    continue
                array1[i, j, k] = max(array1[i, j, k], value)


@cuda.jit()
def _max_value_gpu(array1, value, skip):
    nx, ny, nz = array1.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    if skip[i, j, k]:
        return
    array1[i, j, k] = max(array1[i, j, k], value)

def _array_math(array1,
                array2=None,
                value=None,
                op=None,
                region=None,
                mask=None,
                segmentation=None,
                phases=None):

    #Array assertions must be done at the calling function
    if value is not None:
        _assert.instance('value', value, 'boolean, integer, or float', (bool, int, float))

    if region is not None:
        _assert.rockverse_instance(region, 'region', ('Region',))

    array1.check_mask_and_segmentation(mask=mask, segmentation=segmentation)

    if phases is not None:
        _assert.iterable.any_iterable_non_negative_integers('phases', phases)

    arrays = [array1,]
    if array2 is not None:
        arrays.append(array2)
    if mask is not None:
        arrays.append(mask)
    if segmentation is not None:
        arrays.append(segmentation)

    _assert.same_shape('Arrays', arrays)
    _assert.same_voxel_origin('Arrays', arrays)
    _assert.same_voxel_length('Arrays', arrays)
    _assert.same_voxel_unit('Arrays', arrays)
    ox, oy, oz = array1.voxel_origin
    hx, hy, hz = array1.voxel_length


    device_index = config.rank_select_gpu()
    use_gpu = False if device_index is None else True
    if phases is not None:
        cphases = np.array([k for k in phases], dtype='u8')
        if use_gpu:
            with config._gpus[device_index]:
                dphases = cuda.to_device(cphases)

    desc, OP = '<none>', None
    OPType = None
    #array-array operations
    if array2 is not None and value is None:
        OPType = 'array-array'
        if op == 'copy':
            desc, OP = 'Copy', _copy_array_gpu if use_gpu else _copy_array_cpu
        elif op == 'add':
            desc, OP = 'Add', _add_array_gpu if use_gpu else _add_array_cpu
        elif op == 'subtract':
            desc, OP = 'Subtract', _subtract_array_gpu if use_gpu else _subtract_array_cpu
        elif op == 'multiply':
            desc, OP = 'Multiply', _multiply_array_gpu if use_gpu else _multiply_array_cpu
        elif op == 'divide':
            desc, OP = 'Divide', _divide_array_gpu if use_gpu else _divide_array_cpu
        elif op == 'logical and':
            desc, OP = 'Logical and', _logical_and_array_gpu if use_gpu else _logical_and_array_cpu
        elif op == 'logical or':
            desc, OP = 'Logical or', _logical_or_array_gpu if use_gpu else _logical_or_array_cpu
        elif op == 'logical xor':
            desc, OP = 'Logical xor', _logical_xor_array_gpu if use_gpu else _logical_xor_array_cpu
        elif op == 'min':
            desc, OP = 'Minimium', _min_array_gpu if use_gpu else _min_array_cpu
        elif op == 'max':
            desc, OP = 'Maximum', _max_array_gpu if use_gpu else _max_array_cpu
        elif op == 'average':
            desc, OP = 'Average', _avg_array_gpu if use_gpu else _avg_array_cpu
        elif op == 'absolute difference':
            desc, OP = 'Absolute difference', _absdiff_array_gpu if use_gpu else _absdiff_array_cpu
        else:
            _assert.collective_raise(ValueError(f"Invalid operation '{op}'"))

    #array-constant operations
    elif array2 is None and value is not None:
        OPType = 'array-constant'
        if op == 'set':
            desc, OP = 'Set', _set_value_gpu if use_gpu else _set_value_cpu
        elif op == 'add':
            desc, OP = 'Add', _add_value_gpu if use_gpu else _add_value_cpu
        elif op == 'subtract':
            desc, OP = 'Subtract', _subtract_value_gpu if use_gpu else _subtract_value_cpu
        elif op == 'multiply':
            desc, OP = 'Multiply', _multiply_value_gpu if use_gpu else _multiply_value_cpu
        elif op == 'divide':
            desc, OP = 'Divide', _divide_value_gpu if use_gpu else _divide_value_cpu
        elif op == 'logical and':
            desc, OP = 'Logical and', _and_value_gpu if use_gpu else _and_value_cpu
        elif op == 'logical or':
            desc, OP = 'Logical or', _or_value_gpu if use_gpu else _or_value_cpu
        elif op == 'logical xor':
            desc, OP = 'Logical xor', _xor_value_gpu if use_gpu else _xor_value_cpu
        elif op == 'min':
            desc, OP = 'Min', _min_value_gpu if use_gpu else _min_value_cpu
        elif op == 'max':
            desc, OP = 'Max', _max_value_gpu if use_gpu else _max_value_cpu
        else:
            _assert.collective_raise(ValueError(f"Invalid operation '{op}'"))

    else:
        raise ValueError('array2 and value cannot be None at the same time.')


    if array1.field_name:
        desc = f'({array1.field_name}) {desc}'
    dtype = array1.dtype
    for block_id in rvtqdm(range(array1.nchunks), desc=desc, unit='chunk'):
        if block_id % mpi_nprocs != mpi_rank:
            continue
        box, bex, boy, bey, boz, bez = array1.chunk_slice_indices(block_id)
        carray1 = array1[box:bex, boy:bey, boz:bez].copy()
        skip = np.zeros_like(carray1, dtype='|b1') #voxels that won't be processed
        if array2 is not None:
            carray2 = array2[box:bex, boy:bey, boz:bez].astype(dtype)
        if mask is not None:
            cmask = mask[box:bex, boy:bey, boz:bez].copy()
        if segmentation is not None:
            csegmentation = segmentation[box:bex, boy:bey, boz:bez].astype('u8')

        if use_gpu:
            threadsperblock, blockspergrid = _define_grid(carray1.shape)
            with config._gpus[device_index]:
                darray1 = cuda.to_device(carray1)
                dskip = cuda.to_device(skip)
                if array2 is not None:
                    darray2 = cuda.to_device(carray2)
                if region is not None:
                    region.mask_chunk_gpu[blockspergrid, threadsperblock](
                        dskip, ox, oy, oz, hx, hy, hz, box, boy, boz)
                if mask is not None:
                    dmask = cuda.to_device(cmask)
                    _apply_mask_gpu[blockspergrid, threadsperblock](dskip, dmask)
                if segmentation is not None:
                    dsegmentation = cuda.to_device(csegmentation)
                    _apply_segmentation_gpu[blockspergrid, threadsperblock](
                        dskip, dsegmentation, dphases)

                if OPType == 'array-array':
                    OP[blockspergrid, threadsperblock](darray1, darray2, dskip)
                else:
                    OP[blockspergrid, threadsperblock](darray1, value, dskip)
                carray1 = darray1.copy_to_host()
        else:
            if region is not None:
                region.mask_chunk_cpu(skip, ox, oy, oz, hx, hy, hz, box, boy, boz)
            if mask is not None:
                _apply_mask_cpu(dskip, dmask)
            if segmentation is not None and phases is not None:
                _apply_segmentation_cpu(skip, csegmentation, cphases)

            if OPType == 'array-array':
                OP(carray1, carray2, skip)
            else:
                OP(carray1, value, skip)

        array1[box:bex, boy:bey, boz:bez] = carray1.copy()
    comm.barrier()

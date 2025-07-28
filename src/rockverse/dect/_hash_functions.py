"""
Contains hash functions to verify the integrity and consistency of data and
intermediate results in Dual Energy Computed Tomography (DECT) processing.

Note:
    All functions are designed to work with the DECT Group class and its attributes.
"""

import hashlib
import numpy as np
from rockverse.errors import collective_only_rank0_runs
from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

from rockverse._utils import rvtqdm


def hash_array(array, name):
    """
    Calculate the MD5 hash of a voxel image, processing in blocks for memory efficiency.

    Parameters:
    array (VoxelImage): The 3D array to be hashed.
    name (str): Name of the array for progress bar display.

    Returns:
    str: The hexadecimal digest of the MD5 hash.
    """
    if array is None:
        return ''

    local_md5 = ['']*array.nchunks
    for block_id in rvtqdm(range(array.nchunks), desc=f'Hashing {name}', unit='chunk'):
        chunk_slices = array.chunk_slice_indices(block_id)
        if block_id % mpi_nprocs == mpi_rank:
            block = array[chunk_slices]
            local_md5[block_id] = hashlib.md5(block).hexdigest()
    comm.barrier()
    global_md5 = hashlib.md5()
    for block_id in range(array.nchunks):
        block_hexdigest = comm.bcast(local_md5[block_id], root=(block_id % mpi_nprocs))
        global_md5.update(block_hexdigest.encode('ascii'))
    return global_md5.hexdigest()


def hash_input_data(group):
    """
    Calculate and store hashes for input arrays (lowECT, highECT, mask, segmentation).

    Parameters:
    group (Group): The DECT group object containing the input arrays.

    Note:
    Updates the 'current_hashes' attribute of the group object.
    """
    hashes = ['']*4
    for k, array in enumerate((group.lowECT, group.highECT, group.mask,
                               group.segmentation)):
        if array is not None:
            hashes[k] = hash_array(array, array.field_name)
    comm.barrier()
    group.current_hashes['lowE'] = hashes[0]
    group.current_hashes['highE'] = hashes[1]
    group.current_hashes['mask'] = hashes[2]
    group.current_hashes['segmentation'] = hashes[3]
    group.current_hashes['calibration_material0'] = group.calibration_material[0].hash()
    group.current_hashes['calibration_material1'] = group.calibration_material[1].hash()
    group.current_hashes['calibration_material2'] = group.calibration_material[2].hash()
    group.current_hashes['calibration_material3'] = group.calibration_material[3].hash()


def hash_coefficient_matrices(group):
    """
    Calculate the cumulative hash for the coefficient matrices.

    Parameters:
    group (Group): The DECT group object.

    Returns:
    str: The hexadecimal digest of the MD5 hash.
    """
    dependency = ''
    for k in sorted(group.current_hashes.keys()):
        dependency = dependency + group.current_hashes[k]
    md5 = hashlib.md5(dependency.encode('ascii'))
    matrixl = None
    matrixh = None
    with collective_only_rank0_runs():
        if mpi_rank == 0:
            matrixl = group.zgroup['matrixl'][...]
            matrixh = group.zgroup['matrixh'][...]
    matrixl = comm.bcast(matrixl, root=0)
    matrixh = comm.bcast(matrixh, root=0)
    if matrixl is not None:
        md5.update(matrixl)
    if matrixh is not None:
        md5.update(matrixh)
    return md5.hexdigest()



def need_coefficient_matrices(group):
    """
    Check if the coefficient matrices need to be recalculated.

    Parameters:
    group (Group): The DECT group object.

    Returns:
    bool: True if recalculation is needed, False otherwise.
    """
    if any(k not in group.zgroup for k in ('matrixl', 'matrixh')):
        return True
    str1 = hash_coefficient_matrices(group)
    strl, strh = None, None
    with collective_only_rank0_runs():
        if mpi_rank == 0:
            strl = group.zgroup['matrixl'].attrs['md5sum']
            strh = group.zgroup['matrixh'].attrs['md5sum']
    strl = comm.bcast(strl, root=0)
    strh = comm.bcast(strh, root=0)
    if str1 != strl or str1 != strh:
        return True
    return False



def need_output_array(group, array):
    """
    Check if a specific output array needs to be recalculated.

    Parameters:
    group (Group): The DECT group object.
    array (str): Name of the output array to check.

    Returns:
    bool: True if recalculation is needed, False otherwise.

    Note:
    Valid array names are 'rho_min', 'rho_p25', 'rho_p50', 'rho_p75', 'rho_max',
    'Z_min', 'Z_p25', 'Z_p50', 'Z_p75', 'Z_max', and 'valid'.
    """
    assert array in ('rho_min', 'rho_p25', 'rho_p50', 'rho_p75', 'rho_max',
                     'Z_min', 'Z_p25', 'Z_p50', 'Z_p75', 'Z_max', 'valid')
    dependency = hash_pre_process(group)
    need = False
    if mpi_rank == 0:
        if group.zgroup[array].attrs['md5sum'] != dependency:
                need = True
    need = comm.bcast(need, root=0)
    return need

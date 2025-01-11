"""
This module contains hash functions to verify the integrity and consistency of data and
intermediate results in Dual Energy Computed Tomography (DECT) processing.

Note:
    All functions are designed to work with the DECT Group class and its attributes.
"""

import hashlib
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

from rockverse._utils import rvtqdm


def hash_array(array, name, hash_buffer_size):
    """
    Calculate the MD5 hash of a 3D array, processing in blocks for memory efficiency.

    Parameters:
    array (drp.array): The 3D array to be hashed.
    name (str): Name of the array for progress bar display.

    Returns:
    str: The hexadecimal digest of the MD5 hash.
    """
    if array is None:
        return ''

    buffer = hash_buffer_size
    local_md5 = ['']*array.nchunks
    for block_id in rvtqdm(range(array.nchunks), desc=f'Hashing {name}', unit='chunk'):
        box, bex, boy, bey, boz, bez = array.chunk_slice_indices(block_id)
        if block_id % mpi_nprocs == mpi_rank:
            block = array[box:bex, boy:bey, boz:bez]
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
    hash_buffer_size = group.hash_buffer_size
    hashes = ['']*4
    for k, array in enumerate((group.lowECT, group.highECT, group.mask,
                               group.segmentation)):
        hashes[k] = hash_array(array,
                               array.field_name,
                               hash_buffer_size) # Initialise
    comm.barrier()
    for k in range(4):
        hexdigest = comm.bcast(hashes[k], root=k%mpi_nprocs)
        hashes[k] = hexdigest
        group.current_hashes['lowE'] = hashes[0]
        group.current_hashes['highE'] = hashes[1]
        group.current_hashes['mask'] = hashes[2]
        group.current_hashes['segmentation'] = hashes[3]



def hash_histogram(group, ct):
    """
    Calculate the hash for low or high energy histograms.

    Parameters:
    group (Group): The DECT group object.
    ct (str): Either 'low' or 'high' to specify which energy histogram to hash.

    Returns:
    str: The hexadecimal digest of the MD5 hash.
    """
    assert ct in ('low', 'high')
    dependency = (group.current_hashes[f'{ct}E']             #Image
                 + group.current_hashes['segmentation']     #Segmentation
                 + group.current_hashes['mask'])            #Mask
    md5 = hashlib.md5(dependency.encode('ascii'))
    md5.update(np.int64(group.histogram_bins))              #Histogram bins

    hist = group.lowEhistogram if ct == 'low' else group.highEhistogram
    md5.update(','.join(hist.columns).encode('ascii'))      #Histogram columns
    md5.update(hist.values)                                 #Histogram values
    return md5.hexdigest()



def need_histogram(group, ct):
    """
    Check if the histogram needs to be recalculated.

    Parameters:
    group (Group): The DECT group object.
    ct (str): Either 'low' or 'high' to specify which energy histogram to check.

    Returns:
    bool: True if recalculation is needed, False otherwise.
    """
    assert ct in ('low', 'high')
    hist = group.lowEhistogram if ct == 'low' else group.highEhistogram
    if hist is None:
        return True
    str1 = hash_histogram(group, ct)
    str2 = group.zgroup[ct+'EHistogram'].attrs['md5sum']
    if str1 != str2:
        return True
    return False



def hash_calibration_gaussian_coefficients(group):
    """
    Calculate the cumulative hash for the calibration Gaussian coefficients.

    Parameters:
    group (Group): The DECT group object.

    Returns:
    str: The hexadecimal digest of the MD5 hash.
    """
    dependency = hash_histogram(group, 'low') + hash_histogram(group, 'high')
    md5 = hashlib.md5(dependency.encode('ascii'))
    md5.update(group._calibration_gaussian_coefficient_values)
    return md5.hexdigest()



def need_calibration_gaussian_coefficients(group):
    """
    Check if the calibration Gaussian coefficients need to be recalculated.

    Parameters:
    group (Group): The DECT group object.

    Returns:
    bool: True if recalculation is needed, False otherwise.
    """
    if group._calibration_gaussian_coefficient_values is None:
        return True
    str1 = hash_calibration_gaussian_coefficients(group)
    str2 = group.zgroup['CalibrationGaussianCoefficients'].attrs['md5sum']
    if str1 != str2:
        return True
    return False




def hash_pre_process(group):
    """
    Calculate the cumulative hash for pre-processing data.

    Parameters:
    group (Group): The DECT group object.

    Returns:
    str: The hexadecimal digest of the MD5 hash.
    """
    dependency = hash_calibration_gaussian_coefficients(group)
    md5 = hashlib.md5(dependency.encode('ascii'))
    rho1, Z1v = group.calibration_material1._rhohat_Zn_values()
    md5.update(rho1)
    md5.update(Z1v)
    rho2, Z2v = group.calibration_material2._rhohat_Zn_values()
    md5.update(rho2)
    md5.update(Z2v)
    rho3, Z3v = group.calibration_material3._rhohat_Zn_values()
    md5.update(rho3)
    md5.update(Z3v)
    md5.update(np.float64(group.tol))
    return md5.hexdigest()



def hash_coefficient_matrices(group):
    """
    Calculate the cumulative hash for the coefficient matrices.

    Parameters:
    group (Group): The DECT group object.

    Returns:
    str: The hexadecimal digest of the MD5 hash.
    """
    dependency = hash_pre_process(group)
    md5 = hashlib.md5(dependency.encode('ascii'))
    matrixl = None
    matrixh = None
    if mpi_rank == 0:
        matrixl = group.zgroup['matrixl'][...]
        matrixh = group.zgroup['matrixh'][...]
    matrixl = comm.bcast(matrixl, root=0)
    matrixh = comm.bcast(matrixh, root=0)
    md5.update(matrixl)
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
    strl = group.zgroup['matrixl'].attrs['md5sum']
    strh = group.zgroup['matrixh'].attrs['md5sum']
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

"""
Provides the basic variable classes and creation functions for RockVerse data types.

These classes are built upon `Zarr <https://zarr.readthedocs.io>`_ arrays and groups,
and are tailored for high-performance parallel computation across multiple CPUs or GPUs
using MPI (Message Passing Interface), with optimized I/O operations and memory usage.

Understanding the basics of Zarr groups and chunked arrays is essential for effectively
working with RockVerse data.
"""

import h5py
import zarr
from rockverse.errors import collective_raise

from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

from rockverse.core.group import Group
from rockverse.core.attributes import Attributes
from rockverse.core.parallelarray import ParallelArray
from rockverse.core.tensorcoordinates import TensorCoordinateSet, TensorCoordinate
from rockverse.core.tensorfield import TensorField, TensorComponents, create_tensorfield


#>>>>>>>>>>>>>> PARALELIZE! READ BY CHUNKS, even when not chunked but large dataset
# TODO DEEP REVIEW
def load_array_from_h5_file(fobj, h5path, store, path=None, overwrite=False, **kwargs):

    """
    Loads a RockVerse TensorField from an HDF5 file.
    This function reads an existing RockVerse TensorField stored in an HDF5 file and
    creates a corresponding Array object in the specified Zarr storage.

    The data in the HDF5 file is expected to be in a particular format:

    .. code-block::

            GROUP "arraypath"
                |- ATTRIBUTE "_ROCKVERSE_DATATYPE" (string)
                |- ATTRIBUTE "description" (string)
                |- ATTRIBUTE "latex_name" (string)
                |- ATTRIBUTE "latex_unit" (string)
                |- ATTRIBUTE "name" (string)
                |- ATTRIBUTE "unit" (string)
                |- DATASET "data_0"
                    |- DATA (array)
                |- DATASET "data_1"
                    |- DATA (array)
                .
                .
                .
                |- DATASET "coord_0"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "coord_1"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "coord_2"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "coord_3"
                .
                .
                .

    for as many coord_ as coordinate arrays. Attributes are optional.
    Any extra attribute will also be loaded to the corresponding Zarr arrays.

    Parameters
    ----------
    fobj : h5py.File
        An opened HDF5 file object from which the RockVerse array will be loaded.
    h5path : str
        The path within the HDF5 file where the RockVerse array is located.
    store : str or zarr.storage.StoreLike
        The Zarr storage for the RockVerse Array.
    path : str, optional
        The path within the Zarr store where the array will be saved. Default is None.
    overwrite : bool, optional
        If True, deletes the existing store/path content before creating the new array.
        Default is False.
    **kwargs : keyword arguments
        Additional keyword arguments to be passed to the Zarr array creation function.

    Returns
    -------
    TensorField
        An instance of the RockVerse TensorField class.

    Raises
    ------
    KeyError
        If the specified HDF5 path is not found or if any expected datasets or attributes
        are missing from the HDF5 group.
    ValueError
        If the loaded data or attributes do not conform to expected formats or coordinate.
    TypeError
        If the specified HDF5 path does not point to a valid RockVerse array group.

    Example
    -------
    Load a RockVerse array from an HDF5 file:

    .. code-block:: python

        import h5py
        import rockverse as rv
        with h5py.File('filename.h5', 'r') as fobj:
            field1 = rv.core.load_array_from_h5_file(
                fobj, h5path='/myawesomearray', store='/path/to/zarr/store')

    This will load the contents in '/myawesomearray' from the HDF5 file and store it
    in the specified Zarr storage as a RockVerse TensorField.
    """

    if h5path not in fobj:
        collective_raise(KeyError(f"'{h5path}' not found in fobj."))
    group = fobj[h5path]

    # group must be a HDF5 group
    if not isinstance(group, h5py.Group):
        collective_raise(TypeError(f"fobj['{h5path}'] expected to be a Group. Found {type(group)}."))

    # group must contain data type identifier
    if "_ROCKVERSE_DATATYPE" not in group.attrs:
        collective_raise(KeyError(f"Missing '_ROCKVERSE_DATATYPE' identifier in the fobj['{h5path}'] object."))
    if group.attrs["_ROCKVERSE_DATATYPE"] != "TensorField":
        collective_raise(TypeError(f"fobj['{h5path}']: expected RockVerse TensorField type."))

    # Data arrays must exist
    data_arrays = [k for k in group.keys() if k.startswith('data_')]
    if not data_arrays:
        collective_raise(KeyError(f"Missing data arrays in fobj['{h5path}']."))

    # data arrays must be HDF5 datasets
    if not all(isinstance(group[k], h5py.Dataset) for k in data_arrays):
        collective_raise(TypeError(f"fobj['{h5path}'] data arrays expected to be Datasets."))

    # data array shapes must be the same
    shapes = [group[k].shape for k in data_arrays]
    if not all(k==shapes[0] for k in shapes):
        collective_raise(KeyError(f"Data array shapes must be the same."))
    shape = shapes[0]
    ndim = len(shapes[0])

    # Every coordinate array must exist
    missing_dims = [f"'coord_{k}'" for k in range(ndim) if f"coord_{k}" not in group]
    if len(missing_dims) == 1:
        collective_raise(KeyError(f"Missing {missing_dims[0]} Dataset in fobj['{h5path}']."))
    elif len(missing_dims) == 2:
        collective_raise(KeyError(f"Missing {' and '.join(missing_dims)} Datasets in fobj['{h5path}']."))
    elif len(missing_dims) > 2:
        collective_raise(KeyError(f"Missing {', '.join(missing_dims[:-1])}, and {missing_dims[-1]} Datasets in fobj['{h5path}']."))

    # Every coordinate array must be 1D
    not_1D = [f"fobj['{h5path}/coord_{k}']" for k in range(ndim) if len(group[f"coord_{k}"].shape) != 1]
    if len(not_1D) == 1:
        collective_raise(ValueError(f"Wrong shape in {not_1D[0]} Dataset. Coordinate arrays must be 1-D."))
    elif len(not_1D) == 2:
        collective_raise(ValueError(f"Wrong shape in {' and '.join(not_1D)} Datasets. Coordinate arrays must be 1-D."))
    elif len(not_1D) > 2:
        collective_raise(ValueError(f"Wrong shape in {', '.join(not_1D[:-1])}, and {not_1D[-1]} Datasets. Coordinate arrays must be 1-D."))

    # Shapes must match
    wrong_size = [f"len(coord_{k})={group[f"coord_{k}"].shape[0]}" for k in range(ndim) if group[f"coord_{k}"].shape[0] != shape[k]]
    if len(wrong_size) == 1:
        collective_raise(ValueError(f"fobj['{h5path}']: {wrong_size[0]} does not match data shape={data.shape}."))
    elif len(wrong_size) == 2:
        collective_raise(ValueError(f"fobj['{h5path}']: {' and '.join(wrong_size)} do not match data shape={data.shape}."))
    elif len(wrong_size) > 2:
        collective_raise(ValueError(f"fobj['{h5path}']: {', '.join(wrong_size[:-1])}, and {wrong_size[-1]} do not match data shape={self.shape}."))

    # Array-specific attributes must be string
    for attr in ('name', 'unit', 'description', 'latex_name', 'latex_unit'):
        if attr in group.attrs and not isinstance(group.attrs[attr], str):
            collective_raise(ValueError(f"fobj['{h5path}'].attrs['{attr}'] must be a string."))
        for array in [f"coord_{k}" for k in range(ndim)]:
            if attr in group[array].attrs and not isinstance(group[array].attrs[attr], str):
                collective_raise(ValueError(f"fobj['{h5path}/{array}'].attrs['{attr}'] must be a string."))

    # Import
    data = {tuple(int(i) for i in k.replace('data_', '').split('_')): group[k] for k in data_arrays}
    rvarray = create_tensorfield(data=data, #<<<<<<<<<<<< PARALELIZE!
                                 store=store,
                                 path=path,
                                 coord_data=[group[f'coord_{k}'][...] for k in range(ndim)],
                                 overwrite=overwrite,
                                 **kwargs)
    for k, v in group.attrs.items():
        rvarray.zgroup.attrs[k] = v
    for array in [f"coord_{k}" for k in range(ndim)]:
        for k, v in group[array].attrs.items():
            rvarray.zgroup[array].attrs[k] = v

    return rvarray

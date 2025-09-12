"""
Provides the basic variable classes and creation functions for RockVerse data types.

It includes the `TensorField` class, which represents generic N-dimensional arbitrary order
tensors with coordinates and associated metadata (similar to the
`Xarray project <https://docs.xarray.dev/en/stable/>`_, for example),
and the `Group` class, which facilitates generic data grouping and hierarchization.

These classes are built upon `Zarr <https://zarr.readthedocs.io>`_ arrays and groups,
and are tailored for high-performance parallel computation across multiple CPUs or GPUs
using MPI (Message Passing Interface), with optimized I/O operations and memory usage.

Understanding the basics of Zarr groups and chunked arrays is essential for effectively
working with RockVerse data.
"""

import os
import h5py
import zarr
from zarr.core.sync import sync
from itertools import product
from rockverse import _assert
from rockverse.errors import collective_raise

# TODO PARALELLIZE EVERYTHING
# TODO WRITE PLOT_FRIENDLY FUNCTIONS (labels, etc)
# TODO TENSOR PROPERTY ATTRS
# TODO TENSOR INTERFACE FOR DATA
# TODO Attributes I/O must be only through rank 0
# Create: allocate zarr group and arrays; only rank 0 fill in the attrs,
# rank0 reads data and send chunk to MPI process

from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

from rockverse.core.parallelarray import ParallelArray
from rockverse.core.tensorcoordinateset import TensorCoordinateSet, TensorCoordinate


def _tensor_shape(zgroup):
    data_arrays = [k for k in zgroup.array_keys() if k.startswith('data_')]
    if not data_arrays:
        collective_raise(KeyError("Missing data arrays in the zarr group."))
    data_indices = [k.replace('data_', '') for k in data_arrays]
    if data_indices[0].find('_') < 0:
        return max(int(k) for k in data_indices)+1
    tuple_indices = [tuple(int(i) for i in k.split('_')) for k in data_indices]
    return tuple(max(ind[k] for ind in tuple_indices)+1 for k in range(len(tuple_indices[0])))






class TensorComponent:

    def __init__(self, zgroup):
        _assert.zarr_group('zgroup', zgroup)
        self._zgroup = zgroup

    @property
    def zgroup(self):
        """
        The Zarr group containing the data.
        """
        return self._zgroup

    def __getitem__(self, index):

        data_arrays = [k for k in self.zgroup.array_keys() if k.startswith('data_')]
        data_indices = [k.replace('data_', '') for k in data_arrays]
        if data_indices[0].find('_')>=0:
            data_indices = [tuple(int(i) for i in k.split('_')) for k in data_indices]
        else:
            data_indices = [int(k) for k in data_indices]

        # index must be integer or integer array, same as tensor order
        if all(isinstance(k, int) for k in data_indices): #scalar field or vector field
            if not isinstance(index, int): #expected integer index
                if len(data_indices) > 1:
                    collective_raise(IndexError('expected integer index for order 1 tensor (vector field).'))
                else:
                    collective_raise(IndexError('expected index=0 for zero order tensor (scalar field).'))
        else: #tensor field order>1: expected array of integers
            if not (hasattr(index, '__getitem__') and all(isinstance(k, int) for k in index)):
                collective_raise(IndexError('expected integer array for index.'))

        if index not in data_indices:
            collective_raise(IndexError(f'invalid component index={index} for tensor shape {_tensor_shape(self.zgroup)}.'))

        ind = [k for k, v in enumerate(data_indices) if v == index][0]
        return ParallelArray(self.zgroup[data_arrays[ind]])


class TensorField:

    """
    A class representing generic N-dimensional, arbitrary order tensor fields
    with coordinates and associated metadata.

    This class serves as the basis for all tensor-like variables in RockVerse,
    (scalar fields, vector fields, etc), and is tailored for optimized
    multi-process memory usage and high-performance read and write disk access.
    It builds upon `Zarr <https://zarr.readthedocs.io>`_ arrays and groups, and is adapted for MPI
    (Message Passing Interface) processing, enabling parallel computation across multiple CPUs or GPUs.

    .. note::
        This class should not be instantiated directly. Instead, use the provided
        :ref:`creation functions <core module creation functions>`.

    Parameters
    ----------

    zgroup : zarr group
        An existing Zarr group with data properly organized.
    """

    def __init__(self, zgroup):
        """
        Initializes the Tensor instance with the Zarr group with the corresponding organized data.

        Parameters
        ----------

        zgroup : zarr.group.Group
            A Zarr group that contains the data and associated attributes for this tensor.
        """
        _assert.zarr_group('zgroup', zgroup)
        self._zgroup = zgroup
        self.validate()
        self._coordinate = TensorCoordinateSet(zgroup)
        self._component = TensorComponent(zgroup)


    @property
    def coordinate(self):
        """
        Provides access to the set of coordinates associated with this TensorField.
        Returns a :class:`TensorCoordinateSet` object that allows indexing by coordinate
        index or name, which retrieves of individual tensor coordinates as :class:`TensorCoordinate`
        objects.
        """
        return self._coordinate


    @property
    def component(self):
        """
        Provides access to the individual tensor component???
        """
        return self._component


    def validate(self):
        """
        Checks the consistency of the data and its associated attributes.
        This method verifies if all data in the underlying Zarr group is
        correctly defined and conforms to expected formats. If any inconsistencies
        or issues are found, the method raises appropriate errors to notify the user.
        The following must me True for a successful check (let zgroup be the Zarr Group object):

        - The `_ROCKVERSE_DATATYPE` attribute is in zgroup.attrs.
        - Some data array is in zgroup.
        - Data array indices must have same length (`data_0`, `data_1`, ..., or `data_0_0`, `data_0_1`, ..., etc).
        - Data array shapes and chunk sizes must be the same.
        - Data array data types must be the same.
        - Every coordinate array must exist and be a 1D array.
        - Each coordinate array shape must match the correspnding tensor component shape
        - Attributes `name`, `unit`, `description`, `latex_name`, and `latex_unit`, if defined
        for the tensor field or its components, must be strings.

        Returns
        -------
        None
            If all checks pass, the method returns None. If any validation fails, an exception
            will be raised to indicate the specific issue encountered.

        Raises
        ------
        ValueError
            If invalid values are found in the data attributes.
        KeyError
            If any expected attributes are missing from the data structure.
        """

        zgroup = self._zgroup

        # Data type identifier
        if "_ROCKVERSE_DATATYPE" not in zgroup.attrs:
            collective_raise(KeyError(f"Missing '_ROCKVERSE_DATATYPE' identifier in the zarr group attrs."))

        # data arrays must exist
        data_arrays = [k for k in zgroup.array_keys() if k.startswith('data_')]
        if not data_arrays:
            collective_raise(KeyError(f"Missing data arrays in the zarr group."))

        # array indices must have same length
        data_indices = [tuple(int(i) for i in k.replace('data_', '').split('_')) for k in data_arrays]
        order = [len(k) for k in data_indices]
        if not all(k==order[0] for k in order):
            collective_raise(ValueError(f'Inconsistent component indices: {data_indices}.'))

        # array shapes must be the same
        shapes = [zgroup[k].shape for k in data_arrays]
        if not all(k==shapes[0] for k in shapes):
            collective_raise(KeyError(f"Data array shapes must be the same."))
        shape = shapes[0]
        ndim = len(shapes[0])

        # array chunks must be the same
        chunks = [zgroup[k].chunks for k in data_arrays]
        if not all(k==chunks[0] for k in chunks):
            collective_raise(KeyError(f"Data arrays chunk size must be the same."))

        # array data types must be the same
        dtypes = [zgroup[k].dtype.str for k in data_arrays]
        if not all(k==dtypes[0] for k in dtypes):
            collective_raise(KeyError(f"Data array types must be the same."))

        # Every coordinate array must exist
        missing_dims = [f"'coord_{k}'" for k in range(ndim) if f"coord_{k}" not in zgroup]
        if len(missing_dims) == 1:
            collective_raise(KeyError(f"Missing {missing_dims[0]} array in the zarr group."))
        elif len(missing_dims) == 2:
            collective_raise(KeyError(f"Missing {' and '.join(missing_dims)} arrays in the zarr group."))
        elif len(missing_dims) > 2:
            collective_raise(KeyError(f"Missing {', '.join(missing_dims[:-1])}, and {missing_dims[-1]} arrays in the zarr group."))

        # Every coordinate array must be 1D
        not_1D = [f"'coord_{k}'" for k in range(ndim) if len(zgroup[f"coord_{k}"].shape) != 1]
        if len(not_1D) == 1:
            collective_raise(ValueError(f"Wrong shape in {not_1D[0]} array in the zarr group. Coordinate arrays must be 1-D."))
        elif len(not_1D) == 2:
            collective_raise(ValueError(f"Wrong shape in {' and '.join(not_1D)} arrays in the zarr group. Coordinate arrays must be 1-D."))
        elif len(not_1D) > 2:
            collective_raise(ValueError(f"Wrong shape in {', '.join(not_1D[:-1])}, and {not_1D[-1]} arrays in the zarr group. Coordinate arrays must be 1-D."))

        # Coordinate shapes must match
        wrong_size = [f"len(coord_{k})={zgroup[f"coord_{k}"].shape[0]}" for k in range(ndim) if zgroup[f"coord_{k}"].shape[0] != shape[k]]
        if len(wrong_size) == 1:
            collective_raise(ValueError(f"{wrong_size[0]} does not match data shape={self.shape}."))
        elif len(wrong_size) == 2:
            collective_raise(ValueError(f"{' and '.join(wrong_size)} do not match data shape={self.shape}."))
        elif len(wrong_size) > 2:
            collective_raise(ValueError(f"{', '.join(wrong_size[:-1])}, and {wrong_size[-1]} do not match data shape={self.shape}."))

        # Array-specific attributes must be string
        for attr in ('name', 'unit', 'description', 'latex_name', 'latex_unit'):
            if attr in zgroup.attrs and not isinstance(zgroup.attrs[attr], str):
                collective_raise(ValueError(f"zgroup.attrs['{attr}'] must be a string."))
            for array in [f"coord_{k}" for k in range(ndim)]:
                if attr in zgroup[array].attrs and not isinstance(zgroup[array].attrs[attr], str):
                    collective_raise(ValueError(f"zgroup['{array}'].attrs['{attr}'] must be a string."))

        # Non array-specific attributes won't be tested...
        return

    @property
    def zgroup(self):
        """
        The Zarr group containing the data.
        """
        return self._zgroup

    @property
    def _data_arrays(self):
        return tuple(k for k in self.zgroup.array_keys() if k.startswith('data_'))

    @property
    def dtype(self):
        """
        Numpy data type for the tensor components.
        """
        self.validate()
        return self.zgroup[self._data_arrays[0]].dtype

    @property
    def space_shape(self):
        """
        The shape of the coordinate space. Equivalent to the array shape of each tensor component.
        """
        self.validate()
        return self.zgroup[self._data_arrays[0]].shape

    @property
    def tensor_shape(self):
        """
        The tensor shape at each point in the coordinate space.
        """
        self.validate()
        return _tensor_shape(self.zgroup)

    @property
    def chunks(self):
        """
        The chunk size of each tensor component array.
        """
        self.validate()
        return self.zgroup[self._data_arrays[0]].chunks

    @property
    def order(self):
        """
        Tensor order.
        """
        self.validate()
        order = None
        if mpi_rank == 0:
            if len(self._data_arrays) == 1 and self._data_arrays[0] == 'data_0':
                order = 0
            else:
                data_indices = [tuple(int(i) for i in k.replace('data_', '').split('_')) for k in self._data_arrays]
                order = len(data_indices[0])
        order = comm.bcast(order, root=0)
        return order

    @property
    def ndim(self):
        """
        Number of data coordinates.
        """
        return len(self.zgroup[self._data_arrays[0]].shape)

    def _get_array(self, dim=None):
        """
        Retrieves the Zarr array for the specified coordinate.
        """
        if dim is None:
            return self.zgroup['data']
        if f'coord_{dim}' in self.zgroup:
            return self.zgroup[f'coord_{dim}']
        coord_names = self.coord_names
        pos = [k for k, v in enumerate(coord_names) if v == dim]
        if pos:
            return self.zgroup[f'coord_{pos[0]}']
        # Error from here...
        msg = f"dim='{dim}'" if isinstance(dim, str) else f"dim={dim}"
        collective_raise(KeyError(
            f"{msg} is not a valid coordinate index for this tensor. "
            f"Expected non negative integer < {len(self.zgroup['data'].shape)} or "
            f"one of the dim names {tuple(coord_names)}."))


    def _get_attribute(self, attr_name, dim=None):
        """
        Gets a specified attribute from the array for a given coordinate.
        """
        array = self._get_array(dim)
        attr_value = None
        if mpi_rank == 0:
            if attr_name in array.attrs:
                attr_value = array.attrs[attr_name]
        attr_value = comm.bcast(attr_value, root=0)
        return attr_value


    def _set_attribute(self, attr_name, attr_value, attr_type, dim=None):
        """
        Sets a specified attribute for the array for a given coordinate.
        """
        _assert.condition.non_negative_integer('dim', dim)
        array = self._get_array(dim)
        str_type = 'string' if attr_type == str else attr_type
        if not isinstance(attr_value, attr_type):
            collective_raise(ValueError(f"Expected {str_type} for {attr_name}."))
        if mpi_rank == 0:
            array.attrs[attr_name] = attr_value
        comm.barrier()


    def get_name(self, dim=None):
        """
        Retrieves the name of the data or a specified coordinate.

        Parameters
        ----------
        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th coordinate, or a string
            representing the coordinate name.

        Returns
        -------
        str
            The name of the data or the specified coordinate.
        """

        return self._get_attribute('name', dim=dim)

    def set_name(self, v, dim=None):
        """
        Sets the name of the data or a specified coordinate.

        Parameters
        ----------
        v : str
            The name to be set for the data or the specified coordinate.

        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th coordinate, or a
            string representing the coordinate name.
        """
        self._set_attribute(attr_name='name', attr_value=v, attr_type=str, dim=dim)

    def get_unit(self, dim=None):
        """
        Retrieves the unit of the data or a specified coordinate.

        Parameters
        ----------
        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th coordinate, or a string
            representing the coordinate name.

        Returns
        -------
        str
            The unit of the data or the specified coordinate.
        """

        return self._get_attribute('unit', dim=dim)

    def set_unit(self, v, dim=None):
        """
        Sets the unit of the data or a specified coordinate.

        Parameters
        ----------
        v : str
            The unit to be set for the data or the specified coordinate.

        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th coordinate, or a
            string representing the coordinate name.
        """
        self._set_attribute(attr_name='unit', attr_value=v, attr_type=str, dim=dim)

    def get_description(self, dim=None):
        """
        Retrieves the description of the data or a specified coordinate.

        Parameters
        ----------
        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th coordinate, or a string
            representing the coordinate name.

        Returns
        -------
        str
            The description of the data or the specified coordinate.
        """
        return self._get_attribute('description', dim=dim)

    def set_description(self, v, dim=None):
        """
        Sets the description of the data or a specified coordinate.

        Parameters
        ----------
        v : str
            The description to be set for the data or the specified coordinate.

        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th coordinate, or a
            string representing the coordinate name.
        """
        self._set_attribute(attr_name='description', attr_value=v, attr_type=str, dim=dim)

    def get_latex_name(self, dim=None):
        """
        Retrieves the LaTeX name representation for the data or a specified coordinate.

        Parameters
        ----------
        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th coordinate, or a string
            representing the coordinate name.

        Returns
        -------
        str
            The LaTeX name representation of the data or the specified coordinate.
        """
        return self._get_attribute('latex_name', dim=dim)

    def set_latex_name(self, v, dim=None):
        """
        Sets the LaTeX name representation for the data or a specified coordinate.

        Parameters
        ----------
        v : str
            The LaTeX name representation to be set for the data or the specified coordinate.

        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th coordinate, or a
            string representing the coordinate name.
        """
        self._set_attribute(attr_name='latex_name', attr_value=v, attr_type=str, dim=dim)

    def get_latex_unit(self, dim=None):
        """
        Retrieves the LaTeX unit representation for the data or a specified coordinate.

        Parameters
        ----------
        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th coordinate, or a string
            representing the coordinate name.

        Returns
        -------
        str
            The LaTeX unit representation of the data or the specified coordinate.
        """
        return self._get_attribute('latex_unit', dim=dim)

    def set_latex_unit(self, v, dim=None):
        """
        Sets the LaTeX unit representation for the data or a specified coordinate.

        Parameters
        ----------
        v : str
            The LaTeX unit representation to be set for the data or the specified coordinate.

        dim : None, int, or str, optional
            The coordinate specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th coordinate, or a
            string representing the coordinate name.
        """
        self._set_attribute(attr_name='latex_unit', attr_value=v, attr_type=str, dim=dim)




    @property
    def data(self):
        """
        A view for the underlying data array as a Zarr array.
        """
        return self.zgroup['data']

    @property
    def values(self):
        """
        The data array as a Numpy array.
        """
        return self.zgroup['data'][...]

    @property
    def name(self):
        """
        The tensor name.
        """
        return self.get_name()

    @property
    def unit(self):
        """
        The tensor data unit.
        """
        return self.get_unit()

    @property
    def description(self):
        """
        The tensor description.
        """
        return self.get_description()

    @property
    def latex_name(self):
        """
        The LaTeX representation for the tensor name.
        """
        return self.get_latex_name()

    @property
    def latex_unit(self):
        """
        The LaTeX representation for tensor data units.
        """
        return self.get_latex_unit()

    @property
    def coord_names(self):
        """
        Tuple containing the ordered coordinate names.
        """
        return tuple(self.get_name(dim=k) for k in range(len(self.zgroup['data'].shape)))

    @property
    def coord_units(self):
        """
        Tuple containing the ordered coordinate units.
        """
        return tuple(self.get_unit(dim=k) for k in range(len(self.zgroup['data'].shape)))

    @property
    def coord_descriptions(self):
        """
        Tuple containing the ordered coordinate descriptions.
        """
        return tuple(self.get_description(dim=k) for k in range(len(self.zgroup['data'].shape)))

    @property
    def coord_latex_names(self):
        """
        Tuple containing the ordered LaTeX representation for coordinate names.
        """
        return tuple(self.get_latex_name(dim=k) for k in range(len(self.zgroup['data'].shape)))

    @property
    def coord_latex_units(self):
        """
        Tuple containing the ordered LaTeX representation for coordinate units.
        """
        return tuple(self.get_latex_unit(dim=k) for k in range(len(self.zgroup['data'].shape)))

    def h5_dump(self, file_object, path):
        """
        Dumps the contents of the RockVerse tensor into an HDF5 file.
        This method exports the tensor data and its associated attributes
        into an HDF5 file at the specified path. It creates a
        group in the HDF5 file and stores the data array along with its metadata.
        The resulting HDF5 group will reflect the underlying zarr group:

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
                .
                .
                .

        for as many coordinates as array coordinates. <<<NAO PRECISA, BASTA GERAR UM RANGE(N)>>
        Attributes that are missing in the RockVerse array
        won't be written. Any extra attribute in the underlying Zarr group will also be dumped to the
        HDF5 file.

        .. note::
            This is a dump function executed in series, not in parallel, because we cannot
            guarantee that you have a parallel-enabled build of HDF5. Nevertheless, the
            function will work seamlessly in an MPI paralell environment, as chunked data
            from each process will be sent to the writing process to ensure data integrity
            (we got you covered!).

        Parameters
        ----------
        file_object : h5py.File
            An opened HDF5 file object where the array data will be dumped.
            Ensure to use an appropriate h5py.File open mode that allows file writing.
        path : str
            The path within the HDF5 file where the data will be stored.

        Example
        -------
        Dump the contents of the array into the '/myawesomearray' location in an HDF5 file:

        .. code-block:: python

            import h5py
            import rockverse as rv

            array_instance = rv.create_tensorfield(...)  # Create your array...
            with h5py.File('filename.h5', 'a') as fobj:
                array_instance.h5dump(fobj, path='/myawesomearray')

        Raises
        ------
        ValueError or KeyError
            If the array validation fails before dumping the data.

        """

        self.validate()
        grp = file_object.require_group(path)

        # Upper level attributes (_ROCKVERSE_DATATYPE, etc)
        for k, v in self.zgroup.attrs.items():
            grp.attrs[k] = v

        # Data arrays
        for array in [k for k in self.zgroup.array_keys() if k.startswith('data_')]:
            subgrp = file_object.create_dataset(f"{path}/{array}", data=self.zgroup[array]) #<<<<<<< PARALELIZE!

        # Coordinates and corresponding attributes
        for array in [k for k in self.zgroup.array_keys() if k.startswith('coord_')]:
            subgrp = file_object.create_dataset(f"{path}/{array}", data=self.zgroup[array]) #<<<<<<< PARALELIZE!
            for k, v in self.zgroup[array].attrs.items():
                subgrp.attrs[k] = v


def create_tensorfield(data,
                       store,
                       path=None,
                       chunks=None,
                       name=None,
                       unit=None,
                       description=None,
                       latex_name=None,
                       latex_unit=None,
                       coord_data=None,
                       coord_names=None,
                       coord_units=None,
                       coord_descriptions=None,
                       coord_latex_names=None,
                       coord_latex_units=None,
                       overwrite=False,
                       zarr_group_args=None,
                       zarr_array_args=None):
    """
    Create a RockVerse tensor field from provided data at specified Zarr storage.

    Parameters
    ----------

    data : array-like | dict
        The tensor components to be stored in the tensor field. It must be one of the following:
        an array-like object (for creating scalar fields);
        a list of arrays with identical shapes (for vector fields);
        a dictionary where keys are the component zero-based positions and values are the
        corresponding arrays.

        Examples:

        .. code-block::

            import numpy as np
            import rockverse as rv

            # Create a scalar field in a 10x10x10 grid
            field1 = rv.create_tensorfield(data=np.random.rand(10, 10, 10), ...

            # Create a 3-component vector field in a 10x10x10 grid
            field2 = rv.create_tensorfield(data=[np.random.rand(10, 10, 10),
                                                 np.random.rand(10, 10, 10),
                                                 np.random.rand(10, 10, 10)], ...

            # Create a 3x3 second order tensor field in a 10x10x10 grid
            # Missing components are assumed zero arrays
            field3 = rv.create_tensorfield(data={(0, 0): np.random.rand(10, 10, 10),
                                                 (1, 1): np.random.rand(10, 10, 10),
                                                 (2, 2): np.random.rand(10, 10, 10),
                                                 (0, 2): np.random.rand(10, 10, 10)}, ...

            # Integer dictionary keys are also valid entries
            # This is equivalent to field2 above:
            field4 = rv.create_tensorfield(data={0: np.random.rand(10, 10, 10),
                                                 1: np.random.rand(10, 10, 10),
                                                 2: np.random.rand(10, 10, 10)}, ...

            # And this is equivalent to field1 above:
            field4 = rv.create_tensorfield(data={0: np.random.rand(10, 10, 10)}, ...

            # This will create a 2-component vector, as position 0 will be treated as zero array:
            field5 = rv.create_tensorfield(data={1: np.random.rand(10, 10, 10)}, ...

    store : str or zarr.storage.StoreLike
        The storage location for the underlying Zarr group.

    path : str, optional
        The path within the store where the data will be saved.

    chunks : iterable of ints | None, optional
        If iterable of integers, define the chunk shape for each tensor component.
        Leave as `None` to match the array shape (no chunking).

    name : str, optional
        The name of the tensor field.

    unit : str, optional
        The data unit of the tensor field data.

    description : str, optional
        A description of the tensor field.

    latex_name : str, optional
        The LaTeX representation of the tensor field name.

    latex_unit : str, optional
        The LaTeX representation of the tensor field data unit.

    coord_data : tuple or list, optional
        Data for coordinates. The number of elements should match the shape of the tensor field components.
        Each element must be an 1D array-like with the coordinate values or ``None``.

    coord_names : tuple or list, optional
        Names for coordinates. The number of elements should match the shape of the tensor field components.
        Each element must be a string with the coordinate names or ``None``.

    coord_units : tuple or list, optional
        Units for coordinate data. The number of elements should match the shape of the tensor field components.
        Each element must be a string with the coordinate data unit or ``None``.

    coord_descriptions : tuple or list, optional
        Description for coordinate data. The number of elements should match the shape of the tensor field components.
        Each element must be a string with the coordinate description or ``None``.

    coord_latex_names : tuple or list, optional
        LaTeX names for coordinate data. The number of elements should match the shape of the tensor field components.
        Each element must be a string with the LaTeX representation of coordinate name or ``None``.

    coord_latex_units : tuple or list, optional
        LaTeX units for coordinate data. The number of elements should match the shape of the tensor field components.
        Each element must be a string with the LaTeX representation of coordinate data unit or ``None``.

    overwrite : bool, optional
        If True, deletes the store/path content before creating the new array.

    zarr_group_args : dict
        Dictionary with keyword arguments to be passed to the underlying Zarr group creation function.

    zarr_array_args : dict
        Dictionary with keyword arguments to be passed to the underlying Zarr array creation function.

    Example
    -------

    Create a tensor field with specified coordinates:

    .. code-block:: python

        import numpy as np
        import rockverse as rv
        temperature = rv.create_tensorfield(
            data=np.random.rand(5, 4, 3)+300,
            store='/path/to/zarr/store',
            name='Tr',
            latex_name='$T_r$',
            unit='K',
            description='Random temperature field, in kelvin units',
            coord_data=([10, 20, 30, 40, 50], [45, 55, 65, 75], [5, 6, 7]),
            coord_names=('x', 'y', 'z'),
            coord_units=('m', 'm', 'm')
        )

    Returns
    -------

    TensorField
        An instance of the RockVerse TensorField class representing the created array.
    """

    #PARALLEL I/O?
    #PARALLEL __GETITEM__
    #PARALLEL __SETITEM__
    #NÂO CRIAR DENTRO DE STORE QUE JA CONTENHA ROCKVERSE DATA?

    # Check for valid entries ----------------------------------

    _assert.boolean('overwrite', overwrite)
    if zarr_group_args is not None:
        _assert.dictionary('zarr_group_args', zarr_group_args)

    # data:
    # Array-like: order 0 (scalar field)
    if hasattr(data, '__array__'):
        components = {0: data}

    # List or tuple of arrays: order 1 (vector field)
    elif isinstance(data, (list, tuple)):
        if not all(hasattr(k, '__array__') for k in data):
            collective_raise(ValueError('Invalid value for data: list or tuple elements must be array-like.'))
        components = {k: v for k, v in enumerate(data)}

    # Dictionary with tensor positions and arrays: arbitrary order
    elif isinstance(data, dict):
        if not all(isinstance(k, int) or (isinstance(k, tuple) and all(isinstance(i, int) for i in k)) for k in data.keys()):
            collective_raise(ValueError('Invalid value for data: dictionary keys must be integer or tuple of integers with the zero-based component positions.'))
        if not all(type(k)==type(list(data.keys())[0]) for k in data.keys()):
            collective_raise(ValueError('Invalid value for data: dictionary keys must be of the same type.'))
        if not all(hasattr(v, '__array__') for v in data.values()):
            collective_raise(ValueError('Invalid value for data: dictionary values must be array-like.'))
        order = [1 if isinstance(k, int) else len(k) for k in data.keys()]
        if not all(k==order[0] for k in order):
            collective_raise(ValueError('Invalid value for data: dictionary keys must have same length.'))
        components = data
    else:
        collective_raise(ValueError('Invalid value for data.'))

    # Only component[(0, 0, ...)] will collapse to a scalar field
    if (len(data.keys()) == 1
        and isinstance(list(data.keys())[0], tuple)
        and all(k==0 for k in list(data.keys())[0])):
        components = {0: list(data.values())[0]}



    # All data arrays must have same shape:
    shapes = [v.shape for v in components.values()]
    if not all(s == shapes[0] for s in shapes):
        collective_raise(ValueError('Data components must have identical shape.'))
    shape = shapes[0]

    # Array data types must be numeric or boolean
    dtypes = [v.dtype.kind for v in components.values()]
    if not all(k in 'buifc' for k in dtypes):
        collective_raise(ValueError('Data arrays must be numeric or boolean.'))

    # Chunk shape length must match array shape length
    if chunks is None:
        chunks_ = shape
    else:
        _assert.iterable.ordered_integers_positive('chunks', chunks)
        if len(shape) != len(chunks):
            collective_raise(ValueError(f'chunks={chunks} not compatible with array shape={shape}.'))
        chunks_ = chunks

    # String attributes
    for varname, var in zip(('path', 'name', 'unit', 'description', 'latex_name', 'latex_unit'),
                            (path, name, unit, description, latex_name, latex_unit)):
        if var is not None:
            _assert.string(varname, var)

    for varname, var in zip(('coord_data', 'coord_names', 'coord_units', 'coord_descriptions', 'coord_latex_names', 'coord_latex_units'),
                            (coord_data, coord_names, coord_units, coord_descriptions, coord_latex_names, coord_latex_units)):
        if var is not None:
            _assert.iterable.tuple_or_list(varname, var)
            _assert.iterable.length(varname, var, len(shape))
            if varname != 'coord_data':
                _assert.iterable.ordered_string_or_none(varname, var)
            else: # coord_data
                for k, v in enumerate(var):
                    if v is not None and not isinstance(v, (list, tuple, np.ndarray)):
                        collective_raise(ValueError(f'Elements in {varname} must be list, tuple or 1D Numpy arrays.'))
                    if v is not None and isinstance(v, np.ndarray) and len(v.shape) != 1:
                        collective_raise(ValueError(f'Elements in {varname} must be list, tuple or 1D Numpy arrays.'))
                    if v is not None and len(v) != shape[k]:
                        collective_raise(ValueError(f'len(coord_data[{k}])={len(coord_data[k])} does not match data array shape[{k}]={shape[k]}.'))

    # Coordinate names must be unique
    if coord_names:
        for k1, name1 in enumerate(coord_names):
            if any(name1 and (name2 == name1) and (k2 != k1) for k2, name2 in enumerate(coord_names)):
                collective_raise(ValueError(f'Invalid coord_names={coord_names}: coordinate names must be unique.'))


    # Create the Zarr group and populate the data --------------
    # Should be done in parallel <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if zarr_group_args is None:
        kwargs = {}
    else:
        kwargs = dict(**zarr_group_args)
    kwargs['overwrite'] = overwrite
    kwargs['store'] = store
    kwargs['path'] = path
    kwargs['zarr_format'] = 3
    zgroup = zarr.create_group(**kwargs)

    data_attrs = {'_ROCKVERSE_DATATYPE': 'TensorField'}
    if name is not None:
        data_attrs['name'] = name
    if unit is not None:
        data_attrs['unit'] = unit
    if description is not None:
        data_attrs['description'] = description
    if latex_name is not None:
        data_attrs['latex_name'] = latex_name
    if latex_unit is not None:
        data_attrs['latex_unit'] = latex_unit
    zgroup.attrs.update(**data_attrs)

    # Data array type
    dtypes = [v.dtype.str for v in components.values()]
    dtypes_map = {}
    for t in 'cfiub':
        if any(t in type_ for type_ in dtypes):
            dtypes_map[t] = max(np.dtype(v).itemsize for v in dtypes if np.dtype(v).kind==t)
    if 'c' in dtypes_map:
        type_ = np.dtype(f'c{dtypes_map['c']}')
    elif 'f' in dtypes_map:
        type_ = np.dtype(f'f{dtypes_map['f']}')
    elif 'i' in dtypes_map or 'u' in dtypes_map:
        itemsize = max(v for k, v in dtypes_map.items() if k in 'iu')
        if 'i' in dtypes_map:
            type_ = np.dtype(f'i{itemsize}')
        else:
            type_ = np.dtype(f'u{itemsize}')
    else:
        type_ = np.dtype('bool')

    # Data arrays
    if zarr_array_args is None:
        kwargs = {}
    else:
        kwargs = dict(**zarr_array_args)
    kwargs['overwrite'] = overwrite
    kwargs['chunk_key_encoding'] = {"name": "default", "separator": "/"}

    components_keys = list(components.keys())
    if all(isinstance(k, int) for k in components_keys):
        data_arrays = tuple([str(k) for k in range(max(components_keys)+1)])
    else:
        max_inds = np.zeros(len(components_keys[0])).astype(int)
        for i in range(len(max_inds)):
            max_inds[i] = max([k[i] for k in components.keys()])+1
        ranges = [range(m) for m in max_inds]
        data_arrays = ["_".join(map(str, combo)) for combo in product(*ranges)]
    for name in data_arrays:
        temp = ParallelArray(zgroup.zeros(name=f"data_{name}",
                                          shape=shape,
                                          chunks=chunks_,
                                          dtype=type_,  # TODO specify in input parameters?
                                          **kwargs))
        ind = tuple(int(i) for i in name.split('_')) if name.find('_') >=0 else int(name)
        if ind in components_keys:
            temp[...] = components[ind]

    # Should be done by rank 0... <<<<<<<<<<<<<<<<<<<<<<<<<
    for k in range(len(shape)):
        coord_attrs = {}
        if coord_names is not None and coord_names[k]:
            coord_attrs['name'] = coord_names[k]
        else:
            coord_attrs['name'] = f"coord_{k}"
        if coord_units is not None and coord_units[k]:
            coord_attrs['unit'] = coord_units[k]
        if coord_descriptions is not None and coord_descriptions[k]:
            coord_attrs['description'] = coord_descriptions[k]
        if coord_latex_names is not None and coord_latex_names[k]:
            coord_attrs['latex_name'] = coord_latex_names[k]
        if coord_latex_units is not None and coord_latex_units[k]:
            coord_attrs['latex_unit'] = coord_latex_units[k]

        if coord_data is not None and coord_data[k] is not None:
            coord_data_k = np.array(coord_data[k])
        else:
            coord_data_k = np.arange(shape[k])

        zgroup.create_array(f"coord_{k}",
                            shape=coord_data_k.shape,
                            chunks=coord_data_k.shape, # no chunks in dim data
                            dtype=coord_data_k.dtype,
                            overwrite=overwrite,
                            attributes=coord_attrs)
        zgroup[f"coord_{k}"][...] = coord_data_k

    return TensorField(zgroup)


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


if __name__ == "__main__":
    import numpy as np
    import h5py
    import zarr
    zarr.config.config['array']['order']='F'
    self=create_tensorfield(
        data={(0, 0): np.random.rand(5,2,8).astype(float),
              },
        chunks=(2,2,2),
        #data = np.random.rand(5,2,8),
        store=r"C:\Users\GOB7\Downloads\test",
        #store='/u/gob7/test.zarr',
        path="testpath",
        name='test array',
        unit='m/s',
        description="UMA DESC",
        latex_name=r"$ERF$",
        latex_unit="MM",
        coord_data=([1, 2, 4, 7, 9], [2, 2], None),
        coord_names=("QQ", 'y','z'),
        coord_units=('km', "S", "F"),
        coord_descriptions=("UM", "DOIS", "WW"),
        coord_latex_names=(r"$r$", r"$i$", r"$p$"),
        coord_latex_units=('a', '', '.'),
        overwrite=True)
    self.validate()

    filename = '/u/gob7/test.h5'
    filename = r"C:\Users\GOB7\Downloads\test.h5"
    with h5py.File(filename, 'w') as fobj:
        self.h5_dump(fobj, '/myawesomearray')

    store='/u/gob7/test2.zarr'
    store=r"C:\Users\GOB7\Downloads\test2"
    h5path = '/myawesomearray'
    path=None
    overwrite=True
    kwargs={}
    #with h5py.File(filename, mode='r') as fobj:
    #    self2 = load_array_from_h5_file(fobj, h5path, store, path=None, overwrite=True)

    self.component[0]
    self.component[0].zarray[...]
    self.component[0][...]

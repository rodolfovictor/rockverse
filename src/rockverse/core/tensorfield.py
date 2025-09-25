import h5py
import numpy as np
from itertools import product
from rockverse import _assert
from rockverse.errors import (
    collective_raise,
    CustomCollectiveException
    )

# TODO TENSOR INTERFACE FOR __setitem__

from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

from rockverse.core.group import create_group
from rockverse.core.attributes import Attributes
from rockverse.core.parallelarray import ParallelArray
from rockverse.core.coordinates import CoordinateSet


def _tensor_shape(zgroup):
    component_arrays = [k for k in zgroup.array_keys() if k.startswith('component_')]
    if not component_arrays:
        collective_raise(KeyError("Missing data arrays in the zarr group."))
    component_indices = [k.replace('component_', '') for k in component_arrays]
    if component_indices[0].find('_') < 0:
        return max(int(k) for k in component_indices)+1
    tuple_indices = [tuple(int(i) for i in k.split('_')) for k in component_indices]
    return tuple(max(ind[k] for ind in tuple_indices)+1 for k in range(len(tuple_indices[0])))

def _component_indices(zgroup):
    arrays = [k for k in zgroup.array_keys() if k.startswith('component_')]
    indices = [k.replace('component_', '') for k in arrays]
    if indices[0].find('_')>=0:
        indices = [tuple(int(i) for i in k.split('_')) for k in indices]
    else:
        indices = [int(k) for k in indices]
    return indices, arrays


class TensorComponents:
    """
    Represents the individual components of the tensor field. This class
    provides access to each numeric component array of a tensor field, allowing
    retrieval of specific components by their indices.

    The `TensorComponents` object can be indexed to specific components as
    :class:`ParallelArray` objects. For example:

    .. code-block:: python

        # create your tensor field
        tensor0 = create_tensorfield(...)

        # to get the first tensor component as a ParallelArray object
        comp0 = tensorfield.component[0]

    .. note::
        This class should not be directly instantiated.
        It is managed by RockVerse
        :ref:`creation functions <core module creation functions>`.

    Parameters
    ----------
    zgroup : zarr.group.Group
        The Zarr group containing the component arrays of the tensor field.

    """

    def __init__(self, zgroup):
        _assert.zarr_group('zgroup', zgroup)
        self._zgroup = zgroup

    @property
    def zgroup(self):
        """
        The Zarr group containing the parent tensor field data.
        """
        return self._zgroup

    def __getitem__(self, index):
        component_indices, component_arrays = _component_indices(self.zgroup)
        if index not in component_indices:
            collective_raise(IndexError(f'invalid component index={index} for tensor shape {_tensor_shape(self.zgroup)}.'))
        ind = [k for k, v in enumerate(component_indices) if v == index][0]
        return ParallelArray(self.zgroup[component_arrays[ind]])


class TensorField:

    """
    A class representing generic N-dimensional, arbitrary order tensor fields
    with coordinates and associated metadata.

    This class serves as the basis for all tensor-like variables in RockVerse,
    (scalar fields, vector fields, etc), and is tailored for optimized
    multi-process memory usage and high-performance read and write disk access.
    It builds upon `Zarr <https://zarr.readthedocs.io>`_ arrays and groups, and is
    adapted for MPI (Message Passing Interface) processing, enabling parallel
    computation across multiple CPUs or GPUs.

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
        Initializes the TensorField instance with the Zarr group with the corresponding organized data.

        Parameters
        ----------

        zgroup : zarr.group.Group
            A Zarr group that contains the data and associated attributes for this tensor.
        """
        _assert.zarr_group('zgroup', zgroup)
        self._zgroup = zgroup
        self._coordinates = CoordinateSet(zgroup)
        self._components = TensorComponents(zgroup)
        self._attrs = Attributes(zgroup)
        self.validate()


    def __getitem__(self, index):
        """
        Retrieve the tensor components at a specific coordinate index in the
        coordinate space.

        Parameters
        ----------
        index : tuple of int
            A tuple of integers specifying the coordinate indices within the
            tensor's spatial domain. The length of the tuple must match the
            number of spatial dimensions (`self.shape`).

        Returns
        -------
        numpy.ndarray
            An array containing the values of all tensor components at the
            specified coordinate.
        """
        if not all(isinstance(k, int) for k in index):
            collective_raise(IndexError("Only integers accepeted for tensor index."))
        if len(index) != len(self.shape):
            collective_raise(IndexError("Index must point to one element in the coordinate space."))
        component_indices, component_arrays = _component_indices(self.zgroup)
        result = np.zeros(self.tensor_shape, dtype=self.dtype)
        for indice, array_name in zip(component_indices, component_arrays):
            print(index)
            result[indice] = self.zgroup[array_name][index]
        return result


    @property
    def coordinates(self):
        """
        Provides access to the set of coordinates associated with this TensorField.
        Returns a :class:`CoordinateSet` object that allows indexing by coordinate
        index or name, which retrieves of individual tensor field coordinates as
        :class:`Coordinate` objects.
        """
        return self._coordinates

    @property
    def components(self):
        """
        Provides access to the individual components of the tensor field.
        Returns a :class:`TensorComponents` object that encapsulates the numeric arrays
        representing each component of the tensor field.
        """
        return self._components

    @property
    def attrs(self):
        """
        The collective metadata attributes associated with this tensor field
        as an :class:`Attributes` object.
        """
        return self._attrs


    @property
    def zgroup(self):
        """
        The Zarr group containing the data.
        """
        return self._zgroup


    @property
    def _component_arrays(self):
        return tuple(k for k in self.zgroup.array_keys() if k.startswith('component_'))

    @property
    def dtype(self):
        """
        Numpy data type for the tensor components.
        """
        self.validate()
        return self.zgroup[self._component_arrays[0]].dtype

    @property
    def shape(self):
        """
        The shape of the coordinate space. Equivalent to the array shape of each tensor component.
        """
        self.validate()
        return self.zgroup[self._component_arrays[0]].shape

    @property
    def ndim(self):
        """
        Coordinate space dimension (the number of data coordinates).
        """
        self.validate()
        return len(self.zgroup[self._component_arrays[0]].shape)


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
        return self.zgroup[self._component_arrays[0]].chunks

    @property
    def tensor_order(self):
        """
        Tensor order.
        """
        self.validate()
        if len(self._component_arrays) == 1 and self._component_arrays[0] == 'component_0':
            order = 0
        else:
            component_indices = [tuple(int(i) for i in k.replace('component_', '').split('_')) for k in self._component_arrays]
            order = len(component_indices[0])
        return order

    @property
    def name(self):
        """
        Get or set the tensor name (alias for `attrs['name']`).
        """
        return self.attrs.get('name', default=None)

    @name.setter
    def name(self, v):
        _assert.string('name', v)
        self.attrs['name'] = v

    @property
    def unit(self):
        """
        Get or set the tensor data unit (linked to `attrs['unit']`).
        """
        return self.attrs.get('unit', default=None)

    @unit.setter
    def unit(self, v):
        _assert.string('unit', v)
        self.attrs['unit'] = v

    @property
    def description(self):
        """
        Get or set the tensor description (linked to `attrs['description']`).
        """
        return self.attrs.get('description', default=None)

    @description.setter
    def description(self, v):
        _assert.string('description', v)
        self.attrs['description'] = v

    @property
    def latex_name(self):
        """
        Get or set the tensor LaTeX representation for the tensor name
        (linked to `attrs['latex_name']`).
        """
        return self.attrs.get('latex_name', default=None)

    @latex_name.setter
    def latex_name(self, v):
        _assert.string('latex_name', v)
        self.attrs['latex_name'] = v

    @property
    def latex_unit(self):
        """
        Get or set the tensor LaTeX representation for the tensor data unit
        (linked to `attrs['latex_unit']`).
        """
        return self.attrs.get('latex_unit', default=None)

    @latex_unit.setter
    def latex_unit(self, v):
        _assert.string('latex_unit', v)
        self.attrs['latex_unit'] = v

    def get_plot_label(self, unit=True, latex=True):
        """
        Build a string 'name (unit)' based on the tensor attributes.

        Parameters
        ----------
        unit : bool, optional
            If True, use unit in the resulting string -> `'name (unit)'`.
            If False, discard unit -> `'name'`.
            Default is True.
        latex : bool, optional
            If True, use the LaTeX reprentations instead of the plain ones.
            Fall back to plain labels in case of absent LaTeX versions.
            Default is True.

        Returns
        -------
        str
            The resulting string to be used as a plot label.
        """
        name_str = self.latex_name if self.latex_name and latex else self.name
        if unit:
            unit_str = self.latex_unit if self.latex_unit and latex else self.unit
            if unit_str:
                name_str = f"{name_str} ({unit_str})"
        return name_str

    def validate(self):
        """
        Checks the consistency of the data and its associated attributes.
        This method verifies if all data in the underlying Zarr group is
        correctly defined and conforms to expected formats. If any inconsistencies
        or issues are found, the method raises appropriate errors to notify the user.
        The following must me True for a successful check (let zgroup be the Zarr Group object):

        - The `_ROCKVERSE_DATATYPE` attribute is in zgroup.attrs.
        - Some component array is in zgroup.
        - Component array indices must have same length (`component_0`, `component_1`, ..., or `component_0_0`, `component_0_1`, ..., etc).
        - Component array shapes and chunk sizes must be the same.
        - Component array data types must be the same.
        - Every coordinate array must exist and be a 1D array.
        - Each coordinate array shape must match the corresponding tensor component shape
        - Attributes `name`, `unit`, `description`, `latex_name`, and `latex_unit`, if defined for the tensor field or its components, must be strings.

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

        # Data type identifier
        if "_ROCKVERSE_DATATYPE" not in self.attrs:
            collective_raise(KeyError("Missing '_ROCKVERSE_DATATYPE' identifier in the zarr group attrs."))

        # component arrays must exist
        component_arrays = [k for k in self._zgroup.array_keys() if k.startswith('component_')]
        if not component_arrays:
            collective_raise(KeyError("Missing component arrays in the zarr group."))

        # array indices must have same length
        component_indices = [tuple(int(i) for i in k.replace('component_', '').split('_')) for k in component_arrays]
        order = [len(k) for k in component_indices]
        if not all(k==order[0] for k in order):
            collective_raise(ValueError(f'Inconsistent component indices: {component_indices}.'))

        # component array shapes must be the same
        shapes = [self._zgroup[k].shape for k in component_arrays]
        if not all(k==shapes[0] for k in shapes):
            collective_raise(KeyError("Component array shapes must be the same."))
        shape = shapes[0]
        ndim = len(shapes[0])

        # Component array chunks must be the same
        chunks = [self._zgroup[k].chunks for k in component_arrays]
        if not all(k==chunks[0] for k in chunks):
            collective_raise(KeyError("Component arrays chunk size must be the same."))

        # Component array types must be the same
        dtypes = [self._zgroup[k].dtype.str for k in component_arrays]
        if not all(k==dtypes[0] for k in dtypes):
            collective_raise(KeyError("Component array types must be the same."))

        # Every coordinate array must exist
        missing_dims = [f"'coord_{k}'" for k in range(ndim) if f"coord_{k}" not in self._zgroup]
        if len(missing_dims) == 1:
            collective_raise(KeyError(f"Missing {missing_dims[0]} array in the zarr group."))
        elif len(missing_dims) == 2:
            collective_raise(KeyError(f"Missing {' and '.join(missing_dims)} arrays in the zarr group."))
        elif len(missing_dims) > 2:
            collective_raise(KeyError(f"Missing {', '.join(missing_dims[:-1])}, and {missing_dims[-1]} arrays in the zarr group."))

        # Every coordinate array must be 1D
        not_1D = [f"'coord_{k}'" for k in range(ndim) if len(self._zgroup[f"coord_{k}"].shape) != 1]
        if len(not_1D) == 1:
            collective_raise(ValueError(f"Wrong shape in {not_1D[0]} array in the zarr group. Coordinate arrays must be 1-D."))
        elif len(not_1D) == 2:
            collective_raise(ValueError(f"Wrong shape in {' and '.join(not_1D)} arrays in the zarr group. Coordinate arrays must be 1-D."))
        elif len(not_1D) > 2:
            collective_raise(ValueError(f"Wrong shape in {', '.join(not_1D[:-1])}, and {not_1D[-1]} arrays in the zarr group. Coordinate arrays must be 1-D."))

        # Coordinate shapes must match
        wrong_size = [f"len(coord_{k})={self.zgroup[f"coord_{k}"].shape[0]}" for k in range(ndim) if self._zgroup[f"coord_{k}"].shape[0] != shape[k]]
        if len(wrong_size) == 1:
            collective_raise(ValueError(f"{wrong_size[0]} does not match component shape={self.shape}."))
        elif len(wrong_size) == 2:
            collective_raise(ValueError(f"{' and '.join(wrong_size)} do not match component shape={self.shape}."))
        elif len(wrong_size) > 2:
            collective_raise(ValueError(f"{', '.join(wrong_size[:-1])}, and {wrong_size[-1]} do not match component shape={self.shape}."))

        # Array-specific attributes must be string
        for attr in ('name', 'unit', 'description', 'latex_name', 'latex_unit'):
            if attr in self.attrs and not isinstance(self.attrs[attr], str):
                collective_raise(ValueError(f"attrs['{attr}'] must be a string."))
            for array_name in [f"coord_{k}" for k in range(ndim)]:
                if attr in self.zgroup[array_name].attrs and not isinstance(self.zgroup[array_name].attrs[attr], str):
                    collective_raise(ValueError(f"zgroup['{array_name}'].attrs['{attr}'] must be a string."))

        # Non array-specific attributes won't be tested...
        return


    def h5_dump(self, filename, path, mode='a', **kwargs):
        """
        Export the tensor field data and its attributes to an HDF5 file.

        This method writes the contents of the tensor field into an HDF5 dataset at
        the specified path within the given file. The operation is performed serially
        by all MPI processes in rank order to ensure compatibility with
        non-MPI-enabled HDF5 libraries.

        Parameters
        ----------
        filename : str
            The name of the HDF5 file to write to.
        path : str
            The internal path within the HDF5 file where the dataset will be stored.
        mode : str, optional
            File open mode. Options include:

            - 'r' : Readonly, file must exist.
            - 'r+' : Read/write, file must exist.
            - 'w' : Create file, truncate if exists.
            - 'w-' or 'x' : Create file, fail if exists.
            - 'a' : Read/write if exists, create otherwise (default).

        **kwargs
            Additional keyword arguments to pass to `h5py.File`.

        Example
        -------
        Dump the contents of the tensor field into the '/my/awesome/tensor' location in an HDF5 file:

        .. code-block:: python

            import rockverse as rv
            tensor_instance = rv.create_tensor(...)  # Create your tensor...
            tensor_instance.h5_dump('filename.h5', path='/my/awesome/tensor')
        """

        self.validate()

        # Serial writing. HDF5 installation may not have MPI enabled...

        # Rank 0 writes top-level attributes and coordinate arrays
        error_msg = ''
        if mpi_rank == 0:
            try:
                with h5py.File(filename, mode, **kwargs) as fobj:
                    h5grp = fobj.require_group(path)

                    # Upper level attributes (_ROCKVERSE_DATATYPE, etc)
                    for k, v in self._zgroup.attrs.items():
                        h5grp.attrs[k] = v

                    # Coordinates
                    coord_arrays = sorted(k for k in self._zgroup.array_keys() if k.startswith('coord_'))
                    for array_name in coord_arrays:
                        h5array = h5grp.create_dataset(name=array_name, data=self._zgroup[array_name])
                        for k, v in self._zgroup[array_name].attrs.items():
                            h5array.attrs[k] = v

            except Exception as e:
                error_msg = f"{e.__class__.__name__}: {e}"
        error_msg = comm.bcast(error_msg, root=0)
        if error_msg:
            name, msg = error_msg.split(':')[0].strip(), ''.join(error_msg.split(':')[1:]).strip()
            msg = f"Error exporting tensor field to {path} in {filename}. {msg}"
            collective_raise(CustomCollectiveException(name, msg))

        # Component arrays
        component_arrays = sorted(k for k in self._zgroup.array_keys() if k.startswith('component_'))
        for array_name in component_arrays:
            index = array_name.replace('component_', '')
            if index.find('_') < 0:
                index = int(index)
            else:
                index = tuple(int(i) for i in index.split('_'))
            self.components[index].h5_dump(filename=filename,
                                          path=f"{path}/{array_name}",
                                          mode='a',
                                          **kwargs)


def create_tensorfield(tensor_shape,
                       shape,
                       dtype,
                       store=None,
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

    tensor_shape : tuple | list
        The tensor shape at each point in the coordinate space.

    shape : tuple | list
        The shape of the coordinate space. Equivalent to the array shape of each tensor component.

    dtype :
        Numpy data type for the tensor components.

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

    Returns
    -------

    TensorField
        An instance of the RockVerse TensorField class representing the created array.
    """

    # Check for valid tensor shape --------------
    _assert.iterable.ordered_integers_positive('tensor_shape', tensor_shape)

    # Check for valid shape ---------------------
    _assert.iterable.ordered_integers_positive('shape', shape)

    # Check for valid dtype ---------------------
    _assert.condition.numeric_or_boolean('dtype', dtype)

    # Check for valid chunks --------------------
    if not chunks:
        chunks_ = shape
    else:
        chunks_ = chunks
    _assert.iterable.ordered_numbers_positive('chunks', chunks_)
    if len(shape) != len(chunks):
        collective_raise(ValueError(f'chunks={chunks} not compatible with array shape={shape}.'))

    # Check for valid overwrite -----------------
    _assert.instance('overwrite', overwrite, 'boolean', (bool,))

    # Check for valid path ----------------------
    if path is not None:
        _assert.instance('path', path, 'string', (str,))

    if zarr_group_args is not None:
        _assert.dictionary('zarr_group_args', zarr_group_args)

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

    # Create the group -------------------------------
    if zarr_group_args is None:
        kwargs = {}
    else:
        kwargs = dict(**zarr_group_args)
    kwargs['overwrite'] = overwrite
    kwargs['store'] = store
    kwargs['path'] = path
    group = create_group(**kwargs)

    group.attrs['_ROCKVERSE_DATATYPE'] = 'TensorField'
    if name is not None:
        group.attrs['name'] = name
    if unit is not None:
        group.attrs['unit'] = unit
    if description is not None:
        group.attrs['description'] = description
    if latex_name is not None:
        group.attrs['latex_name'] = latex_name
    if latex_unit is not None:
        group.attrs['latex_unit'] = latex_unit

    comm.barrier()

    # Coordinate arrays
    for k in range(len(shape)):
        if coord_data is not None and coord_data[k] is not None:
            coord_data_k = np.array(coord_data[k])
        else:
            coord_data_k = np.arange(shape[k])
        new_coord = group.create_array(path=f"coord_{k}",
                                       shape=coord_data_k.shape,
                                       chunks=coord_data_k.shape, # single chunk
                                       dtype=coord_data_k.dtype,
                                       overwrite=overwrite)
        new_coord[...] = coord_data_k
        if coord_names is not None and coord_names[k]:
            new_coord.attrs['name'] = coord_names[k]
        else:
            new_coord.attrs['name'] = f"coord_{k}"
        if coord_units is not None and coord_units[k]:
            new_coord.attrs['unit'] = coord_units[k]
        if coord_descriptions is not None and coord_descriptions[k]:
            new_coord.attrs['description'] = coord_descriptions[k]
        if coord_latex_names is not None and coord_latex_names[k]:
            new_coord.attrs['latex_name'] = coord_latex_names[k]
        if coord_latex_units is not None and coord_latex_units[k]:
            new_coord.attrs['latex_unit'] = coord_latex_units[k]

    comm.barrier()

    # Data arrays -----------------------------------------
    if zarr_array_args is None:
        kwargs = {}
    else:
        kwargs = dict(**zarr_array_args)
    kwargs['overwrite'] = overwrite
    kwargs['dtype'] = dtype
    kwargs['shape'] = shape
    kwargs['chunks'] = chunks_

    ranges = [range(m) for m in tensor_shape]
    component_arrays = ["_".join(map(str, combo)) for combo in product(*ranges)]
    for name in component_arrays:
        _ = group.create_array(path=f"component_{name}", **kwargs)

    return TensorField(group._zgroup)

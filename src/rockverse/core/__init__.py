"""
Provides the basic variable classes and creation functions
for all data types handled in RockVerse.

It includes the `Array` class, which represents generic N-dimensional arrays with
coordinates and associated metadata (similar to the
`Xarray project <https://docs.xarray.dev/en/stable/>`_, for example),
and the `Group` class, which facilitates generic data grouping and hierarchization.

These classes are built upon `Zarr <https://zarr.readthedocs.io>`_ arrays and groups,
and are tailored for high-performance parallel computation across multiple CPUs or GPUs
using MPI (Message Passing Interface), with optimized I/O operations and memory usage.
"""

import os
import h5py
import zarr
from rockverse import _assert
from rockverse.errors import collective_raise

# TODO PARALELLIZE EVERYTHING
# TODO WRITE PLOT_FRIENDLY FUNCTIONS (labels, etc)
# TODO ARRAY PROPERTY ATTRS
# TODO ARRAY INTERFACE FOR DATA

from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs


class Array:

    """
    A class representing a generic N-dimensional array with coordinates and associated metadata.

    This class serves as the basis for all array-like variables in RockVerse, tailored for optimized
    multi-process memory usage and high-performance read and write access.
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
        Initializes the Array instance with the Zarr group with the corresponding organized data.

        Parameters
        ----------

        zgroup : zarr.group.Group
            A Zarr group that contains the data and associated attributes for this array.
        """
        _assert.zarr_group('zgroup', zgroup)
        self._zgroup = zgroup

    @property
    def zgroup(self):
        """
        The Zarr group containing the data.
        """
        return self._zgroup

    def _get_array(self, dim=None):
        """
        Retrieves the Zarr array for the specified dimension.
        """
        if dim is None:
            return self.zgroup['data']
        if f'dim_{dim}' in self.zgroup:
            return self.zgroup[f'dim_{dim}']
        dim_names = self.dim_names
        pos = [k for k, v in enumerate(dim_names) if v == dim]
        if pos:
            return self.zgroup[f'dim_{pos[0]}']
        # Error from here...
        msg = f"dim='{dim}'" if isinstance(dim, str) else f"dim={dim}"
        collective_raise(KeyError(
            f"{msg} is not a valid dimension index for this array. "
            f"Expected non negative integer < {len(self.zgroup['data'].shape)} or "
            f"one of the dim names {tuple(dim_names)}."))


    def _get_attribute(self, attr_name, dim=None):
        """
        Gets a specified attribute from the array for a given dimension.
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
        Sets a specified attribute for the array for a given dimension.
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
        Retrieves the name of the data or a specified dimension.

        Parameters
        ----------
        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th dimension, or a string
            representing the dimension name.

        Returns
        -------
        str
            The name of the data or the specified dimension.
        """

        return self._get_attribute('name', dim=dim)

    def set_name(self, v, dim=None):
        """
        Sets the name of the data or a specified dimension.

        Parameters
        ----------
        v : str
            The name to be set for the data or the specified dimension.

        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th dimension, or a
            string representing the dimension name.
        """
        self._set_attribute(attr_name='name', attr_value=v, attr_type=str, dim=dim)

    def get_unit(self, dim=None):
        """
        Retrieves the unit of the data or a specified dimension.

        Parameters
        ----------
        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th dimension, or a string
            representing the dimension name.

        Returns
        -------
        str
            The unit of the data or the specified dimension.
        """

        return self._get_attribute('unit', dim=dim)

    def set_unit(self, v, dim=None):
        """
        Sets the unit of the data or a specified dimension.

        Parameters
        ----------
        v : str
            The unit to be set for the data or the specified dimension.

        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th dimension, or a
            string representing the dimension name.
        """
        self._set_attribute(attr_name='unit', attr_value=v, attr_type=str, dim=dim)

    def get_description(self, dim=None):
        """
        Retrieves the description of the data or a specified dimension.

        Parameters
        ----------
        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th dimension, or a string
            representing the dimension name.

        Returns
        -------
        str
            The description of the data or the specified dimension.
        """
        return self._get_attribute('description', dim=dim)

    def set_description(self, v, dim=None):
        """
        Sets the description of the data or a specified dimension.

        Parameters
        ----------
        v : str
            The description to be set for the data or the specified dimension.

        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th dimension, or a
            string representing the dimension name.
        """
        self._set_attribute(attr_name='description', attr_value=v, attr_type=str, dim=dim)

    def get_latex_name(self, dim=None):
        """
        Retrieves the LaTeX name representation for the data or a specified dimension.

        Parameters
        ----------
        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th dimension, or a string
            representing the dimension name.

        Returns
        -------
        str
            The LaTeX name representation of the data or the specified dimension.
        """
        return self._get_attribute('latex_name', dim=dim)

    def set_latex_name(self, v, dim=None):
        """
        Sets the LaTeX name representation for the data or a specified dimension.

        Parameters
        ----------
        v : str
            The LaTeX name representation to be set for the data or the specified dimension.

        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th dimension, or a
            string representing the dimension name.
        """
        self._set_attribute(attr_name='latex_name', attr_value=v, attr_type=str, dim=dim)

    def get_latex_unit(self, dim=None):
        """
        Retrieves the LaTeX unit representation for the data or a specified dimension.

        Parameters
        ----------
        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to retrieve the name of the main data,
            a non-negative integer `i` to retrieve the i-th dimension, or a string
            representing the dimension name.

        Returns
        -------
        str
            The LaTeX unit representation of the data or the specified dimension.
        """
        return self._get_attribute('latex_unit', dim=dim)

    def set_latex_unit(self, v, dim=None):
        """
        Sets the LaTeX unit representation for the data or a specified dimension.

        Parameters
        ----------
        v : str
            The LaTeX unit representation to be set for the data or the specified dimension.

        dim : None, int, or str, optional
            The dimension specifier. Use ``None`` to set the unit of the main data,
            a non-negative integer `i` to set the unit for the i-th dimension, or a
            string representing the dimension name.
        """
        self._set_attribute(attr_name='latex_unit', attr_value=v, attr_type=str, dim=dim)

    @property
    def ndim(self):
        """
        Number of data array dimensions.
        """
        return self.zgroup['data'].ndim

    @property
    def shape(self):
        """
        The data array shape.
        """
        return self.zgroup['data'].shape

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
        The array name.
        """
        return self.get_name()

    @property
    def unit(self):
        """
        The array data unit.
        """
        return self.get_unit()

    @property
    def description(self):
        """
        The array description.
        """
        return self.get_description()

    @property
    def latex_name(self):
        """
        The LaTeX representation for the array name.
        """
        return self.get_latex_name()

    @property
    def latex_unit(self):
        """
        The LaTeX representation for array data units.
        """
        return self.get_latex_unit()

    @property
    def dim_names(self):
        """
        Tuple containing the ordered dimension names.
        """
        return tuple(self.get_name(dim=k) for k in range(len(self.zgroup['data'].shape)))

    @property
    def dim_units(self):
        """
        Tuple containing the ordered dimension units.
        """
        return tuple(self.get_unit(dim=k) for k in range(len(self.zgroup['data'].shape)))

    @property
    def dim_descriptions(self):
        """
        Tuple containing the ordered dimension descriptions.
        """
        return tuple(self.get_description(dim=k) for k in range(len(self.zgroup['data'].shape)))

    @property
    def dim_latex_names(self):
        """
        Tuple containing the ordered LaTeX representation for dimension names.
        """
        return tuple(self.get_latex_name(dim=k) for k in range(len(self.zgroup['data'].shape)))

    @property
    def dim_latex_units(self):
        """
        Tuple containing the ordered LaTeX representation for dimension units.
        """
        return tuple(self.get_latex_unit(dim=k) for k in range(len(self.zgroup['data'].shape)))

    def validate(self):
        """
        Checks the consistency of the data and its associated attributes.
        This method verifies if all data in the underlying Zarr group is
        correctly defined and conforms to expected formats. If any inconsistencies
        or issues are found, the method raises appropriate errors to notify the user.

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
        zgroup = self.zgroup

        # Data type identifier
        if "_ROCKVERSE_DATATYPE" not in zgroup.attrs:
            collective_raise(KeyError(f"Missing '_ROCKVERSE_DATATYPE' identifier in the zarr group attrs."))

        # data array must exist
        if 'data' not in zgroup:
            collective_raise(KeyError(f"Missing 'data' array in the zarr group."))

        # Every dimension array must exist
        missing_dims = [f"'dim_{k}'" for k in range(self.ndim) if f"dim_{k}" not in zgroup]
        if len(missing_dims) == 1:
            collective_raise(KeyError(f"Missing {missing_dims[0]} array in the zarr group."))
        elif len(missing_dims) == 2:
            collective_raise(KeyError(f"Missing {' and '.join(missing_dims)} arrays in the zarr group."))
        elif len(missing_dims) > 2:
            collective_raise(KeyError(f"Missing {', '.join(missing_dims[:-1])}, and {missing_dims[-1]} arrays in the zarr group."))

        # Every dimension array must be 1D
        not_1D = [f"'dim_{k}'" for k in range(self.ndim) if len(zgroup[f"dim_{k}"].shape) != 1]
        if len(not_1D) == 1:
            collective_raise(ValueError(f"Wrong shape in {not_1D[0]} array in the zarr group. Dimension arrays must be 1-D."))
        elif len(not_1D) == 2:
            collective_raise(ValueError(f"Wrong shape in {' and '.join(not_1D)} arrays in the zarr group. Dimension arrays must be 1-D."))
        elif len(not_1D) > 2:
            collective_raise(ValueError(f"Wrong shape in {', '.join(not_1D[:-1])}, and {not_1D[-1]} arrays in the zarr group. Dimension arrays must be 1-D."))

        # Shapes must match
        wrong_size = [f"len(dim_{k})={zgroup[f"dim_{k}"].shape[0]}" for k in range(self.ndim) if zgroup[f"dim_{k}"].shape[0] != self.shape[k]]
        if len(wrong_size) == 1:
            collective_raise(ValueError(f"{wrong_size[0]} does not match data shape={self.shape}."))
        elif len(wrong_size) == 2:
            collective_raise(ValueError(f"{' and '.join(wrong_size)} do not match data shape={self.shape}."))
        elif len(wrong_size) > 2:
            collective_raise(ValueError(f"{', '.join(wrong_size[:-1])}, and {wrong_size[-1]} do not match data shape={self.shape}."))

        # Array-specific attributes must be string
        for array in ['data',] + [f"dim_{k}" for k in range(self.ndim)]:
            for attr in ('name', 'unit', 'description', 'latex_name', 'latex_unit'):
                if attr in zgroup[array].attrs and not isinstance(zgroup[array].attrs[attr], str):
                    collective_raise(ValueError(f"zgroup['{array}'].attrs['{attr}'] must be a string."))

        # Not array-specific attributes won't be tested...
        return


    def h5_dump(self, file_object, path):
        """
        Dumps the contents of the RockVerse array into an HDF5 file.
        This method exports the array data and its associated attributes from the
        RockVerse array into an HDF5 file at the specified path. It creates a
        group in the HDF5 file and stores the data array along with its metadata.
        The resulting HDF5 group will reflect the underlying zarr group:

        .. code-block::

            GROUP "arraypath"
                |- ATTRIBUTE "_ROCKVERSE_DATATYPE" (string)
                |- DATASET "data"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "dim_0"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "dim_1"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "dim_2"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "dim_3"
                    .
                    .
                    .

        for as many dims as array dimensions. Attributes that are missing in the RockVerse array
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

            array_instance = rv.create_array(...)  # Create your array...
            with h5py.File('filename.h5', 'a') as fobj:
                array_instance.h5dump(fobj, '/myawesomearray')

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

        # Arrays and corresponding attributes
        for array in ['data',] + [f"dim_{k}" for k in range(self.ndim)]:
            subgrp = file_object.create_dataset(f"{path}/{array}", data=self.zgroup[array]) #<<<<<<< PARALELIZE!
            for k, v in self.zgroup[array].attrs.items():
                subgrp.attrs[k] = v



def create_array(data,
                 store,
                 path=None,
                 name=None,
                 unit=None,
                 description=None,
                 latex_name=None,
                 latex_unit=None,
                 dim_data=None,
                 dim_names=None,
                 dim_units=None,
                 dim_descriptions=None,
                 dim_latex_names=None,
                 dim_latex_units=None,
                 attrs=None,
                 overwrite=False,
                 **kwargs):
    """
    Create a RockVerse array from provided data at specified Zarr storage.

    Parameters
    ----------

    data : array-like
        The data to be stored in the array.
    store : str or zarr.storage.StoreLike
        The storage location for the array.
    path : str, optional
        The path within the store where the array will be saved.
    name : str, optional
        The name of the array.
    unit : str, optional
        The unit of the array data.
    description : str, optional
        A description of the array.
    latex_name : str, optional
        The LaTeX representation of the array name.
    latex_unit : str, optional
        The LaTeX representation of the array unit.
    dim_data : tuple or list, optional
        Data for dimensions. The number of elements should match the shape of the data array.
        Each element must be an 1D array-like with the dimension coordinates.
    dim_names : tuple or list, optional
        Names for dimensions. The number of elements should match the shape of the data array.
        Each element must be a string with the dimension names.
    dim_units : tuple or list, optional
        Units for dimension data. The number of elements should match the shape of the data array.
        Each element must be a string with the dimension data unit.
    dim_descriptions : tuple or list, optional
        Description for dimension data. The number of elements should match the shape of the data array.
        Each element must be a string with the dimension description.
    dim_latex_names : tuple or list, optional
        LaTeX names for dimension data. The number of elements should match the shape of the data array.
        Each element must be a string with the LaTeX representation of dimension name.
    dim_latex_units : tuple or list, optional
        LaTeX units for dimension data. The number of elements should match the shape of the data array.
        Each element must be a string with the LaTeX representation of dimension data unit.
    attrs : dict, optional
        Additional attributes to be stored with the array.
    overwrite : bool, optional
        If True, deletes the store/path content before creating the new array.
    **kwargs
        Keyword arguments to be passed to the underlying Zarr group creation function.

    Returns
    -------

    Array
        An instance of the RockVerse Array class representing the created array.
    """

    #CHUNKS?
    #PARALLEL I/O?
    #PARALLEL __GETITEM__
    #PARALLEL __SETITEM__
    #NÂO CRIAR DENTRO DE STORE QUE JA CONTENHA ROCKVERSE DATA?

    # Check for valid entries ----------------------------------
    _assert.array_like('data', data)
    for varname, var in zip(('path', 'name', 'unit', 'description', 'latex_name', 'latex_unit'),
                            (path, name, unit, description, latex_name, latex_unit)):
        if var is not None:
            _assert.string(varname, var)

    shape = data.shape
    for varname, var in zip(('dim_data', 'dim_names', 'dim_units', 'dim_descriptions', 'dim_latex_names', 'dim_latex_units'),
                            (dim_data, dim_names, dim_units, dim_descriptions, dim_latex_names, dim_latex_units)):
        if var is not None:
            _assert.iterable.tuple_or_list(varname, var)
            _assert.iterable.length(varname, var, len(shape))
            if varname != 'dim_data':
                _assert.iterable.ordered_string_or_none(varname, var)
            else: # dim_data
                for k, v in enumerate(var):
                    if v is not None and not isinstance(v, (list, tuple, np.ndarray)):
                        collective_raise(ValueError(f'Elements in {varname} must be list, tuple or 1D Numpy arrays.'))
                    if v is not None and isinstance(v, np.ndarray) and len(v.shape) != 1:
                        collective_raise(ValueError(f'Elements in {varname} must be list, tuple or 1D Numpy arrays.'))
                    if v is not None and len(v) != shape[k]:
                        collective_raise(ValueError(f'len(dim_data[{k}])={len(dim_data[k])} does not match data.shape[{k}]={data.shape[k]}.'))

    # Dimension names must be unique
    if dim_names:
        for k1, name1 in enumerate(dim_names):
            if any(name1 and (name2 == name1) and (k2 != k1) for k2, name2 in enumerate(dim_names)):
                collective_raise(ValueError(f'Invalid dim_names={dim_names}: dimension names must be unique.'))

    if attrs is not None:
        _assert.dictionary('attrs', attrs)
        kwargs['attrs'] = attrs
    _assert.boolean('overwrite', overwrite)
    kwargs['overwrite'] = overwrite

    # Create the Zarr group and populate the data --------------
    kwargs['store'] = store
    kwargs['path'] = path
    zgroup = zarr.create_group(**kwargs)
    zgroup.attrs['_ROCKVERSE_DATATYPE'] = 'Array'

    # Should be done in parallel <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    data_attrs = {}
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

    zgroup.create_array(f"data",
                        shape=data.shape,
                        chunks=data.shape, # should be possible <<<<<<<<<<
                        dtype=data.dtype,
                        overwrite=overwrite,
                        attributes=data_attrs)
    zgroup[f"data"][...] = data

    # Should be done by rank 0... <<<<<<<<<<<<<<<<<<<<<<<<<
    for k in range(len(shape)):
        dim_attrs = {}
        if dim_names is not None and dim_names[k]:
            dim_attrs['name'] = dim_names[k]
        else:
            dim_attrs['name'] = f"dim_{k}"
        if dim_units is not None and dim_units[k]:
            dim_attrs['unit'] = dim_units[k]
        if dim_descriptions is not None and dim_descriptions[k]:
            dim_attrs['description'] = dim_descriptions[k]
        if dim_latex_names is not None and dim_latex_names[k]:
            dim_attrs['latex_name'] = dim_latex_names[k]
        if dim_latex_units is not None and dim_latex_units[k]:
            dim_attrs['latex_unit'] = dim_latex_units[k]

        if dim_data is not None and dim_data[k] is not None:
            dim_data_k = np.array(dim_data[k])
        else:
            dim_data_k = np.arange(shape[k])

        zgroup.create_array(f"dim_{k}",
                            shape=dim_data_k.shape,
                            chunks=dim_data_k.shape, # no chunks in dim data
                            dtype=dim_data_k.dtype,
                            overwrite=overwrite,
                            attributes=dim_attrs)
        zgroup[f"dim_{k}"][...] = dim_data_k

    return Array(zgroup)


#>>>>>>>>>>>>>> PARALELIZE! READ BY CHUNKS, even when not chunked but large dataset
def load_array_from_h5_file(fobj, h5path, store, path=None, overwrite=False, **kwargs):

    """
    Loads a RockVerse array from an HDF5 file.
    This function reads an existing RockVerse array stored in an HDF5 file and
    creates a corresponding Array object in the specified Zarr storage.

    The data in the HDF5 file is expected to be in a particular format:

    .. code-block::

            GROUP "arraypath"
                |- ATTRIBUTE "_ROCKVERSE_DATATYPE" (string)
                |- DATASET "data"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "dim_0"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "dim_1"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "dim_2"
                    |- DATA (array)
                    |- ATTRIBUTE "description" (string)
                    |- ATTRIBUTE "latex_name" (string)
                    |- ATTRIBUTE "latex_unit" (string)
                    |- ATTRIBUTE "name" (string)
                    |- ATTRIBUTE "unit" (string)
                |- DATASET "dim_3"
                    .
                    .
                    .

    for as many dims as array dimensions. Attributes are optional.
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
    Array
        An instance of the RockVerse Array class representing the loaded array.

    Raises
    ------
    KeyError
        If the specified HDF5 path is not found or if any expected datasets or attributes
        are missing from the HDF5 group.
    ValueError
        If the loaded data or attributes do not conform to expected formats or dimensions.
    TypeError
        If the specified HDF5 path does not point to a valid RockVerse array group.

    Example
    -------
    Load a RockVerse array from an HDF5 file:

    .. code-block:: python

        import h5py
        import rockverse as rv
        with h5py.File('filename.h5', 'r') as fobj:
            array_instance = rv.core.load_array_from_h5_file(
                fobj, h5path='/myawesomearray', store='/path/to/zarr/store')

    This will load the contents in '/myawesomearray' from the HDF5 file and store it
    in the specified Zarr storage as a RockVerse Array.
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
    if group.attrs["_ROCKVERSE_DATATYPE"] != "Array":
        collective_raise(TypeError(f"fobj['{h5path}']: expected RockVerse Array type."))

    # group['data'] must exist
    if 'data' not in group:
        collective_raise(KeyError(f"Missing 'data' dataset in fobj['{h5path}']."))
    data = group['data']

    # data must be a HDF5 dataset
    if not isinstance(data, h5py.Dataset):
        collective_raise(TypeError(f"fobj['{h5path}/data'] expected to be a Dataset. Found {type(data)}."))

    # Every dimension array must exist
    missing_dims = [f"'dim_{k}'" for k in range(data.ndim) if f"dim_{k}" not in group]
    if len(missing_dims) == 1:
        collective_raise(KeyError(f"Missing {missing_dims[0]} Dataset in fobj['{h5path}']."))
    elif len(missing_dims) == 2:
        collective_raise(KeyError(f"Missing {' and '.join(missing_dims)} Datasets in fobj['{h5path}']."))
    elif len(missing_dims) > 2:
        collective_raise(KeyError(f"Missing {', '.join(missing_dims[:-1])}, and {missing_dims[-1]} Datasets in fobj['{h5path}']."))

    # Every dimension array must be 1D
    not_1D = [f"fobj['{h5path}/dim_{k}']" for k in range(data.ndim) if len(group[f"dim_{k}"].shape) != 1]
    if len(not_1D) == 1:
        collective_raise(ValueError(f"Wrong shape in {not_1D[0]} Dataset. Dimension arrays must be 1-D."))
    elif len(not_1D) == 2:
        collective_raise(ValueError(f"Wrong shape in {' and '.join(not_1D)} Datasets. Dimension arrays must be 1-D."))
    elif len(not_1D) > 2:
        collective_raise(ValueError(f"Wrong shape in {', '.join(not_1D[:-1])}, and {not_1D[-1]} Datasets. Dimension arrays must be 1-D."))

    # Shapes must match
    wrong_size = [f"len(dim_{k})={group[f"dim_{k}"].shape[0]}" for k in range(data.ndim) if group[f"dim_{k}"].shape[0] != data.shape[k]]
    if len(wrong_size) == 1:
        collective_raise(ValueError(f"fobj['{h5path}']: {wrong_size[0]} does not match data shape={data.shape}."))
    elif len(wrong_size) == 2:
        collective_raise(ValueError(f"fobj['{h5path}']: {' and '.join(wrong_size)} do not match data shape={data.shape}."))
    elif len(wrong_size) > 2:
        collective_raise(ValueError(f"fobj['{h5path}']: {', '.join(wrong_size[:-1])}, and {wrong_size[-1]} do not match data shape={self.shape}."))

    # Array-specific attributes must be string
    for array in ['data',] + [f"dim_{k}" for k in range(data.ndim)]:
        for attr in ('name', 'unit', 'description', 'latex_name', 'latex_unit'):
            if attr in group[array].attrs and not isinstance(group[array].attrs[attr], str):
                collective_raise(ValueError(f"fobj['{h5path}/{array}'].attrs['{attr}'] must be a string."))

    # Start importing
    rvarray = create_array(data=group['data'][...], #<<<<<<<<<<<< PARALELIZE!
                           store=store,
                           path=path,
                           dim_data=[group[f'dim_{k}'][...] for k in range(data.ndim)],
                           overwrite=overwrite,
                           **kwargs)
    for array in ['data',] + [f"dim_{k}" for k in range(data.ndim)]:
        for k, v in group[array].attrs.items():
            rvarray.zgroup[array].attrs[k] = v

    return rvarray


if __name__ == "__main__":
    import numpy as np
    import h5py
    self=create_array(
        data=np.random.rand(2,2,2),
        #store=r"C:\Users\GOB7\Downloads\test",
        store='/u/gob7/test.zarr',
        path="testpath",
        name='test array',
        unit='m/s',
        description="UMA DESC",
        latex_name=r"$ERF$",
        latex_unit="MM",
        dim_data=([1, 2], [2, 2], None),
        dim_names=("QQ", 'y','z'),
        dim_units=('km', "S", "F"),
        dim_descriptions=("UM", "DOIS", "WW"),
        dim_latex_names=(r"$r$", r"$i$", r"$p$"),
        dim_latex_units=('a', '', '.'),
        attrs=None,
        overwrite=True)
    self.validate()

    filename = '/u/gob7/test.h5'
    with h5py.File(filename, 'w') as fobj:
        self.h5_dump(fobj, '/myawesomearray')

    store='/u/gob7/test2.zarr'
    h5path = '/myawesomearray'
    path=None
    overwrite=True
    kwargs={}
    with h5py.File(filename, mode='r') as fobj:
        self2 = load_array_from_h5_file(fobj, h5path, store, path=None, overwrite=True)

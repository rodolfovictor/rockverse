"""
This module provides the basic variable classes and creation functions
for all data types handled in RockVerse.

It includes the `Array` class, which represents generic N-dimensional arrays with
coordinates and associated metadata (similar to the
`Xarray project <https://docs.xarray.dev/en/stable/>`, for example),
and the `Group` class, which facilitates generic data grouping and hierarchization.

These classes are built upon `Zarr <https://zarr.readthedocs.io>`_ arrays and groups,
and are tailored for high-performance parallel computation across multiple CPUs or GPUs
using MPI (Message Passing Interface), with optimized I/O operations and memory usage.
"""

import os
import zarr
from rockverse import _assert
from rockverse.errors import collective_raise

# TODO PARALELLIZE EVERYTHING

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
        This class should not be directly instantiated.
        Use the :func:`create_array` function instead.

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
        self.zgroup = zgroup


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
        return self._get_attribute('unit', dim=dim)

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
        self._set_attribute(attr_name='unit', attr_value=v, attr_type=str, dim=dim)

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

    @property
    def values(self):
        AIAIAI
        return self.zgroup['data']

def create_array(data,
                 store,
                 path=None,
                 name=None,
                 unit=None,
                 description=None,
                 latex_name=None,
                 latex_unit=None,
                 dim_data=None,
                 dim_name=None,
                 dim_unit=None,
                 dim_description=None,
                 dim_latex_name=None,
                 dim_latex_unit=None,
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
    dim_name : tuple or list, optional
        Names for dimensions. The number of elements should match the shape of the data array.
        Each element must be a string with the dimension names.
    dim_unit : tuple or list, optional
        Units for dimension data. The number of elements should match the shape of the data array.
        Each element must be a string with the dimension data unit.
    dim_description : tuple or list, optional
        Description for dimension data. The number of elements should match the shape of the data array.
        Each element must be a string with the dimension description.
    dim_latex_name : tuple or list, optional
        LaTeX names for dimension data. The number of elements should match the shape of the data array.
        Each element must be a string with the LaTeX representation of dimension name.
    dim_latex_unit : tuple or list, optional
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
    for varname, var in zip(('dim_data', 'dim_name', 'dim_unit', 'dim_description', 'dim_latex_name', 'dim_latex_unit'),
                            (dim_data, dim_name, dim_unit, dim_description, dim_latex_name, dim_latex_unit)):
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
    for k1, name1 in enumerate(dim_name):
        if any(name1 and (name2 == name1) and (k2 != k1) for k2, name2 in enumerate(dim_name)):
            collective_raise(ValueError(f'Invalid dim_data={dim_name}: dimension names must be unique.'))

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
        if dim_name is not None and dim_name[k]:
            dim_attrs['name'] = dim_name[k]
        else:
            dim_attrs['name'] = f"dim_{k}"
        if dim_unit is not None and dim_unit[k]:
            dim_attrs['unit'] = dim_unit[k]
        if dim_description is not None and dim_description[k]:
            dim_attrs['description'] = dim_description[k]
        if dim_latex_name is not None and dim_latex_name[k]:
            dim_attrs['latex_name'] = dim_latex_name[k]
        if dim_latex_unit is not None and dim_latex_unit[k]:
            dim_attrs['latex_unit'] = dim_latex_unit[k]

        if dim_data is not None and dim_data[k]:
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


if __name__ == "__main__":
    import numpy as np
    self=create_array(
        data=np.random.rand(2,2,2),
        #store=r"C:\Users\GOB7\Downloads\test",
        store='/u/gob7/test',
        path="testpath",
        name='test array',
        unit='m/s',
        description=None,
        latex_name=None,
        latex_unit=None,
        dim_data=([1, 2], [2, 2], None),
        dim_name=(None, 'y','z'),
        dim_unit=('km', None, None),
        dim_description=None,
        dim_latex_name=None,
        dim_latex_unit=None,
        attrs=None,
        overwrite=True)

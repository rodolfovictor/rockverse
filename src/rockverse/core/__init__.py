import zarr
from rockverse import _assert
from rockverse.errors import collective_raise

# TODO PARALELLIZE EVERYTHING


from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs



class Array:

    "Generic N-d array with coordinates."

    def __init__(self, zgroup):
        """
        Generic informed array.
        zgroup: Zarr group
        """
        _assert.zarr_group('zgroup', zgroup)
        self.zgroup = zgroup

    def _get_array(self, dim=None):
        if dim is None:
            return self.zgroup['data']
        if f'axis_{dim}' in self.zgroup:
            return self.zgroup[f'axis_{dim}']
        collective_raise(KeyError(f"dim={dim} is not a valid axis dimension index for this array. "
                                  f"Expected non negative integer < {len(self.zgroup['data'].shape)}."))

    def _get_attribute(self, attr_name, dim=None):
        array = self._get_array(dim)
        attr_value = None
        if mpi_rank == 0:
            if attr_name in array.attrs:
                attr_value = array.attrs[attr_name]
        attr_value = comm.bcast(attr_value, root=0)
        return attr_value

    def _set_attribute(self, attr_name, attr_value, attr_type, dim=None):
        _assert.condition.non_negative_integer('dim', dim)
        array = self._get_array(dim)
        str_type = 'string' if attr_type == str else attr_type
        if not isinstance(attr_value, attr_type):
            collective_raise(ValueError(f"Expected {str_type} for {attr_name}."))
        if mpi_rank == 0:
            array.attrs[attr_name] = attr_value
        comm.barrier()

    def get_name(self, dim=None):
        return self._get_attribute('name', dim=dim)

    def set_name(self, v, dim=None):
        self._set_attribute(attr_name='name', attr_value=v, attr_type=str, dim=dim)

    def get_unit(self, dim=None):
        return self._get_attribute('unit', dim=dim)

    def set_unit(self, v):
        self._set_attribute(attr_name='unit', attr_value=v, attr_type=str, dim=dim)

    def get_description(self, dim=None):
        return self._get_attribute('description', dim=dim)

    def set_description(self, v, dim=None):
        self._set_attribute(attr_name='description', attr_value=v, attr_type=str, dim=dim)

    def get_latex_name(self, dim=None):
        return self._get_attribute('latex_name', dim=dim)

    def set_latex_name(self, v, dim=None):
        self._set_attribute(attr_name='latex_name', attr_value=v, attr_type=str, dim=dim)

    def get_latex_unit(self, dim=None):
        return self._get_attribute('unit', dim=dim)

    def set_latex_unit(self, v, dim=None):
        self._set_attribute(attr_name='unit', attr_value=v, attr_type=str, dim=dim)


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
                 axis_data=None,
                 axis_name=None,
                 axis_unit=None,
                 axis_description=None,
                 axis_latex_name=None,
                 axis_latex_unit=None,
                 attrs=None,
                 overwrite=False,
                 **kwargs):
    '''
    Create RockVerse array from data at store.
    CHUNKS?
    PARALLEL I/O?
    PARALLEL __GETITEM__
    PARALLEL __SETITEM__
    NÂO CRIAR DENTRO DE STORE QUE JA CONTENHA ROCKVERSE DATA?
    '''

    # Check for valid entries ----------------------------------
    _assert.array_like('data', data)
    for varname, var in zip(('path', 'name', 'unit', 'description', 'latex_name', 'latex_unit'),
                            (path, name, unit, description, latex_name, latex_unit)):
        if var is not None:
            _assert.string(varname, var)

    shape = data.shape
    for varname, var in zip(('axis_data', 'axis_name', 'axis_unit', 'axis_description', 'axis_latex_name', 'axis_latex_unit'),
                            (axis_data, axis_name, axis_unit, axis_description, axis_latex_name, axis_latex_unit)):
        if var is not None:
            _assert.iterable.tuple_or_list(varname, var)
            _assert.iterable.length(varname, var, len(shape))
            if varname != 'axis_data':
                _assert.iterable.ordered_string_or_none(varname, var)
            else: # axis_data
                for k, v in enumerate(var):
                    if v is not None and not isinstance(v, (list, tuple, np.ndarray)):
                        collective_raise(ValueError(f'Elements in {varname} must be list, tuple or 1D Numpy arrays.'))
                    if v is not None and isinstance(v, np.ndarray) and len(v.shape) != 1:
                        collective_raise(ValueError(f'Elements in {varname} must be list, tuple or 1D Numpy arrays.'))
                    if v is not None and len(v) != shape[k]:
                        collective_raise(ValueError(f'len(axis_data[{k}])={len(axis_data[k])} does not match data.shape[{k}]={data.shape[k]}.'))

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
        ax_attrs = {}
        if axis_name is not None and axis_name[k]:
            ax_attrs['name'] = axis_name[k]
        if axis_unit is not None and axis_unit[k]:
            ax_attrs['unit'] = axis_unit[k]
        if axis_description is not None and axis_description[k]:
            ax_attrs['description'] = axis_description[k]
        if axis_latex_name is not None and axis_latex_name[k]:
            ax_attrs['latex_name'] = axis_latex_name[k]
        if axis_latex_unit is not None and axis_latex_unit[k]:
            ax_attrs['latex_unit'] = axis_latex_unit[k]

        if axis_data is not None and axis_data[k]:
            ax_data = np.array(axis_data[k])
        else:
            ax_data = np.arange(shape[k])

        zgroup.create_array(f"axis_{k}",
                            shape=ax_data.shape,
                            chunks=ax_data.shape, # no chunks in axis data
                            dtype=ax_data.dtype,
                            overwrite=overwrite,
                            attributes=ax_attrs)
        zgroup[f"axis_{k}"][...] = ax_data

    return Array(zgroup)



import numpy as np
self=create_array(
    data=np.random.rand(2,2,2),
store=r"C:\Users\GOB7\Downloads\test",
path="Rodolfo",
name='test array',
unit='m/s',
description=None,
latex_name=None,
latex_unit=None,
axis_data=([1, 2], [2,2], None),
axis_name=('x', 'y','z'),
axis_unit=None,
axis_description=None,
axis_latex_name=None,
axis_latex_unit=None,
attrs=None,
overwrite=True)

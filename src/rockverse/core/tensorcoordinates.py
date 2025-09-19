from rockverse import _assert
from rockverse.errors import collective_raise, collective_only_rank0_runs
from rockverse.core.attributes import Attributes
from rockverse.core.parallelarray import ParallelArray
from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs


class TensorCoordinateSet:
    """
    Represents the collection of tensor coordinates in a :class:`TensorField` object.
    It allows for easy access to the attributes of each coordinate and
    facilitates iteration over the coordinate set.

    The `TensorCoordinateSet` can be indexed to retrieve specific :class:`TensorCoordinate`
    objects by either their index or name. For example:

    .. code-block:: python

        coord0 = tensorfield.coordinates[0] # to get the first coordinate
        xcomp = tensorfield.coordinates['x-comp'] # to get the coordinate with the name 'x-comp'.

    .. note::
        This class should not be instantiated directly. It is managed by RockVerse
        :ref:`creation functions <core module creation functions>`.

    Parameters
    ----------
    zgroup : zarr.group.Group
        An existing Zarr group that contains the coordinates' data and attributes.
    """

    def __init__(self, zgroup):
        self._zgroup = zgroup

    @property
    def zgroup(self):
        """
        The Zarr group containing the parent tensor field data.
        """
        return self._zgroup

    @property
    def array_keys(self):
        """
        A sorted tuple of names for all coordinate arrays in the TensorField Zarr group.
        """
        return tuple(sorted(k for k in self.zgroup.array_keys() if k.startswith('coord_')))

    def __len__(self):
        return len(self.array_keys)


    def _get_attr(self, key):
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = tuple(self.zgroup[k].attrs.get(key) for k in self.array_keys)
        comm.barrier()
        value = comm.bcast(value, root=0)
        return value

    @property
    def names(self):
        """
        A sorted tuple of all coordinate names in the TensorField.
        """
        return self._get_attr('name')

    @property
    def units(self):
        """
        A sorted tuple of all coordinate data units in the TensorField.
        """
        return self._get_attr('unit')

    @property
    def descriptions(self):
        """
        A sorted tuple of all coordinate descriptions in the TensorField.
        """
        return self._get_attr('description')

    @property
    def latex_names(self):
        """
        A sorted tuple of all LaTeX representations for the coordinate names.
        """
        return self._get_attr('latex_name')

    @property
    def latex_units(self):
        """
        A sorted tuple of all LaTeX representations for the coordinate data units.
        """
        return self._get_attr('latex_unit')


    def get_plot_labels(self, unit=True, latex=True):
        """
        A sorted tuple with plot labels for each coordinate.

        Parameters
        ----------
        unit : bool, optional
            If True, use unit in the resulting string -> `'name (unit)'`.
            If False, discard unit -> `'name'`.
            Default is True.
        latex : bool, optional
            If True, use the LaTeX reprentations instead of the plain ones.
            Default is True.

        Returns
        -------
        tuple
            The resulting strings to be used as a plot labels.
        """
        return tuple(self[k].get_plot_label(unit=unit, latex=latex) for k in range(len(self)))


    def __getitem__(self, index):
        if index in range(len(self.names)):
            return TensorCoordinate(self.zgroup, index=index)
        if index in self.names:
            names = self.names
            if len([n for n in names if n == index]) > 1:
                collective_raise(KeyError(f'Coordinate names must be unique. Got {names}.'))
            return TensorCoordinate(self.zgroup, index=[k for k, v in enumerate(self.names) if v == index][0])
        collective_raise(KeyError(f'Expected key as integer in range({len(self.names)}) or string in {self.names}.'))



class TensorCoordinate:

    """
    Represents a coordinate of a :class:`TensorField` object.
    This class encapsulates the functionality for managing the attributes and data
    associated with a specific coordinate in a tensor field.

    .. note::
        This class should not be instantiated directly. It is managed by RockVerse
        :ref:`creation functions <core module creation functions>`.

    Parameters
    ----------
    zgroup : zarr.group.Group
        An existing Zarr group that contains the data and attributes for this coordinate.
    index : int
        The index of the coordinate within the tensor field.
    """

    def __init__(self, zgroup, index):
        self._zgroup = zgroup
        self._array_name = f'coord_{index}'
        self._attrs = Attributes(zgroup[f'coord_{index}'])

    @property
    def attrs(self):
        """
        The collective metadata attributes associated with this coordinate
        array as an :class:`Attributes` object.
        """
        return self._attrs

    @property
    def zgroup(self):
        """
        The Zarr group containing the parent TensorField data.
        """
        return self._zgroup

    @property
    def array(self):
        """
        The ParallelArray associated with this coordinate data.
        """
        return ParallelArray(self.zgroup[self._array_name])


    @property
    def name(self):
        """
        Gets or sets the name of the coordinate.
        """
        return self.attrs.get('name')

    @property
    def unit(self):
        """
        Gets or sets the coordinate data unit.
        """
        return self.attrs.get('unit')

    @property
    def latex_name(self):
        """
        Gets or sets the LaTeX representation of the coordinate name.
        """
        return self.attrs.get('latex_name')

    @property
    def latex_unit(self):
        """
        Gets or sets the LaTeX representation of the coordinate data unit.
        """
        return self.attrs.get('latex_unit')

    @property
    def description(self):
        """
        Gets or sets the coordinate description.
        """
        return self.attrs.get('description')

    @name.setter
    def name(self, value):
        _assert.string('name', value)
        self.attrs['name'] = value

    @unit.setter
    def unit(self, value):
        _assert.string('unit', value)
        self.attrs['unit'] = value

    @latex_name.setter
    def latex_name(self, value):
        _assert.string('latex_name', value)
        self.attrs['latex_name'] = value

    @latex_unit.setter
    def latex_unit(self, value):
        _assert.string('latex_unit', value)
        self.attrs['latex_unit'] = value

    @description.setter
    def description(self, value):
        _assert.string('description', value)
        self.attrs['description'] = value

    def get_plot_label(self, unit=True, latex=True):
        """
        Build a string 'name (unit)' based on the coordinate attributes.

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

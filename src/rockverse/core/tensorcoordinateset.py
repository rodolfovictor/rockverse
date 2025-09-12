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

from rockverse.core.tensorcoordinate import TensorCoordinate

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
        This class should not be instantiated directly. It will be handled by the creation
        functions when creating a new TensorField instance.

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
        The Zarr group containing the parent TensorField data.
        """
        return self._zgroup

    @property
    def array_keys(self):
        """
        A sorted tuple of names for all coordinate arrays in the TensorField Zarr group.
        """
        return tuple(sorted(k for k in self.zgroup.array_keys() if k.startswith('coord_')))

    @property
    def names(self):
        """
        A sorted tuple of all coordinate names in the TensorField.
        """
        return tuple(self.zgroup[k].attrs['name'] if 'name' in self.zgroup[k].attrs else None for k in self.array_keys)

    @property
    def units(self):
        """
        A sorted tuple of all coordinate data units in the TensorField.
        """
        return tuple(self.zgroup[k].attrs['unit'] if 'unit' in self.zgroup[k].attrs else None for k in self.array_keys)

    @property
    def descriptions(self):
        """
        A sorted tuple of all coordinate descriptions in the TensorField.
        """
        return tuple(self.zgroup[k].attrs['description'] if 'description' in self.zgroup[k].attrs else None for k in self.array_keys)

    @property
    def latex_names(self):
        """
        A sorted tuple of all LaTeX representations for the coordinate names.
        """
        return tuple(self.zgroup[k].attrs['latex_name'] if 'latex_name' in self.zgroup[k].attrs else None for k in self.array_keys)

    @property
    def latex_units(self):
        """
        A sorted tuple of all LaTeX representations for the coordinate data units.
        """
        return tuple(self.zgroup[k].attrs['latex_unit'] if 'latex_unit' in self.zgroup[k].attrs else None for k in self.array_keys)

    def _exit_error(self):
        collective_raise(KeyError(f'Expected key as integer in range({len(self.names)}) or string in {self.names}.'))

    def __getitem__(self, index):
        if index in range(len(self.names)):
            return TensorCoordinate(self.zgroup, index=index)
        if index in self.names:
            return TensorCoordinate(self.zgroup, index=[k for k, v in enumerate(self.names) if v == index][0])
        self._exit_error()

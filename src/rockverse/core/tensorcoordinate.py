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
from rockverse.core.attributes import Attributes

comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs


class TensorCoordinate:

    """
    Represents a coordinate of a :class:`TensorField` object.
    This class encapsulates the functionality for managing the attributes and data
    associated with a specific coordinate in a tensor field.

    .. note::
        This class should not be instantiated directly. It will be handled by the creation
        functions when creating a new TensorField instance.

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
    def zgroup(self):
        """
        The Zarr group containing the parent TensorField data.
        """
        return self._zgroup

    @property
    def zarray(self):
        """
        The Zarr array associated with this coordinate data.
        """
        return self.zgroup[self._array_name]

    def _get_attribute(self, name):
        value = None
        if mpi_rank == 0:
            value = self.zarray.attrs[name] if name in self.zarray.attrs else None
        value = comm.bcast(value, root=0)
        return value

    def _set_attribute(self, name, value):
        _assert.string(name, value)
        if mpi_rank == 0:
            self.zarray.attrs[name] = value
        comm.barrier()
        return

    @property
    def name(self):
        """
        Gets or sets the name of the coordinate.
        """
        return self._get_attribute('name')

    @property
    def unit(self):
        """
        Gets or sets the coordinate data unit.
        """
        return self._get_attribute('unit')

    @property
    def latex_name(self):
        """
        Gets or sets the LaTeX representation of the coordinate name.
        """
        return self._get_attribute('latex_name')

    @property
    def latex_unit(self):
        """
        Gets or sets the LaTeX representation of the coordinate data unit.
        """
        return self._get_attribute('latex_unit')

    @property
    def description(self):
        """
        Gets or sets the coordinate description.
        """
        return self._get_attribute('description')

    @name.setter
    def name(self, value):
        return self._set_attribute('name', value)

    @unit.setter
    def unit(self, value):
        return self._set_attribute('unit', value)

    @latex_name.setter
    def latex_name(self, value):
        return self._set_attribute('latex_name', value)

    @latex_unit.setter
    def latex_unit(self, value):
        return self._set_attribute('latex_unit', value)

    @description.setter
    def description(self, value):
        return self._set_attribute('description', value)

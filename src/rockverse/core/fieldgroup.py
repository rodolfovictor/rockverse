from rockverse.core.group import Group, create_group
from rockverse.core.coordinates import Coordinate, CoordinateSet
from rockverse.errors import collective_raise

from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

class FieldGroup(Group):
    """
    A specialized RockVerse Group for managing collections of fields (ScalarField,
    VectorField, TensorField) sharing a common CoordinateSet.

    This class provides synchronized management of multiple fields defined over the
    same coordinate space, ensuring metadata consistency and parallel-safe operations
    across MPI ranks.

    .. note::
        This class should not be directly instantiated. Use the
        :func:`create_fieldgroup <rockverse.create_fieldgroup>` function instead.

    Parameters
    ----------
    zgroup : zarr.group.Group
        An existing Zarr group to be managed in parallel.
    """
    def __init__(self, zgroup):
        super().__init__(zgroup)
        self._coords = []

    def create_group(self, *args, **kwargs):
        collective_raise(NotImplementedError('FieldGroups cannot contain other groups.'))

    def _create_parents(self, path, *args, **kwargs):
        if path.find('/') > 0:
            collective_raise(ValueError(f'{path}: FieldGroups cannot contain other groups.'))

    def create_coordinate(self, path, data, overwrite=False, **kwargs):
        kwargs['path'] = path
        if 'name' not in kwargs:
            kwargs['name'] = path
        kwargs['data'] = data
        kwargs['overwrite'] = overwrite
        kwargs['parent_attrs'] = None
        super().create_coordinate(**kwargs)
        return self[path]

    @property
    def coords(self):
        """
        The :class:`CoordinateSet` describing the coordinates stored in the group.
        """
        return CoordinateSet(*[self[k] for k in self.coord_keys()])

    def create_array(self, path, coords, overwrite=False, **kwargs):
        if 'shape' in kwargs:
            collective_raise(TypeError("You can not pass 'shape' as a parameter. Array shape will be the coordinate space shape."))
        if not isinstance(coords, (tuple, list)) or not all(isinstance(k, str) for k in coords):
            collective_raise(ValueError("Expected tuple of strings for coords."))
        coords_names = self.coords.names
        if any(k not in coords_names for k in coords):
            collective_raise(ValueError(f"Values in coords must be in {coords_names}."))
        kwargs['shape'] = [self.coords[k].shape[0] for k in coords]
        super().create_array(path=path, overwrite=overwrite, parent_attrs=None, **kwargs)
        self[path].attrs['coords'] = coords
        return self[path]


def create_fieldgroup(store=None, path=None, overwrite=False, **kwargs):
    """
    Create a RockVerse FieldGroup at the specified storage location and path.

    Parameters
    ----------
    store : str, zarr.storage.StoreLike, or None, optional
        Storage location for the FieldGroup. Can be a file path, a Zarr-compatible
        store object, or None for in-memory storage. Default is None.
    path : str or None, optional
        Internal path within the store where the FieldGroup will be created.
        If None, the root path is used. Default is None.
    overwrite : bool, optional
        If True, existing data at the target location will be overwritten.
        Default is False.
    **kwargs
        Additional keyword arguments passed to the underlying create_group function.

    Returns
    -------
    FieldGroup
        The newly created FieldGroup instance with the copied coordinates.

    """
    group = create_group(store=store, path=path, overwrite=overwrite, **kwargs)
    group.attrs['_ROCKVERSE_DATATYPE'] = 'FieldGroup'
    return FieldGroup(group.zgroup)

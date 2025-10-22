from rockverse.core.group import Group, create_group
from rockverse.core.coordinates import Coordinate, CoordinateSet
from rockverse.errors import collective_raise

class FieldGroup(Group):
    """
    A specialized RockVerse Group for managing collections of fields (ScalarField,
    VectorField, TensorField) sharing a common CoordinateSet.

    This class provides synchronized management of multiple fields defined over the
    same coordinate space, ensuring metadata consistency and parallel-safe operations
    across MPI ranks.

    .. note::
        This class should not be directly instantiated. Use the
        :ref:`creation functions <core module creation functions>` instead.

    Parameters
    ----------
    zgroup : zarr.group.Group
        An existing Zarr group to be managed in parallel.
    """

    def __init__(self, zgroup, coords):
        super().__init__(zgroup)
        self.attrs['_ROCKVERSE_DATATYPE'] = 'FieldGroup'
        coords = [f'_coord{k}' for k in range(len(zgroup.array_keys())) if '_coord{k}' in zgroup.array_keys()]
        self._coords = CoordinateSet(coords)

    def create_group(*args, **kwargs):
        collective_raise(NameError('FieldGroups cannot contain other groups.'))

def create_fieldgroup(coords, store=None, path=None, overwrite=False, **kwargs):
    """
    Create a RockVerse FieldGroup at the specified storage location and path.

    Parameters
    ----------
    coords : CoordinateSet or sequence of array-like objects
        The coordinate set or list of array-like coordinate objects defining the
        coordinate space shared by all fields in the group. Passed coordinates will
        be copied into the created object (coordinates in the created object will
        be independent from the passedcoordinates).
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

    if ((not isinstance(coords, CoordinateSet)) or
        (not isinstance(coords, (tuple, list)) or
         not all(hasattr(k, '__array__') for k in coords)
        )):
        collective_raise(TypeError("Expected CoordinateSet or a list of array-like objects for coords."))

    group = create_group(store=store, path=path, overwrite=overwrite, **kwargs)

    # Coordinates will be copied
    for k in range(len(coords)):
        coord = coords[k]
        group.create_coordinate(path=f'_coord{k}',
                                data=coord[...],
                                attrs=coord.attrs.as_dict())
        if isinstance(coord, Coordinate):
            group.attrs.update(coord.attrs.as_dict())

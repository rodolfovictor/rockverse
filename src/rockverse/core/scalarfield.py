#%%
from rockverse import _assert
from rockverse.errors import collective_raise, collective_only_rank0_runs
from rockverse.core.parallelarray import ParallelArray
from rockverse.core.coordinates import CoordinateSpace, Coordinate, coordinate

import numpy as np

class ScalarField:
    """
    Represents a scalar-valued dataset defined over a multidimensional coordinate space.
    This class combines a ParallelArray holding scalar data with a CoordinateSpace
    defining the coordinates for each dimension. It supports flexible coordinate
    input and automatic coordinate creation when none are provided.

    Parameters
    ----------
    array : ParallelArray
        The multidimensional parallel array containing scalar data values.
    coords : CoordinateSpace or sequence of Coordinate objects or None
        The coordinates corresponding to each dimension of the array.
        If None, default index-based coordinates will be created automatically.
    """

    def __init__(self, array, coords):
        if not isinstance(array, ParallelArray):
            collective_raise(TypeError("Expected ParallelArray object for array."))

        if coords is not None and not isinstance(coords, CoordinateSpace) and not all(isinstance(k, Coordinate) for k in coords):
            collective_raise(TypeError("Expected CoordinateSpace or a list of Coordinate objects for coords."))

        self._array = array

        if coords is None:
            self._coords = CoordinateSpace(*[coordinate(np.arange(k, dtype=float)) for k in array.shape])
        elif isinstance(coords, CoordinateSpace):
            self._coords = coords
        else:
            self._coords = CoordinateSpace(*coords)

    @property
    def array(self):
        """
        The :class:`ParallelArray` containing the scalar data values.
        """
        return self._array

    @property
    def coords(self):
        """
        The :class:`CoordinateSpace` describing the coordinates for each dimension.
        """
        return self._coords


def scalarfield(array, coords=None):
    """
    Create a ScalarField instance.

    Parameters
    ----------
    array : ParallelArray
        The multidimensional parallel array containing scalar data.
    coords : CoordinateSpace or sequence of Coordinate objects or None, optional
        Coordinates corresponding to each dimension of the array.
        If None, default index-based coordinates will be created.

    Returns
    -------
    ScalarField
        A new ScalarField instance combining the array with the specified coordinates.
    """
    return ScalarField(array, coords)

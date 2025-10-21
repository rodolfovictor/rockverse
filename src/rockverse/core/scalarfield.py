#%%
from rockverse import _assert
from rockverse.errors import collective_raise, collective_only_rank0_runs
from rockverse.core.parallelarray import ParallelArray
from rockverse.core.coordinates import CoordinateSet, Coordinate, coordinate

import numpy as np

class ScalarField:
    """
    Represents a scalar-valued dataset defined over a multidimensional coordinate space.
    This class combines a ParallelArray holding scalar data with a CoordinateSet
    defining the coordinates for each dimension. It supports flexible coordinate
    input and automatic coordinate creation when none are provided.

    Parameters
    ----------
    array : ParallelArray
        The multidimensional parallel array containing scalar data values.
    coords : CoordinateSet or sequence of Coordinate objects or None
        The coordinates corresponding to each dimension of the array.
        If None, default index-based coordinates will be created automatically.
    """

    def __init__(self, array, coords=None):
        if not isinstance(array, ParallelArray):
            collective_raise(TypeError("Expected ParallelArray object for array."))

        if coords is not None and not isinstance(coords, CoordinateSet) and not all(isinstance(k, Coordinate) for k in coords):
            collective_raise(TypeError("Expected CoordinateSet or a list of Coordinate objects for coords."))

        self._array = array

        if coords is None:
            self._coords = CoordinateSet(*[coordinate(np.arange(k, dtype=float)) for k in array.shape])
        elif isinstance(coords, CoordinateSet):
            self._coords = coords
        else:
            self._coords = CoordinateSet(*coords)

    @property
    def array(self):
        """
        The :class:`ParallelArray` containing the scalar data values.
        """
        return self._array

    @property
    def coords(self):
        """
        The :class:`CoordinateSet` describing the coordinates for each dimension.
        """
        return self._coords

import numpy as np
from rockverse import _assert
from rockverse.errors import collective_raise, collective_only_rank0_runs
from rockverse.core.parallelarray import ParallelArray, array
from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs


class Coordinate(ParallelArray):

    """
    One-dimensional, numeric, sorted, and unchunked ParallelArray representing
    a single coordinate of TensorField-like objects.

    This class extends `ParallelArray` to specifically handle coordinate data
    associated with tensor fields. Coordinates must be one-dimensional, sorted
    in ascending or descending order, and stored as single chunks to ensure
    consistency and efficient parallel access.

    .. note::
        This class should not be instantiated directly. It is managed by
        RockVerse :ref:`creation functions <core module creation functions>`.

    Parameters
    ----------
    array : rockverse.core.ParallelArray
        The parallel array to be handled as the Coordinate object.
    """

    def __init__(self, array):
        if not isinstance(array, ParallelArray):
            collective_raise(TypeError("Expected ParallelArray object for array."))
        if len(array.shape) != 1:
            collective_raise(ValueError("Coordinate arrays must be one-dimensional."))
        if array.shape != array.chunks:
            collective_raise(ValueError("Coordinate arrays must not have multiple chunks."))
        if array.dtype.kind not in 'fui': # Numeric, not complex only
            collective_raise(TypeError("Expected integer or float data type."))

        is_sorted = False
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                diff = np.diff(array._zarray[...])
                is_sorted = np.all(diff>0) or np.all(diff<0)
        if not comm.bcast(is_sorted, root=0):
            collective_raise(ValueError("Coordinate arrays must be sorted and without repeated values."))

        super().__init__(array._zarray)
        self.attrs['_ROCKVERSE_DATATYPE'] = 'Coordinate'

    def closest_index(self, value):
        """
        Find the index of the coordinate element closest to the specified value.

        Parameters
        ----------
        value : numeric
            The value, in coordinate data units, to find the closest coordinate
            element to.

        Returns
        -------
        int
            The index of the coordinate element closest to the specified value.
        """
        index = 0
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                index = np.argmin(np.abs(self._zarray[...]-value))
        return comm.bcast(index, root=0)

    def closest_value(self, value):
        """
        Find the coordinate element closest to the specified value.

        Parameters
        ----------
        value : numeric
            The value, in coordinate data units, to find the closest coordinate
            element to.

        Returns
        -------
        value
            The the coordinate element closest to the specified value.
        """
        coord_value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                index = np.argmin(np.abs(self._zarray[...]-value))
                coord_value = self._zarray[index]
        return comm.bcast(coord_value, root=0)


    @property
    def is_equally_spaced(self):
        """
        Determine whether the coordinate values are equally spaced.

        Returns
        -------
        bool
            True if the coordinate values are equally spaced; False otherwise.
        """
        dx = np.diff(self[...])
        if len(dx) == 0:
            return True  # Single element coordinate is trivially equally spaced

        ref = dx[0]
        dtype = self.dtype

        if np.issubdtype(dtype, np.integer):
            return np.all(dx == ref)

        if np.issubdtype(dtype, np.inexact):
            tol = 2 * np.finfo(dtype).eps * np.abs(ref)
            return np.all(np.abs(dx - ref) <= tol)

        collective_raise(TypeError("Expected integer or float data type for coordinate."))


def coordinate(data, store=None, path=None, overwrite=False, **kwargs):
    """
    Create a Coordinate object from data.

    Parameters
    ----------
    data : array-like
        The source array whose data will populate the new coordinate.
    store : str, zarr.storage.StoreLike, or None, optional
        Storage location for the coordinate. Can be a file path, a Zarr-compatible
        store object, or None for in-memory storage.
    path : str or None, optional
        The path within the store where the array will be located. If None,
        the root path is used.
    overwrite : bool, optional
        If True, any existing data at the target location will be overwritten.
        Default is False.
    **kwargs
        Additional keyword arguments passed to the underlying create_array
        function. Keyword 'chunks' will be ignored, as Coordinate objects
        must have only one chunk.

    Returns
    -------
    coord : Coordinate
        The the coordinate object.
    """
    _assert.array_like('data', data)
    if len(data.shape) != 1:
        collective_raise(ValueError("Coordinate arrays must be one-dimensional."))
    if data.dtype.kind not in 'fui': # non complex numeric, only
        collective_raise(TypeError("data: expected integer or float data type."))
    kwargs['chunks'] = None
    kwargs['store'] = store
    kwargs['path'] = path
    kwargs['overwrite'] = overwrite
    return Coordinate(array(data, **kwargs))


class CoordinateSet:
    """
    Represents the collection of coordinate objects.

    This class provides indexed access to individual Coordinate instances
    either by integer index or by coordinate name. It also offers collective
    management of coordinate metadata attributes across MPI ranks, ensuring
    consistent views in parallel environments.

    Usage example:

    .. code-block:: python

        coord0 = some_field.coordinates[0] # Access by index
        xcomp = some_field.coordinates['x-comp'] # Access by name

    .. note::
        This class should not be instantiated directly. It is managed by RockVerse
        :ref:`creation functions <core module creation functions>`.

    Parameters
    ----------
    args :
        A sequence of :class:`Coordinate` objects.
    """

    def __init__(self, *args):
        if not all(isinstance(k, Coordinate) for k in args):
            collective_raise(TypeError("CoordinateSet input variables must be Coordinate objects."))
        self._coordinates = args

    def __len__(self):
        return len(self._coordinates)

    def _get_attr(self, key):
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = tuple(k._zarray.attrs.get(key, default=None) for k in self._coordinates)
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

    @property
    def is_equally_spaced(self):
        """
        A tuple of boolean values indicating whether each coordinate in the
        CoordinateSet is equally spaced.
        """
        return tuple(coord.is_equally_spaced for coord in self._coordinates)


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
        """
        Retrieve a Coordinate by integer index or coordinate name.

        Example:

        .. code-block:: python

            coord0 = some_field.coordinates[0] # Access by index
            xcomp = some_field.coordinates['x-comp'] # Access by name

        """
        names = self.names
        if len(names) == 0:
            collective_raise(IndexError(f'CoordinateSet is empty.'))
        if index in range(len(names)):
            return self._coordinates[index]
        if index in names:
            if len([n for n in names if n == index]) > 1:
                collective_raise(KeyError(f'Coordinate names must be unique. Got {names}.'))
            ind = [k for k, v in enumerate(names) if v == index][0]
            return self._coordinates[ind]
        if isinstance(index, str):
            collective_raise(KeyError(f"Coordinate name '{index}' not found."))
        collective_raise(KeyError(f'Expected key as integer in range({len(names)}) or string with coordinate name.'))

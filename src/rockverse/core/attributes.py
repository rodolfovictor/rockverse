"""
Provides the Attributes class for parallel management of Zarr array and group attributes.
"""

import pprint
from rockverse import _assert
from rockverse.errors import collective_only_rank0_runs, collective_raise

from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

class Attributes:
    """
    A dict-like interface for parallel management of attributes in RockVerse
    arrays and groups.

    This class extends the Zarr Attributes interface to provide synchronized
    access and modification of attributes in MPI environments. Only the MPI
    rank 0 process performs actual attribute operations, while other ranks
    participate in collective synchronization to maintain consistency and
    prevent race conditions.

    Attributes can be accessed, set, and queried similarly to a Python
    dictionary.

    .. note::
        This class should not be directly instantiated.
        It is managed by RockVerse
        :ref:`creation functions <core module creation functions>`.


    Parameters
    ----------
    zobj: zarr.core.array.Array, zarr.core.group.Group
        An existing Zarr array or group to be managed in parallel.
    """

    def __init__(self, zobj):
        _assert.zarr_array_or_group('zobj', zobj)
        self._zobj = zobj


    @property
    def zobj(self):
        """
        The underlying Zarr array or Zarr group managed by this Attributes instance.
        """
        return self._zobj


    def __getitem__(self, key):
        """
        Return `self[index]`.

        :bdg-primary:`MPI collective`
        """
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = self._zobj.attrs[key]
        value = comm.bcast(value, root=0)
        return value


    def get(self, key, default=None):
        """
        Return the value for the given key if it exists, otherwise return default.

        :bdg-primary:`MPI collective`

        Parameters
        ----------
        key : str
            The attribute key to retrieve.
        default : any, optional
            The value to return if the key is not found. Default is None.

        Returns
        -------
        value
            The attribute value associated with `key`, or `default` if key is not found.
        """
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                if key in self._zobj.attrs:
                    value = self._zobj.attrs[key]
                else:
                    value = default
        value = comm.bcast(value, root=0)
        return value


    def __setitem__(self, key, value):
        """
        Set the value for `self[index]`.

        :bdg-primary:`MPI collective`
        """
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                self._zobj.attrs[key] = value


    def __iter__(self):
        """
        Return an iterator over the attribute keys managed by this Attributes
        instance. This allows iteration like a standard dictionary over
        attribute keys.

        :bdg-primary:`MPI collective`

        """
        return self.keys()


    def __contains__(self, key):
        """
        Check if the specified key exists among the attributes.

        :bdg-primary:`MPI collective`

        """
        value = False
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = key in self._zobj.attrs
        value = comm.bcast(value, root=0)
        return value


    def __len__(self):
        """
        Return the number of attributes managed by this Attributes instance.

        :bdg-primary:`MPI collective`
        """
        value = 0
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = len(self._zobj.attrs)
        value = comm.bcast(value, root=0)
        return value


    def __repr__(self):
        """
        Return a string representation of the Attributes instance showing the
        underlying attributes.

        :bdg-primary:`MPI collective`

        Returns
        -------
        str
            A string representation of the Attributes object.
        """
        try:
            attrs_dict = self.asdict()
        except Exception:
            attrs_dict = "<unable to retrieve attributes>"
        return f"{self.__class__.__name__}({attrs_dict})"


    def __str__(self):
        """
        Return a pretty-printed string representation of the attributes.
        """
        try:
            attrs_dict = self.asdict()
        except Exception:
            return f"{self.__class__.__name__}(<unable to retrieve attributes>)"
        return pprint.pformat(attrs_dict)


    def asdict(self):
        """
        Retrieve all attributes as a standard Python dictionary.

        :bdg-primary:`MPI collective`

        """
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = {k: v for k, v in self._zobj.attrs.items()}
        value = comm.bcast(value, root=0)
        return value


    def keys(self):
        """
        Return an iterator over the attribute keys.

        :bdg-primary:`MPI collective`

        """
        keys = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                keys = list(self._zobj.attrs.keys())
        keys = comm.bcast(keys, root=0)
        return iter(keys)


    def values(self):
        """
        Return an iterator over the attribute values.

        :bdg-primary:`MPI collective`

        """
        values = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                values = list(self._zobj.attrs.values())
        values = comm.bcast(values, root=0)
        return iter(values)


    def items(self):
        """
        Return an iterator over the attribute (key, value) pairs.

        :bdg-primary:`MPI collective`

        """
        items = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                items = list(self._zobj.attrs.items())
        items = comm.bcast(items, root=0)
        return iter(items)


    def pop(self, key, default=None):
        """
        Remove the specified attribute and return its value.
        If the key is not found, return the default value if provided,
        otherwise raise a KeyError.

        :bdg-primary:`MPI collective`

        Parameters
        ----------
        key : str
            The attribute key to remove.
        default : any, optional
            The value to return if the key is not found. If not provided,
            a KeyError is raised when the key does not exist.

        Returns
        -------
        value
            The value associated with `key` before it was removed.

        Raises
        ------
        KeyError
            If the key is not found and no default is provided.
        """
        value = None
        found = False
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                if key in self._zobj.attrs:
                    value = self._zobj.attrs[key]
                    del self._zobj.attrs[key]
                    found = True
                else:
                    found = False
        found = comm.bcast(found, root=0)
        value = comm.bcast(value, root=0)
        if not found:
            if default is not None:
                return default
            else:
                collective_raise(KeyError(f"Key '{key}' not found in attributes."))
        return value


    def clear(self):
        """
        Remove all attributes from the array or group.

        :bdg-primary:`MPI collective`

        """
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                keys = list(self._zobj.attrs.keys())
                for key in keys:
                    del self._zobj.attrs[key]
        comm.barrier()


    def update(self, other=None, **kwargs):
        """
        Update attributes from another dictionary or iterable of key-value pairs.

        :bdg-primary:`MPI collective`

        Parameters
        ----------
        other : dict or iterable of (key, value) pairs, optional
            Attributes to update. Can be a dictionary or any iterable of key-value pairs.
        **kwargs
            Additional key-value pairs to update.
        """
        if other is None:
            other = {}
        # Convert iterable of pairs to dict if needed
        if not isinstance(other, dict):
            other = dict(other)
        # Merge kwargs into other
        other.update(kwargs)
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                for key, value in other.items():
                    self._zobj.attrs[key] = value
        comm.barrier()

import fnmatch
from rockverse.errors import collective_raise
from rockverse.core.parallelarray import array
from rockverse.core.coordinates import coordinate
from rockverse.core.scalarfield import scalarfield
from rockverse.configure import config
mpi_comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs


def _print_parameter(param):
    """
    Format a parameter dictionary into a human-readable string.

    Parameters
    ----------
    param : dict
        Dictionary containing parameter information.

    Returns
    -------
    str
        Formatted string representation of the parameter.
    """
    str = f"{param['mnem']}"
    if param['value']:
        str = f"{str}: {param['value']}"
    if param['unit']:
        str = f"{str} {param['unit']}"
    if param['description']:
        str = f"{str} ({param['description']})"
    if 'association' in param and param['association']:
        str = f"{str} | {param['association']}"
    return str

def _print_data(data):
    """
    Format a data dictionary into a human-readable string.

    Parameters
    ----------
    data : dict
        Dictionary containing data information.

    Returns
    -------
    str
        Formatted string representation of the data.
    """
    str = f"{data['mnem']}"
    if data['unit']:
        str = f"{str}, {data['unit']}"
    if data['code']:
        str = f"{str} ({data['code']})"
    str = f"{str}:"
    if data['description']:
        str = f"{str} {data['description']}"
    if 'association' in data and data['association']:
        str = f"{str} | {data['association']}"
    return str

class LasSubSection():
    """
    Base class representing a collection of LAS section entries, either
    parameters or data.

    Provides methods for item access by index or mnemonic, pattern searching,
    and printing a hierarchical tree representation.

    Parameters
    ----------
    list_ : list of dict
        List of entry dictionaries representing parameters or data columns.
    type_ : str
        Type of entries, either 'parameters' or 'data'.
    """
    def __init__(self, list_, type_):
        self.entries = list_
        self.type = type_

    def tree(self, prepend=''):
        """
        Print a tree representation of the subsection entries.

        Parameters
        ----------
        prepend : str, optional
            String to prepend for indentation or formatting. Default is ''.

        Examples
        --------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas
        """
        if not self.entries:
            print(f"{prepend}|- <empty>")
        elif self.type == 'parameters':
            for k, entry in enumerate(self.entries):
                print(f"{prepend}|-[{k}] {_print_parameter(entry)}")
        elif self.type == 'data':
            for k, entry in enumerate(self.entries):
                print(f"{prepend}|-[{k}] {_print_data(entry)}")
        else:
            raise Exception('What is happening?...')


    def __getitem__(self, key):
        """
        Access an entry by integer index or mnemonic string.

        Parameters
        ----------
        key : int or str
            Index or mnemonic of the entry.

        Returns
        -------
        dict
            The entry dictionary corresponding to the key.

        Examples
        --------

        Retrieve the second entry in the Curve parameter section (index is zero-based):

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las(/path/to/las/file.las)
            param = las_data['Curve'].parameters[1]

        Retrieve the entry with the RHO mnemonic in the Curve data section:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las(/path/to/las/file.las)
            curve = las_data['Curve'].data['RHO']

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas


        """
        if isinstance(key, int):
            return self.entries[key]
        elif isinstance(key, str):
            ind = [k for k, v in enumerate(self.entries) if v['mnem'] == key]
            if not ind:
                raise KeyError(key)
            if len(ind) > 1:
                raise KeyError(f'multiple entries for {key}')
            return self.entries[ind[0]]

    def _find(self, pattern, prepend=''):
        """
        Find entries matching a pattern in their mnemonic using Unix shell-style
        wildcards.

        Parameters
        ----------
        pattern : str
            Pattern to match against entry mnemonics.
        prepend : str, optional
            String to prepend for formatting. Default is ''.

        Returns
        -------
        list of str
            Formatted strings of matching entries.
        """
        out = []
        matching_entries = [k for k, v in enumerate(self.entries) if fnmatch.fnmatch(v['mnem'], pattern)]
        if matching_entries:
            if self.type == 'parameters':
                for k in matching_entries:
                    out.append(f"{prepend}|-[{k}] {_print_parameter(self.entries[k])}")
            elif self.type == 'data':
                for k in matching_entries:
                    out.append(f"{prepend}|-[{k}] {_print_data(self.entries[k])}")
            else:
                raise Exception('What is happening?...')
        return out

    def find(self, pattern, prepend=''):
        """
        Look for all entries matching a pattern using Unix shell-style
        wildcards.

        Parameters
        ----------
        pattern : str
            Pattern to search for.
        prepend : str, optional
            String to prepend to each line in the output. Default is ''.

        Examples
        --------

        Find all mnemonics containing NMR in the name in the Curve parameters subsection:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las(/path/to/las/file.las)
            las_data['Curve'].parameters.find('*NMR*')

        Find all mnemonics starting with DT in the Curve data subsection:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las(/path/to/las/file.las)
            las_data['Curve'].data.find('DT*')

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        out = self._find(pattern, prepend)
        if out:
            print('\n'.join(out))
        else:
            print('<no match>')

class LasParam(LasSubSection):
    """
    Represents a LAS subsection containing parameter entries.

    .. note::
        This class should not be instantiated directly. It is properly created
        when importing a LAS file using the RockVerse :func:`read_las <rockverse.read_las>`
        function.
    """

    def __init__(self, list_):
        super().__init__(list_, 'parameters')


class LasData(LasSubSection):
    """
    Represents a LAS subsection containing data entries.

    .. note::
        This class should not be instantiated directly. It is properly created
        when importing a LAS file using the RockVerse :func:`read_las <rockverse.read_las>`
        function.
    """

    def __init__(self, list_):
        super().__init__(list_, 'data')

    def create_scalarfield(self, column, coordinate_column=None, **kwargs):
        """
        Create a RockVerse scalarfield object from a data entry.

        Parameters
        ----------
        column : int or str
            Index or mnemonic of the data column to use as scalarfield data.
        coordinate_column : int or str, optional
            Index or mnemonic of the coordinate column to use as coordinates.
            Defaults to 0 (the first entry in the LAS section).
        **kwargs
            Additional keyword arguments for coordinate and array creation.

        Returns
        -------
        ScalarField
            A RockVerse scalarfield object constructed from the specified columns.

        Examples
        --------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        array_dict = self[column]
        coordinate_dict = self[0] if coordinate_column is None else self[coordinate]
        array_dict = mpi_comm.bcast(array_dict, root=0)
        coordinate_dict = mpi_comm.bcast(coordinate_dict, root=0)

        top_path = '' if 'path' not in kwargs else kwargs['path']
        kwargs['path'] = f"{top_path}/coords/0" if top_path else "coords/0"
        coord = coordinate(data=coordinate_dict['value'],
                           name=coordinate_dict['mnem'],
                           unit=coordinate_dict['unit'],
                           description=coordinate_dict['description'],
                           **kwargs)
        kwargs['path'] = f"{top_path}/array" if top_path else "array"
        parray = array(data=array_dict['value'],
                       name=array_dict['mnem'],
                       unit=array_dict['unit'],
                       description=array_dict['description'],
                       **kwargs)
        parray.attrs['code'] = array_dict['code'] if 'code' in array_dict and array_dict['code'] else ''
        return scalarfield(parray, coords=(coord,))


class LasSection():
    """
    Represents a LAS section containing parameters and data entries.

    .. note::
        This class should not be instantiated directly. It is properly created
        when importing a LAS file using the RockVerse :func:`read_las <rockverse.read_las>`
        function.
    """

    def __init__(self, dict_):
        self._parameters = dict_['parameters'] if 'parameters' in dict_ else dict()
        self._data = dict_['data'] if 'data' in dict_ else dict()

    @property
    def parameters(self):
        """
        Return the parameters section as a :class:`LasParam` object.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        return LasParam(self._parameters)

    @property
    def data(self):
        """
        Return the data section as a :class:`LasData` object.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        return LasData(self._data)

    def tree(self, prepend=''):
        """
        Print a tree representation of the LAS section.

        Parameters
        ----------
        prepend : str, optional
            String to prepend to each line for indentation. Default is ''.

        Examples
        --------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        print(f"{prepend}|- parameters")
        self.parameters.tree(prepend=f'{prepend}|   ')
        print(f"{prepend}|- data")
        self.data.tree(prepend=f'{prepend}|   ')

    def _find(self, pattern, prepend=''):
        """
        Find entries matching a pattern in their mnemonic using Unix shell-style
        wildcards.

        Parameters
        ----------
        pattern : str
            Pattern to match against entry mnemonics.
        prepend : str, optional
            String to prepend for formatting. Default is ''.

        Returns
        -------
        list of str
            Formatted strings of matching entries.

        """
        out = []
        out_parameter = self.parameters._find(pattern)
        out_data = self.data._find(pattern)
        if out_parameter:
            out.append(f"{prepend}|- parameters:")
            out += [f"{prepend}|   {k}" for k in out_parameter]
        if out_data:
            out.append(f"{prepend}|- data:")
            out += [f"{prepend}|   {k}" for k in out_data]
        return out

    def find(self, pattern, prepend=''):
        """
        Print entries matching a pattern in parameters and data.

        Parameters
        ----------
        pattern : str
            Pattern to search for.
        prepend : str, optional
            String to prepend to each line for indentation. Default is ''.

        Examples
        --------

        Find all mnemonics in Curve section containing NMR in the name:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las(/path/to/las/file.las)
            las_data['Curve'].find('*NMR*')

        Find all mnemonics starting with DT:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las(/path/to/las/file.las)
            las_data['Curve'].find('DT*')

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas
        """
        out = self._find(pattern, prepend)
        if out:
            print('\n'.join(out))
        else:
            print('<no match>')


class Las():
    """
    Organizes the loaded LAS file content.

    .. note::
        This class should not be instantiated directly. It is properly created
        when importing a LAS file using the RockVerse :func:`read_las <rockverse.read_las>`
        function.
    """

    def __init__(self):
        self.dict = {}

    @property
    def version(self):
        """
        Return the LAS version in the original LAS file.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        return self.dict['_version']

    @property
    def initial_comments(self):
        """
        Return the initial comments from the LAS file.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        return self.dict['_initial_comments']

    def section_keys(self):
        """
        Return a list with the LAS section keys.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """

        dict_ = {'Well': self.dict['Well']}
        dict_.update({k: v for k, v in self.dict.items() if k not in ('_version', '_initial_comments')})
        return dict_.keys()

    def __getitem__(self, key):
        """
        Access a LAS section by key.

        Parameters
        ----------
        key : str
            The section name.

        Returns
        -------
        LasSection
            The corresponding section object.

        Examples
        --------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        if key == 'Well':
            return LasParam(self.dict[key])
        if key not in self.dict.keys():
            collective_raise(KeyError(key))
        return LasSection(self.dict[key])

    def tree(self):
        """
        Print a tree representation of the LAS file, including sections and comments.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        if self.dict['_initial_comments']:
            print(self.dict['_initial_comments'])

        print('|- Well')
        self['Well'].tree(prepend='|   ')

        sections = [k for k in self.dict.keys() if k not in ('Well', 'Other', '_initial_comments', '_version')]
        for sec in sections:
            print(f"|- {sec}")
            self[sec].tree(prepend='|   ')

        if 'Other' in self.dict.keys():
            print('|- Other:')
            print(self.dict['Other'])

    def find(self, pattern):
        """
        Look for all entries matching a pattern using Unix shell-style
        wildcards.

        Parameters
        ----------
        pattern : str
            Pattern to search for.

        Examples
        --------

        Find all mnemonics containing NMR in the name:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las(/path/to/las/file.las)
            las_data.find('*NMR*')

        Find all mnemonics starting with DT:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las(/path/to/las/file.las)
            las_data.find('DT*')

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        found_any = False
        for sec in self.section_keys():
            if sec != "Other":
                out = self[sec]._find(pattern, prepend="|   ")
                if out:
                    found_any = True
                    print(f"|- {sec}")
                    print("\n".join(out))
        if not found_any:
            print('<no match>')

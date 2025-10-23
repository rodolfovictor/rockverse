"""
Module for reading and parsing LAS (Log ASCII Standard) files.

This module provides functionality to read, parse, and structure LAS files
commonly used in well logging data. It supports LAS versions 2.0 and 3.0,
integrating with RockVerse's parallel data structures.
"""

import os
import fnmatch
from rockverse import __path__ as RVPATH
from rockverse._utils.text import load_text_file
from rockverse.las.exceptions import LasImportError
from rockverse.las.las2 import break_las2_line, assemble_las2_dict
from rockverse.las.las3 import break_las3_line, assemble_las3_dict
from rockverse.errors import collective_raise, collective_only_rank0_runs
import rockverse._assert as _assert
from rockverse.core.parallelarray import array
from rockverse.core.coordinates import coordinate, Coordinate, CoordinateSet
from rockverse.core.fieldgroup import FieldGroup, create_fieldgroup
from rockverse.core.scalarfield import ScalarField
from rockverse.configure import config
mpi_comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

def _lprint(object):
    if mpi_rank == 0:
        print(object)

def _get_first_comment_lines(lines):
    """
    Extract initial comment lines from LAS file lines.

    Parameters
    ----------
    lines : list of str
        Lines from a LAS file.

    Returns
    -------
    str
        Concatenated initial comment lines (without the leading '#'), or empty string if none.
    """
    initial_comments = []
    for line in lines:
        if line.startswith('#'):
            initial_comments.append(line[1:])
        else:
            break
    if initial_comments:
        initial_comments = ''.join(initial_comments)
    return initial_comments


def _split_sections(lines):
    """
    Parse LAS file lines into sections and extract metadata.

    Parameters
    ----------
    lines : list of str
        Lines from a LAS file.

    Returns
    -------
    imported_sections : dict
        Dictionary mapping section headers to their content lines.
    section_order : list
        List of section headers in the order they appeared.
    las_version : int or None
        LAS file version (2 or 3).
    las_wrap : bool or None
        Whether WRAP mode is enabled.
    las_delimiter : str
        Delimiter character used in the file (space by default).

    Raises
    ------
    LasImportError
        If the file structure violates LAS standards or has invalid sections.
    """

    # This function is only called by MPI rank 0.
    # raise does not need to be collective_raise

    las_version = None
    las_wrap = None
    las_delimiter = ' '
    break_las_line = None

    current_section = None
    sections = []
    data = []
    imported_sections = {}
    for line_number, line_original in enumerate(lines):
        line = line_original.strip()
        if not line:
            continue

        # Comment line
        if line.startswith("#"):
            continue

        # Section start ----------------------------------------
        if line.startswith("~"):

            # Flush accumulated data to the corresponding section and reset the accumulator
            if len(data)>0:
                imported_sections[sections[-1]] = data.copy()
                data = []

            sections.append(line)
            imported_sections[line] = []
            current_section = imported_sections[line]

            # If starting the second section, version and wrap must have been already defined.
            if len(sections)>1 and las_version is None:
                raise LasImportError("LAS version must be defined in the first file section.")

            if len(sections)>1 and las_wrap is None:
                raise LasImportError("LAS wrap mode must be defined in the first file section.")

            # Version must be the first section
            if len(sections)>0 and not sections[0].upper().startswith('~V'):
                raise LasImportError("The ~Version section must be the first section of a LAS file", line_number)

            # Only one ~Version section allowed
            if len([s for s in sections if s.upper().startswith('~V')]) > 1:
                raise LasImportError("LAS file can only have one ~VERSION section.", line_number)

            # Well must be the second section
            if len(sections)>1 and not sections[1].upper().startswith('~W') and las_version==3:
                raise LasImportError("The ~Well section must be the second section of a LAS 3.0 file", line_number)

            # Only one ~Well section allowed
            if len([s for s in sections if s.upper().startswith('~W')]) > 1:
                raise LasImportError("LAS file can only have one ~Well section.", line_number)

            # Only one ascii section in LAS 2.0
            if las_version == 2 and len([s for s in sections if s.upper().startswith('~A')]) > 1:
                raise LasImportError("LAS file can only have one ~ASCII section.", line_number)

            continue

        # Version information section --------------------------
        if sections[-1].upper().startswith('~V'):

            brline = break_las2_line(line_number, line, las_delimiter)
            current_section.append(brline)

            if brline['mnem'].upper() == 'VERS':
                if all(k in '023.' for k in brline['value']):
                    if abs(float(brline['value']) - 2.0) < 1e-10:
                        las_version = 2
                        break_las_line = break_las2_line
                    elif abs(float(brline['value']) - 3.0) < 1e-10:
                        las_version = 3
                        break_las_line = break_las3_line
                    else:
                        raise LasImportError("invalid LAS file version.", line_number)
                else:
                    raise LasImportError("invalid LAS file version.", line_number)

            elif brline['mnem'].upper() == 'WRAP':
                if brline['value'].upper() == 'YES':
                    las_wrap = True
                elif brline['value'].upper() == 'NO':
                    las_wrap = False
                else:
                    raise LasImportError("~Version WRAP value must be 'YES' or 'NO'.", line_number)

                if las_version == 3 and las_wrap:
                    raise LasImportError("'NO' is the only legal LAS 3.0 value for WRAP.", line_number)

            elif brline['mnem'].upper() == 'DLM':
                if brline['value'].upper() == 'SPACE':
                    las_delimiter = ' '
                elif brline['value'].upper() == 'COMMA':
                    las_delimiter = ','
                elif brline['value'].upper() == 'TAB':
                    las_delimiter = '\t'
                else:
                    raise LasImportError("~Version DLM value must be 'SPACE', 'COMMA', or 'TAB'.", line_number)

            if len(current_section)>0 and current_section[0]['mnem'] != 'VERS':
                raise LasImportError("VERS must be the first line in ~Version section.", line_number)

            if len(current_section)>1 and current_section[1]['mnem'] != 'WRAP':
                raise LasImportError("WRAP must be the second line in ~Version section.", line_number)

            if len(current_section)>2 and current_section[2]['mnem'] != 'DLM' and las_version == 3:
                raise LasImportError("DLM must be the thrird line in ~Version section.", line_number)

            continue

        sec = sections[-1].split('|')[0].split('[')[0].strip().upper()

        # Parameter or curve type section ----------------------
        condition = [
            las_version == 2 and sec.startswith('~W'), # Well information LAS 2.0
            sec.startswith('~WELL'),                   # Well information LAS 3.0
            las_version == 2 and sec.startswith('~P'), # Parameter section LAS 2.0
            sec.startswith('~PARAMETER'),              # Parameter section LAS 3.0
            sec.startswith('~LOG_PARAMETER'),          # Parameter section LAS 3.0
            las_version == 2 and sec.startswith('~C'), # Curve definition section LAS 2.0
            sec.startswith('~CURVE'),                  # Curve definition section LAS 3.0
            sec.startswith('~LOG_DEFINITION'),         # Curve definition section LAS 3.0
            sec.endswith('_PARAMETER'),                # Parameter section LAS 3.0
            sec.endswith('_DEFINITION'),               # Data definition section LAS 3.0
        ]

        if any(condition):
            brline = break_las_line(line_number, line, las_delimiter)
            current_section.append(brline)
            continue

        # Data section -----------------------------------------
        condition = [
            las_version == 2 and sec.startswith('~A'), # ASCII section LAS 2.0
            sec.startswith('~ASCII'),                  # Data values LAS 3.0
            sec.endswith('_DATA'),                     # Data values section LAS 3.0
            las_version == 2 and sec.startswith('~O'), # Legacy LAS 2.0 ~Other section
        ]
        if any(condition):
            data.append(line)
            continue

        # If the execution gets here, something bad happened...
        raise LasImportError("This line does not fit the LAS standards.", line_number)

    # Final data flush
    if len(data)>0:
        imported_sections[sections[-1]] = data.copy()

    section_order = sections
    return imported_sections, section_order, las_version, las_wrap, las_delimiter


def read_las(filename, encoding=None):
    """
    Read and parse a LAS (Log ASCII Standard) file, returning a structured LAS object.

    Parameters
    ----------
    filename : str
        Path to the LAS file.
    encoding : str or None, optional
        File encoding to use when reading the text file.
        If None (default), the function will try to read the file using
        'ascii', 'utf-8', and 'latin-1' options.

    Returns
    -------
    Las
        Parsed LAS data encapsulated in a RockVerse LAS object.
    """
    final_data = {}
    # Imported data sits only on rank 0
    with collective_only_rank0_runs():
        if mpi_rank == 0:
            lines = load_text_file(filename, encoding=encoding)
            initial_comments = _get_first_comment_lines(lines)
            imported_sections, section_order, las_version, las_wrap, las_delimiter = _split_sections(lines)
            if las_version == 2:
                final_data = assemble_las2_dict(imported_sections, las_wrap)
                final_data['_version'] = 2
            elif las_version == 3:
                final_data = assemble_las3_dict(imported_sections, section_order, las_delimiter)
                final_data['_version'] = 3
            else: # Maybe another version in the future?...
                raise NotImplementedError(f"I don't know how to read LAS version {las_version}.")
            final_data['_initial_comments'] = initial_comments

            # Change "value" to "code" and "data" to "value" in data entries
            sections = [k for k in final_data.keys() if k not in ('Well', 'Other', '_initial_comments', '_version')]
            for sec in sections:
                for k in final_data[sec]['data']:
                    k['code'] = k.pop('value')
                    k['value'] = k.pop('data')
    las_data = Las()
    las_data.dict = final_data
    return las_data


def cwls_las_sample(version=3, sample=1):
    """
    Load LAS sample from the Canadian Well Logging Society LAS standard documents.

    Parameters
    ----------
    version : {2, 3}
        The LAS version.

    sample : int
        The sample file. Must be from 1 to 5 for LAS 2.0 or 1 for LAS 3.0.

    Returns
    -------
    Las
        Parsed LAS data from sample file.
    """
    _assert.in_group('version', version, (2, 3))
    if version == 2:
        _assert.in_group('for version=2, sample', sample, (1, 2, 3, 4, 5))
    if version == 3:
        _assert.in_group('for version=3, sample', sample, (1,))
    if version == 2 and sample == 1:
        filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_1.las')
    elif version == 2 and sample == 2:
        filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_2.las')
    elif version == 2 and sample == 3:
        filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_3.las')
    elif version == 2 and sample == 4:
        filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_4.las')
    elif version == 2 and sample == 5:
        filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_5.las')
    else:
        filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS3_example_1.las')
    return read_las(filename)


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
    if mpi_rank != 0:
        return ''
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
    if mpi_rank != 0:
        return ''
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


class LasEntry(dict):

    def as_parallelarray(self, **kwargs):
        parray = array(data=self['value'],
                       name=self['mnem'],
                       unit=self['unit'],
                       description=self['description'],
                       **kwargs)
        if 'code' in self and self['code']:
            parray.attrs['code'] = self['code']
        return parray

    def as_coordinate(self, **kwargs):
        parray = coordinate(data=self['value'],
                            name=self['mnem'],
                            unit=self['unit'],
                            description=self['description'],
                            **kwargs)
        if 'code' in self and self['code']:
            parray.attrs['code'] = self['code']
        return parray


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
        self._entries = list_
        self._type = type_

    def __contains__(self, item):
        """
        Check if an entry with the given mnemonic exists in this LAS subsection.

        Parameters
        ----------
        item : str
            The mnemonic to look for in the subsection entries.

        Returns
        -------
        bool
            True if an entry with the mnemonic exists, False otherwise.

        Examples
        --------

        >>> 'DPHI' in las_subsection
        True
        """
        contains = len([k for k in self._entries if k['mnem'] == item])>0
        return mpi_comm.bcast(contains, root=0)

    def __len__(self):
        length = len(self._entries)
        return mpi_comm.bcast(length, root=0)

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
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                if isinstance(key, int):
                    value = self._entries[key]
                elif isinstance(key, str):
                    ind = [k for k, v in enumerate(self._entries) if v['mnem'] == key]
                    if not ind:
                        raise KeyError(key)
                    if len(ind) > 1:
                        raise KeyError(f'multiple entries for {key}')
                    value = self._entries[ind[0]]
        value = mpi_comm.bcast(value, root=0)
        entry = LasEntry()
        entry.update(value)
        return entry

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
        matching_entries = [k for k, v in enumerate(self._entries) if fnmatch.fnmatch(v['mnem'], pattern)]
        if matching_entries:
            if self._type == 'parameters':
                for k in matching_entries:
                    out.append(f"{prepend}|-[{k}] {_print_parameter(self._entries[k])}")
            elif self._type == 'data':
                for k in matching_entries:
                    out.append(f"{prepend}|-[{k}] {_print_data(self._entries[k])}")
            else:
                raise Exception('What is happening? Have you tampered with the Las objects?...')
        out = mpi_comm.bcast(out, root=0)
        return out

    def find_path(self, pattern, prepend=''):
        """
        Find all entries with mnemonics matching the given Unix shell-style wildcard
        pattern and return their mnemonics.

        Parameters
        ----------
        pattern : str
            Pattern to search for, using Unix shell-style wildcards (e.g., '*NMR*').

        Returns
        -------
        list of str
            List of matching mnemonics.

        Examples
        --------
        Find all mnemonics containing 'NMR' in Curve data section:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las('/path/to/las/file.las')
            paths = las_data['Curve/data'].find_path('*NMR*')
            for p in paths:
                print(p)

        Related Tutorials
        -----------------
        .. nblinkgallery::
            ../../../tutorials/data/welllog/importinglas
        """
        out = []
        matching_entries = [k for k, v in enumerate(self._entries) if fnmatch.fnmatch(v['mnem'], pattern)]
        if matching_entries:
            if self._type == 'parameters':
                for k in matching_entries:
                    out.append(f"{prepend}{self._entries[k]['mnem']}")
            elif self._type == 'data':
                for k in matching_entries:
                    out.append(f"{prepend}{self._entries[k]['mnem']}")
            else:
                raise Exception('What is happening? Have you tampered with the Las objects?...')
        out = mpi_comm.bcast(out, root=0)
        return out

    def find(self, pattern):
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
        out = self._find(pattern)
        if out:
            _lprint('')
            _lprint('\n'.join(out))
        else:
            _lprint('<no match>')

    def tree(self):
        """
        Print a tree representation of the subsection entries.

        Examples
        --------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas
        """
        self.find("*")

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

    def create_fieldgroup(self, columns=None, coordinate_column=None, chunks=None, **kwargs):
        """
        Examples
        --------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        coord = self[0] if coordinate_column is None else self[coordinate_column]
        coords = CoordinateSet(coord.as_coordinate(),)
        field_group = create_fieldgroup(coords=coords, **kwargs)
        columns_ = columns if columns is not None else [k['mnem'] for k in self._entries if k['mnem'] != coord['mnem']]
        for col in columns_:
            key = self[col]['mnem']
            array = self[col].as_parallelarray()
            field_group.create_array(key, dtype=array.dtype)
            field_group[key][...] = array[...]
            field_group[key].attrs.update(array.attrs.asdict())
        return field_group


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

    def __contains__(self, item):
        """
        Check if 'parameters' or 'data' subsection exists in this LAS section.

        Parameters
        ----------
        item : str
            The name of the subsection to check for. Expected values are 'parameters' or 'data'.

        Returns
        -------
        bool
            True if the subsection exists, False otherwise.

        Examples
        --------

        >>> 'parameters' in las_section
        True

        """
        return item in ('parameters', 'data')

    def __getitem__(self, key):
        """
        Access the parameters or data subsections by key.
        Supports nested access using forward slash ('/') notation for deeper access.

        Parameters
        ----------
        key : str
            Subsection name or nested key with '/' separator, e.g., 'parameters' or 'parameters/0'.

        Returns
        -------
        LasSubSection or dict
            The corresponding subsection object or entry dictionary.

        Examples
        --------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        if key == 'parameters':
            return self.parameters
        if key == 'data':
            return self.data

        pos = key.find('/')
        if pos > 0:
            if key[:pos] in ('parameters', 'data'):
                return self[key[:pos]][key[pos+1:]]
            collective_raise(KeyError(key[:pos]))

        collective_raise(KeyError(key))

    def find_path(self, pattern, prepend=''):
        """
        Find all entries with mnemonics matching the given Unix shell-style wildcard
        pattern and return their full paths within the LAS Section hierarchy using
        forward slash ('/') notation.

        Parameters
        ----------
        pattern : str
            Pattern to search for, using Unix shell-style wildcards (e.g., '*NMR*').

        Returns
        -------
        list of str
            List of full paths to matching entries, formatted as 'Subsection/Mnemonic'.

        Examples
        --------
        Find all mnemonics containing 'NMR' and get their full paths:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las('/path/to/las/file.las')
            paths = las_data['Curve'].find_path('*NMR*')
            for p in paths:
                print(p)

        Related Tutorials
        -----------------
        .. nblinkgallery::
            ../../../tutorials/data/welllog/importinglas
        """

        out = []
        out_parameter = self.parameters.find_path(pattern)
        out_data = self.data.find_path(pattern)
        if out_parameter:
            out += [f"{prepend}parameters/{k}" for k in out_parameter]
        if out_data:
            out += [f"{prepend}data/{k}" for k in out_data]
        out = mpi_comm.bcast(out, root=0)
        return out

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
        out = mpi_comm.bcast(out, root=0)
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
            _lprint('')
            _lprint('\n'.join(out))
        else:
            _lprint('<no match>')


    def tree(self):
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
        self.find('*')


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

    def _get_attribute(self, key):
        value = None
        if mpi_rank == 0:
            value = self.dict[key]
        return mpi_comm.bcast(value, root=0)

    @property
    def version(self):
        """
        Return the LAS version in the original LAS file.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        return self._get_attribute('_version')

    @property
    def initial_comments(self):
        """
        Return the initial comments from the LAS file.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        return self._get_attribute('_initial_comments')

    def section_keys(self):
        """
        Return a list with the LAS section keys.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        keys = None
        if mpi_rank == 0:
            keys = [k for k in self.dict.keys() if k not in ('_version', '_initial_comments')]
        keys = mpi_comm.bcast(keys, root=0)
        dict_ = {'Well': None}
        dict_.update({k: None for k in keys})
        return dict_.keys()

    def __contains__(self, item):
        """
        Check if a given section name exists in the LAS object.

        Parameters
        ----------
        item : str
            The name of the LAS section to check for.

        Returns
        -------
        bool
            True if the section exists in the LAS data, False otherwise.

        Examples
        --------
        >>> 'Curve' in las_data
        True

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        return item in self.section_keys()

    def __getitem__(self, key):
        """
        Access a LAS section by key.
        Supports accessing nested subsections using forward slash ('/') notation.


        Parameters
        ----------
        key : str
            The section name or a full path to a nested subsection using '/' as a
            separator. For example: 'Curve/parameters' or 'Well'.

        Returns
        -------
        LasSection or LasParam or LasData or dict
            The corresponding section object or section entry.

        Examples
        --------

        Access top-level section:
        >>> las_data['Curve']
        Access nested subsection:
        >>> las_data['Curve/parameters']
        Access specific entry by mnemonic:
        >>> las_data['Curve/parameters/PDAT']

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        if key == 'Well':
            if mpi_rank == 0:
                return LasParam(self.dict[key])
            return LasParam([])

        if key in self.section_keys():
            if mpi_rank == 0:
                return LasSection(self.dict[key])
            return LasSection([])

        pos = key.find('/')
        if pos > 0:
            if key[:pos] in self.section_keys():
                return self[key[:pos]][key[pos+1:]]
            collective_raise(KeyError(key[:pos]))
        collective_raise(KeyError(key))

    def find_path(self, pattern):
        """
        Find all entries with mnemonics matching the given Unix shell-style wildcard pattern
        and return their full paths within the LAS hierarchy using forward slash ('/') notation.

        Parameters
        ----------
        pattern : str
            Pattern to search for, using Unix shell-style wildcards (e.g., '*NMR*').

        Returns
        -------
        list of str
            List of full paths to matching entries, formatted as 'Section/Subsection/Mnemonic'.

        Examples
        --------
        Find all mnemonics containing 'NMR' and get their full paths:

        .. code-block:: python

            import rockverse as rv
            las_data = rv.read_las('/path/to/las/file.las')
            paths = las_data.find_path('*NMR*')
            for p in paths:
                print(p)

        Related Tutorials
        -----------------
        .. nblinkgallery::
            ../../../tutorials/data/welllog/importinglas
        """

        main_out = []
        aux_out = self['Well'].find_path(pattern, prepend="Well/")
        if aux_out:
            main_out += aux_out
        for sec in self.section_keys():
            if sec not in ('Well', 'Other'):
                aux_out = self[sec].find_path(pattern, prepend=f"{sec}/")
                if aux_out:
                    main_out += aux_out
        main_out = mpi_comm.bcast(main_out, root=0)
        return main_out

    def _find(self, pattern):
        main_out = []
        aux_out = self['Well']._find(pattern, prepend="|   ")
        if aux_out:
            main_out.append('|- Well')
            main_out += aux_out
        for sec in self.section_keys():
            if sec not in ('Well', 'Other'):
                aux_out = self[sec]._find(pattern, prepend="|   ")
                if aux_out:
                    main_out.append(f"|- {sec}")
                    main_out += aux_out
        main_out = mpi_comm.bcast(main_out, root=0)
        return main_out


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
        out = self._find(pattern)
        if out:
            _lprint('\n')
            _lprint("\n".join(out))
        else:
            _lprint('<no match>')

    def tree(self):
        """
        Print a tree representation of the LAS file, including sections and comments.

        Related Tutorials
        -----------------

        .. nblinkgallery::

            ../../../tutorials/data/welllog/importinglas

        """
        if '_initial_comments' in self.dict and self.dict['_initial_comments']:
            _lprint(self.dict['_initial_comments'])
        self.find('*')

        if 'Other' in self.section_keys():
            _lprint('|- Other:')
            _lprint(self.dict['Other'])

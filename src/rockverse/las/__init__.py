"""
Module for reading and parsing LAS (Log ASCII Standard) files.

This module provides functionality to read, parse, and structure LAS files
commonly used in well logging data. It supports LAS versions 2.0 and 3.0,
integrating with RockVerse's parallel data structures.
"""

import os
from rockverse import __path__ as RVPATH
from rockverse._utils.text import load_text_file
from rockverse.las.exceptions import LasImportError
from rockverse.las.las2 import break_las2_line, assemble_las2_dict
from rockverse.las.las3 import break_las3_line, assemble_las3_dict
from rockverse.las.las import Las, LasSection, LasParam, LasData


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
    lines = load_text_file(filename, encoding=encoding)
    initial_comments = _get_first_comment_lines(lines)
    imported_sections, section_order, las_version, las_wrap, las_delimiter = _split_sections(lines)
    if las_version == 2:
        final_data = assemble_las2_dict(imported_sections, las_wrap)
        final_data.dict['_version'] = 2
    elif las_version == 3:
        final_data = assemble_las3_dict(imported_sections, section_order, las_delimiter)
        final_data.dict['_version'] = 3
    else: # Maybe another version in the future?...
        raise NotImplementedError(f"I don't know how to read LAS version {las_version}.")
    final_data.dict['_initial_comments'] = initial_comments

    # Change "value" to "code" and "data" to "value" in data entries
    sections = [k for k in final_data.dict.keys() if k not in ('Well', 'Other', '_initial_comments', '_version')]
    for sec in sections:
        for k in final_data.dict[sec]['data']:
            k['code'] = k.pop('value')
            k['value'] = k.pop('data')

    return final_data


def las_sample1():
    """
    Load sample LAS2 example 1 file.

    Returns
    -------
    Las
        Parsed LAS data from sample file.
    """
    filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_1.las')
    return read_las(filename)

def las_sample2():
    """
    Load sample LAS2 example 2 file.

    Returns
    -------
    Las
        Parsed LAS data from sample file.
    """
    filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_2.las')
    return read_las(filename)

def las_sample3():
    """
    Load sample LAS2 example 3 file.

    Returns
    -------
    Las
        Parsed LAS data from sample file.
    """
    filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_3.las')
    return read_las(filename)

def las_sample4():
    """
    Load sample LAS2 example 4 file.

    Returns
    -------
    Las
        Parsed LAS data from sample file.
    """
    filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_4.las')
    return read_las(filename)

def las_sample5():
    """
    Load sample LAS2 example 5 file.

    Returns
    -------
    Las
        Parsed LAS data from sample file.
    """
    filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS2_example_5.las')
    return read_las(filename)

def las_sample6():
    """
    Load sample LAS3 example 1 file.

    Returns
    -------
    Las
        Parsed LAS data from sample file.
    """
    filename = os.path.join(RVPATH[0], 'sample_data', 'las', 'LAS3_example_1.las')
    return read_las(filename)
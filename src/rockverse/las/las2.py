import numpy as np
from rockverse.las.exceptions import LasImportError
from rockverse.las.las import Las

def break_las2_line(line_number, line, las_delimiter):
    """
    Parse a single line from a LAS 2.0 section into its components.

    Parameters
    ----------
    line_number : int
        The line number in the LAS file (for error reporting).
    line : str
        The line content to parse.
    las_delimiter : str
        Delimiter used in the LAS file (for compatibility, not used here).

    Returns
    -------
    dict
        Dictionary with keys 'mnem', 'unit', 'value', and 'description' parsed from the line.

    Raises
    ------
    LasImportError
        If the line does not conform to expected LAS 2.0 format.
    """

    # las_delimiter must be here for compatibility with break_las3_line

    out = {'mnem': '', 'unit': '', 'value': '', 'description': ''}

    aux = line.strip()

    #Mnemonic up to, but not including, the first period
    pos = [i for i, v in enumerate(aux) if v == '.']
    if pos:
        pos = min(pos)
        out['mnem'] = aux[:pos].strip()
        aux = aux[pos+1:] if len(aux) > pos+1 else ''
        # Mnemonic must not contain any internal spaces, dots, or colons:
        if any([k in out['mnem'] for k in (' ', '.', ',')]):
            raise LasImportError("Mnemonic must not contain spaces, dots, or colons.", line_number)
    else:
        raise LasImportError("Missing '.' delimiter.'", line_number)

    #Unit up to, but not including, the first space
    pos = [i for i, v in enumerate(aux) if v == ' ']
    if pos:
        pos = min(pos)
        out['unit'] = aux[:pos].strip()
        aux = aux[pos+1:] if len(aux) > pos+1 else ''
        if any([k in out['unit'] for k in (',', ' ')]):
            raise LasImportError("Unit must not contain colons or internal spaces.", line_number)
    else:
        out['unit'] = aux.strip()
        aux = ''

    #Value up to, but not including, the last colon
    pos = [i for i, v in enumerate(aux) if v == ':']
    if pos:
        pos = max(pos)
        out['value'] = aux[:pos].strip()
        aux = aux[pos+1:] if len(aux) > pos+1 else ''
    else:
        raise LasImportError("Missing ':' delimiter.'", line_number)

    #Description: everything remaining
    out['description'] = aux.strip()
    aux = ''

    return out

def convert_value_from_las2(value, null=None):
    """
    Convert a LAS 2.0 string value to a numeric or string type, handling null values.

    Parameters
    ----------
    value : str
        The raw value string from the LAS file.
    null : int or float, optional
        The null value indicator to map to NaN.

    Returns
    -------
    int, float, str, or None
        The converted numeric value, NaN for nulls, or trimmed string if conversion fails.

    Raises
    ------
    ValueError
        If input types are invalid.
    """

    if not isinstance(value, str):
        raise ValueError('value must be string.')

    if null is not None and not isinstance(null, (int, float)):
        raise ValueError('null must be a number.')

    if not value:
        return None

    if all(k in '0123456789.-' for k in value): # Number
        if '.' in value:
            number = float(value)
        else:
            number = int(value)
        if null is not None and abs(number - null) < 1e-10:
            return np.nan
        return number

    # Everything failed, just trim
    return value.strip()


def assemble_las2_dict(imported_sections, las_wrap):
    """
    Assemble LAS 2.0 parsed sections into a structured Las object.

    Parameters
    ----------
    imported_sections : dict
        Sections parsed from the LAS file.
    las_wrap : bool
        Indicates whether ASCII data is wrapped.

    Returns
    -------
    Las
        Structured LAS data with sections mapped to parameters and data.

    Raises
    ------
    LasImportError
        If required sections are missing or malformed.
    """

    # Well section
    key = [k for k in imported_sections if k.upper().startswith('~W')]
    if len(key) == 0:
        raise LasImportError("I didn't find the LAS ~Well section.")
    if len(key) > 1:
        raise LasImportError("Only one LAS ~Well section is allowed.")
    key = key[0]
    well_section = imported_sections[key]
    for parameter in well_section:
        parameter['value'] = convert_value_from_las2(parameter['value'])
    null_value = [v['value'] for v in well_section if v['mnem'] == 'NULL']
    if not null_value:
        raise LasImportError("I didn't find NULL parameter in ~Well section.")
    null_value = null_value[0]

    # Parameter section
    parameter_section = None
    key = [k for k in imported_sections if k.upper().startswith('~P')]
    if len(key) > 1:
        raise LasImportError("Only one LAS ~Parameter section is allowed.")
    if key:
        key = key[0]
        parameter_section = imported_sections[key]
        for parameter in parameter_section:
            parameter['value'] = convert_value_from_las2(parameter['value'])

    # Other section
    other_section = None
    key = [k for k in imported_sections if k.upper().startswith('~O')]
    if len(key) > 1:
        raise LasImportError("Only one LAS ~Other section is allowed.")
    if key:
        key = key[0]
        other_section = '\n'.join(imported_sections[key])

    # Curve section
    key = [k for k in imported_sections if k.upper().startswith('~C')]
    if len(key) == 0:
        raise LasImportError("I didn't find the LAS ~Curve section.")
    if len(key) > 1:
        raise LasImportError("Only one LAS ~Curve section is allowed.")
    key = key[0]
    curve_section = imported_sections[key]
    for parameter in curve_section:
        parameter['value'] = convert_value_from_las2(parameter['value'])

    # ASCII section
    key = [k for k in imported_sections if k.upper().startswith('~A')]
    if len(key) == 0:
        raise LasImportError("I didn't find the LAS ~Ascii section.")
    if len(key) > 1:
        raise LasImportError("Only one LAS ~Ascii section is allowed.")
    key = key[0]
    data = imported_sections[key]

    # LAS 2.0 enforces space delimiter and numeric data
    conv_data = []
    if not las_wrap:
        for dataline in data:
            conv_data.append([convert_value_from_las2(k, null_value) for k in dataline.strip().split(' ') if k])
    else:
        aux_data = []
        for dataline in data:
            aux_data += [convert_value_from_las2(k, null_value) for k in dataline.strip().split(' ') if k]
            if len(aux_data) == len(curve_section):
                conv_data.append(aux_data)
                aux_data = []
            if len(aux_data) > len(curve_section):
                raise LasImportError("LAS ~Ascii wrapped data doesn't match ~Curve items.")

    if len(curve_section) != len(conv_data[0]):
        raise LasImportError("LAS ~Ascii columns must match the number of curves in ~Curve section.")

    for k, curve in enumerate(curve_section):
        curve['data'] = np.array([v[k] for v in conv_data])

    final_data = Las()
    final_data.dict['Well'] = [k for k in well_section if k['mnem'] not in ('NULL', 'STRT', 'STOP', 'STEP')]
    final_data.dict['Curve'] = {}
    if parameter_section is not None:
        final_data.dict['Curve']['parameters'] = parameter_section
    final_data.dict['Curve']['data'] = curve_section
    if other_section is not None:
        final_data.dict['Other'] = other_section


    return final_data

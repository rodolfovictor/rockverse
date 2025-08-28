import datetime
import numpy as np
from rockverse.las.exceptions import LasImportError
from rockverse.las.las2 import convert_value_from_las2
from rockverse.las.las import LAS

def break_las3_line(line_number, line, las_delimiter):

    out = {'mnem': '', 'unit': '', 'value': '', 'description': '',
           'format': '', 'association': []}

    aux = line.strip()

    #Mnemonic up to, but not including, the first period
    pos = [i for i, v in enumerate(aux) if v == '.']
    if pos:
        pos = min(pos)
        out['mnem'] = aux[:pos].strip()
        aux = aux[pos+1:] if len(aux) > pos+1 else ''
        if any([k in out['mnem'] for k in ('.', ',', ' ', '\t', '{', '}', '|')]):
            raise LasImportError(("Mnemonic must not contain periods, colons, embedded spaces, "
                                  "tabs, { }, or | (bar) characters."), line_number)
    else:
        raise LasImportError("Missing '.' delimiter.'", line_number)

    #LAS 3.0 unit up to, but not including, the first space or first colon
    pos = [i for i, v in enumerate(aux) if v in (' ', ',')]
    if pos:
        pos = min(pos)
        out['unit'] = aux[:pos].strip()
        aux = aux[pos+1:] if len(aux) > pos+1 else ''
        if any([k in out['unit'] for k in (',', ' ', '\t', '{', '}', '|')]):
            raise LasImportError(("Unit must not contain colons, embedded spaces, "
                                  "tabs, { } or | (bar) characters."), line_number)
    else:
        out['unit'] = aux.strip()
        aux = ''

    #Value up to, but not including, the last colon
    pos = [i for i, v in enumerate(aux) if v == ':']
    if pos:
        pos = max(pos)
        out['value'] = aux[:pos].strip()
        aux = aux[pos+1:] if len(aux) > pos+1 else ''
        if any([k in out['value'] for k in ('{', '}', '|')]):
            raise LasImportError(("Value must not contain { } or | (bar) characters."), line_number)
    else:
        out['value'] = aux.strip()
        aux = ''

    #LAS 3.0 description up to the last left brace or the last bar, whichever appears first
    pos = []
    pos1 = [i for i, v in enumerate(aux) if v == r'{']
    if pos1:
        pos.append(max(pos1))
    pos1 = [i for i, v in enumerate(aux) if v == r'|']
    if pos1:
        pos.append(max(pos1))
    if pos:
        pos = min(pos)
        out['description'] = aux[:pos].strip()
        if aux[pos] == '{':
            aux = aux[pos:] #preserves brace
        elif len(aux) > pos+1:
            aux = aux[pos+1:]
        else:
            aux = ''
    else:
        out['description'] = aux.strip()
        aux = ''

    # Association after the last bar |
    pos = [i for i, v in enumerate(aux) if v == '|']
    if pos:
        pos = max(pos)
        out['association'] = tuple(aux[pos+1:].strip().split(las_delimiter))
        aux = aux[:pos]

    # Format: last set of matching {}
    close_brackets = [i for i, val in enumerate(aux) if val == r'}']
    if close_brackets:
        close_pos = close_brackets[-1]
        open_pos = [i for i, val in enumerate(aux) if val == r'{' and i<close_pos]
        if open_pos:
            open_pos = open_pos[-1]
            out['format'] = aux[open_pos:close_pos+1].strip()[1:-1]

    return out




def convert_value_from_las3(value, format=None, null=None, delimiter=None):

    def trouble(value, format):
        raise ValueError(f"Error converting '{value}' to {{{format}}} format.")

    # No format: fall back to LAS 2.0 strategy
    if format in (None, ''):
        return convert_value_from_las2(value=value, null=null)

    if not isinstance(format, str):
        raise ValueError('format must be string.')

    if not isinstance(value, str):
        raise ValueError('value must be string.')

    if null is not None and not isinstance(null, (int, float)):
        raise ValueError('null must be a number.')

    if not value:
        return None

    # Array
    if format[0].upper() in 'A':
        try:
            return convert_value_from_las3(value, format=format[1:], null=null, delimiter=delimiter)
        except Exception:
            trouble()

    # Floating point or exponential
    if format[0].upper() in 'FE':
        number = [float(k) for k in value.split(delimiter)]
        if null is not None:
            number = [np.nan if abs(k-null)<1e-10 else k for k in number]
        if len(number) == 1:
            return number[0]
        return tuple(number)

    # Integer
    elif format[0].upper() == 'I':
        number = [int(k) for k in value.split(delimiter)]
        if len(number) == 1:
            return number[0]
        return tuple(number)

    # String
    elif format[0].upper() == 'S':
        return value.strip()

    # Date and Time
    elif format[0] in 'DMYhms':

        split_value = [k for k in value.split(' ') if k]
        split_format = [k for k in format.split(' ') if k]
        for s in '/-:':
            aux_value = []
            aux_format = []
            for m in range(len(split_value)):
                aux_value += split_value[m].split(s)
                aux_format += split_format[m].split(s)
            split_value = aux_value.copy()
            split_format = aux_format.copy()
            if len(split_value) != len(split_format):
                trouble(value, format)

        year, month, day = None, None, None
        hour, minute, sec = None, None, None
        for v, f in zip(split_value, split_format):
            if all(s == 'D' for s in f):
                day = int(v)
            elif all(s == 'M' for s in f):
                month = int(v)
            elif all(s == 'Y' for s in f):
                year = int(v)
            elif all(s == 'h' for s in f):
                hour = int(v)
            elif all(s == 'm' for s in f):
                minute = int(v)
            elif all(s == 's' for s in f):
                sec = int(v)
            else:
                trouble(value, format)

        if any(v is None for v in (year, month, day)):
            trouble(value, format)

        date = datetime.date(year=year, month=month, day=day)

        kwargs = {}
        if hour is not None:
            kwargs['hour'] = hour
        if minute is not None:
            kwargs['minute'] = minute
        if sec is not None:
            kwargs['second'] = sec
        if kwargs:
            date = datetime.datetime.combine(date, datetime.time(**kwargs))

        return date

    elif format == '''°'"''':
        try:
            deg = float(value.split('°')[0].strip())
            minute = float(value.split('°')[1].split('\'')[0].strip())
            second = float(value.split('°')[1].split('\'')[1].split('"')[0].strip())
            return (deg, minute, second)
        except Exception:
            trouble(value, format)

    else:
        raise ValueError(f'Invalid format {{{format}}}.')



def assemble_las3_section_trio(section_keys, imported_sections, las_delimiter):
        """
        section_keys = (parameter mnen, definition mnem, data mnem)
        """

        final_group = {}
        sections = [None, None, None]
        for n, section_key in enumerate(section_keys):
            dict_key = [k for k in imported_sections if k.upper() == section_key.upper()]
            if len(dict_key) > 1:
                raise LasImportError(f"Duplicated {section_key} section.")
            if len(dict_key) > 0:
                sections[n] = imported_sections[dict_key[0]]

        if sections[0] is not None:
            for p in sections[0]:
                p['value'] = convert_value_from_las3(value=p['value'], format=p['format'], delimiter=las_delimiter)
            final_group['parameters'] = sections[0]

        if sections[1] is not None and sections[2] is None:
            raise LasImportError(f"Found {section_keys[1]} section but missing {section_keys[2]} section.")

        if sections[2] is not None and sections[1] is None:
            raise LasImportError(f"Found {section_keys[2]} section but missing {section_keys[1]} section.")

        # Split and convert values, must preserve elements with delimiter but enclosed in " "
        formats = [k['format'] for k in sections[1]]
        data = []
        for line in sections[2]:
            result = []
            current = ''
            in_quotes = False
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes  # Toggle the quote state
                elif char == las_delimiter and not in_quotes:
                    result.append(current)
                    current = ''
                else:
                    current += char
            result.append(current)  # Add the last part
            data.append([convert_value_from_las3(value=v, format=f, delimiter=las_delimiter)
                         for v, f in zip(result, formats)])

        # Add to definitions
        for k, item in enumerate(sections[1]):
            item['data'] = np.array([line[k] for line in data])

        final_group['data'] = sections[1]

        return final_group

def assemble_las3_dict(imported_sections, section_order, las_delimiter):

    final_data = LAS()

    # Well section
    key = [k for k in imported_sections if k.upper().startswith('~WELL')]
    if len(key) == 0:
        raise LasImportError("I didn't find the LAS ~Well section.")
    if len(key) > 1:
        raise LasImportError("Only one LAS ~Well section is allowed.")
    key = key[0]
    well_section = imported_sections[key]
    for parameter in well_section:
        parameter['value'] = convert_value_from_las3(value=parameter['value'], format=parameter['format'])
    null_value = [v['value'] for v in well_section if v['mnem'] == 'NULL']
    if not null_value:
        raise LasImportError("I didn't find NULL parameter in ~Well section.")
    null_value = null_value[0]
    final_data['Well'] = well_section

    # Legacy ~Parameter, ~Curve, ~Ascii
    final_data['Curve'] = assemble_las3_section_trio(section_keys=('~Parameter', '~Curve', '~Ascii'),
                                                     imported_sections=imported_sections,
                                                     las_delimiter=las_delimiter)

    # Remaining sections
    section_names = [name for name in section_order if "_DATA" in name.upper()]
    for section_name in section_names:
        section_keys = ['None', 'None', section_name]
        definition = section_name.split('|')
        if len(definition) != 2:
            raise LasImportError(f"{section_name} has no associated definition section.")
        definition = '~'+definition[1].strip()
        pos = [k for k, name in enumerate(section_order) if name.upper().strip() == definition.upper().strip()]
        if len(pos) == 0:
            raise LasImportError(f"Missing {definition} section associated to {section_name} section.")
        if len(pos) == 0:
            raise LasImportError(f"Duplicate {definition} section.")
        pos = pos[0]
        section_keys[1] = section_order[pos]
        if section_order[pos-1].upper() == definition.upper().replace('_DEFINITION', '_PARAMETER'):
            section_keys[0] = section_order[pos-1]

        final_data[section_name.split('|')[0].strip().replace('~', '')] = assemble_las3_section_trio(section_keys, imported_sections, las_delimiter)

    return final_data

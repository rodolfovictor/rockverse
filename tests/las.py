

#filename='/togp/GOB7/Pseudowell/Buzios2023/pocos/Basicos/CurvasFinais_BUZ-3.las'
#filename='/u/gob7/rockverse/tests/LAS2_example_1.las'
#filename='/u/gob7/rockverse/tests/LAS2_example_2.las'
#filename='/u/gob7/rockverse/tests/LAS2_example_3.las'
#filename='/u/gob7/rockverse/tests/LAS2_example_4.las'
#filename='/u/gob7/rockverse/tests/LAS2_example_5.las'
#filename='/u/gob7/rockverse/tests/LAS3_example_1.las'
filename=r'C:\Users\GOB7\Downloads\rockverse\tests\LAS3_example_1.las'
filename=r'C:\Users\GOB7\Downloads\rockverse\tests\LAS2_example_5.las'
encoding=None

#def import_las(filename, encoding=None):
if True:
    lines = load_text_file(filename, encoding=encoding)
    imported_sections, section_order, las_version, las_wrap, las_delimiter = split_sections(lines)
    if las_version == 2:
        final_data = assemble_las2_dict(imported_sections, las_wrap)
    elif las_version == 3:
        final_data = assemble_las3_dict(imported_sections, section_order, las_delimiter)
    else: # Maybe another version in the future?...
        raise NotImplementedError(f"I don't know how to read LAS version {las_version}.")

    # Change "value" to "code" and "data" to "value" in data entries
    sections = [k for k in final_data.keys() if k not in ('Well', 'Other')]
    for sec in sections:
        for k in final_data[sec]['data']:
            k['code'] = k.pop('value')
            k['value'] = k.pop('data')

    self=final_data
    final_data.tree()

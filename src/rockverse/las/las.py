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

class LasSectionEntries():

    def __init__(self, list_, type_):
        self.entries = list_
        self.type = type_

    def tree(self, prepend=''):
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
        out = self._find(pattern, prepend)
        if out:
            print('\n'.join(out))
        else:
            print('<no match>')

class LasParameterSection(LasSectionEntries):

    def __init__(self, list_):
        super().__init__(list_, 'parameters')

class LasColumnSection(LasSectionEntries):

    def __init__(self, list_):
        super().__init__(list_, 'data')

    def create_scalarfield(self, column, coordinate_column=None, **kwargs):
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

    def __init__(self, dict_):
        self._parameters = dict_['parameters'] if 'parameters' in dict_ else dict()
        self._data = dict_['data'] if 'data' in dict_ else dict()

    @property
    def parameters(self):
        return LasParameterSection(self._parameters)

    @property
    def data(self):
        return LasColumnSection(self._data)

    def tree(self, prepend=''):
        print(f"{prepend}|- parameters")
        self.parameters.tree(prepend=f'{prepend}|   ')
        print(f"{prepend}|- data")
        self.data.tree(prepend=f'{prepend}|   ')

    def find(self, pattern, prepend=''):
        out_parameter = self.parameters._find(pattern)
        out_data = self.data._find(pattern)
        if out_parameter:
            print(f"|- parameters:")
            self.parameters.find(pattern, prepend=f'{prepend}|   ')
        if out_data:
            print(f"|- data:")
            self.data.find(pattern, prepend=f'{prepend}|   ')


class Las():

    def __init__(self):
        self.dict = {}

    @property
    def version(self):
        return self.dict['_version']

    @property
    def initial_comments(self):
        return self.dict['_initial_comments']

    def section_keys(self):
        dict_ = {'Well': self.dict['Well']}
        dict_.update({k: v for k, v in self.dict.items() if k not in ('_version', '_initial_comments')})
        return dict_.keys()

    def __getitem__(self, key):
        if key == 'Well':
            return LasParameterSection(self.dict[key])
        if key not in self.dict.keys():
            collective_raise(KeyError(key))
        return LasSection(self.dict[key])

    def tree(self):

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

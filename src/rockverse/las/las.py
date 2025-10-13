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


class LAS(dict):

    @property
    def well(self):
        return self['Well']

    @property
    def version(self):
        return self['_version']

    @property
    def initial_comments(self):
        return self['_Initial_Comments']

    @property
    def sections(self):
        return self.keys()

    def tree(self):
        if self['_Initial_Comments']:
            print(self['_Initial_Comments'])

        print('|- Well')
        for k, param in enumerate(self['Well']):
            print(f"|   |-[{k}] {_print_parameter(param)}")

        sections = [k for k in self.keys() if k not in ('Well', 'Other', '_Initial_Comments', '_version')]
        for sec in sections:
            print(f"|- {sec}")
            if 'parameters' in self[sec]:
                print('|   |- parameters:')
                for k, param in enumerate(self[sec]['parameters']):
                    print(f"|   |   |-[{k}] {_print_parameter(param)}")
            print('|   |- data:')
            for k, data in enumerate(self[sec]['data']):
                print(f"|   |   |-[{k}] {_print_data(data)}")

        if 'Other' in self.keys():
            print('|- Other:')
            print(self['Other'])

    def _get_group_by_path(self, group, path):
        subgroup = group
        split_path = path.split('/')
        if len(split_path) > 1:
            subgroup = group[split_path[0]]
            return self._get_group_by_path(subgroup, '/'.join(split_path[1:]))
        return subgroup[int(split_path[0])]


    def as_scalar_field(self, las_path, **kwargs):
        las_coord = mpi_comm.bcast(self._get_group_by_path(self, '/'.join(las_path.split('/')[:-1]+['0',])), root=0)
        las_array = mpi_comm.bcast(self._get_group_by_path(self, las_path), root=0)
        top_path = '' if 'path' not in kwargs else kwargs['path']
        kwargs['path'] = f"{top_path}/coords/0" if top_path else "coords/0"
        coord = coordinate(data=las_coord['value'],
                           name=las_coord['mnem'],
                           unit=las_coord['unit'],
                           description=las_coord['description'],
                           **kwargs)
        kwargs['path'] = f"{top_path}/array" if top_path else "array"
        parray = array(data=las_array['value'],
                       name=las_array['mnem'],
                       unit=las_array['unit'],
                       description=las_array['description'],
                       **kwargs)
        print('code' in las_array)
        parray.attrs['code'] = las_array['code'] if 'code' in las_array else ''
        return scalarfield(parray, coords=(coord,))


if __name__ == "__main__":
    import rockverse as rv
    final_data = rv.las_sample6()
    #final_data.tree()
    self=final_data

    #AQUI
    key = 'Inclinometry_Data/data/1'
    a=self.as_scalar_field('Curve/data/2')
    #group = self

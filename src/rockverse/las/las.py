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
    def sections(self):
        return self.keys()

    def tree(self):
        print('|- Well')
        for k, param in enumerate(self['Well']):
            print(f"|   |-[{k}] {_print_parameter(param)}")

        sections = [k for k in self.keys() if k not in ('Well', 'Other')]
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

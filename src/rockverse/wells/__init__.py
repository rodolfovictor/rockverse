from rockverse import _assert

class WellLog():

    def __init__(self, zgroup):
        _assert.zarr_group('zgroup', zgroup)
        self.zgroup = zgroup

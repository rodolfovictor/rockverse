import numpy as np

class RcParams(dict):

    def __init__(self):
        super().__init__()

        self['latex.strings'] = {
            'um': r'$\mu$m',
            }

        self['voxel.connectivity'] = 6

rcparams = RcParams()
import numpy as np
import copy

from rockverse.rc.orthogonal_viewer import ORTHOGONAL_VIEWER


class RcParams(dict):

    def __init__(self):
        super().__init__()

        self['latex.strings'] = {
            'um': r'$\mu$m',
            }

        self['voxel.connectivity'] = 6

        self['orthogonal_viewer'] = copy.deepcopy(ORTHOGONAL_VIEWER)

rcparams = RcParams()
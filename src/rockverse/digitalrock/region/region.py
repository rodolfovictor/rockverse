import rockverse._assert as _assert
from numba import njit, cuda

#TODO: extensive check on inside/outside implementations

MASK_CHUNK_CPU_BLOCK = '''
    nx, ny, nz = mask.shape
    for i in range(nx):
        x = float(ox) + float(i+box)*float(hx)
        for j in range(ny):
            y = float(oy) + float(j+boy)*float(hy)
            for k in range(nz):
                z = float(oz) + float(k+boz)*float(hz)
                if not _is_inside(x, y, z):
                    mask[i, j, k] = True
'''


MASK_CHUNK_GPU_BLOCK = '''
    nx, ny, nz = mask.shape
    i, j, k = cuda.grid(3)
    if i<0 or i>=nx or j<0 or j>=ny or k<0 or k>=nz:
        return
    x = float(ox) + float(i+box)*float(hx)
    y = float(oy) + float(j+boy)*float(hy)
    z = float(oz) + float(k+boz)*float(hz)
    if not _is_inside(x, y, z):
        mask[i, j, k] = True
'''

class Region():
    '''
    This is the base class for defining abstract regions of interest
    in voxel images. It should not be directly instantiated.
    '''

    def __repr__(self):
        return "Region()"

    def __str__(self):
        return ("Region")

    def __init__(self):
        exec(self._contains_point_source_code(), globals(), locals())
        self._contains_point = njit(locals().get('_is_inside'))
        self.contains_point_cuda = cuda.jit(locals().get('_is_inside'))
        exec(self._mask_chunk_source_code(), globals(), locals())
        self._mask_chunk = njit(locals().get('_mask_chunk_cpu'))
        self._mask_chunk_gpu = cuda.jit(locals().get('_mask_chunk_gpu'))

    def _contains_point_source_code(self):
        string = '''def _is_inside(x, y, z):
        return False
        '''
        return string

    def contains_point(self, x, y, z):
        '''
        Check if a point (x, y, z) belongs to the region.

        Parameters
        ----------
        x : float
            Point x-coordinate in voxel units.
        y : float
            Point y-coordinate in voxel units.
        z : float
            Point z-coordinate in voxel units.

        Returns
        -------
        b : bool
            True if point inside region, False otherwise.
        '''
        return self._contains_point(x, y, z)


    def _mask_chunk_source_code(self):
        string = 'def _mask_chunk_cpu(mask, ox, oy, oz, hx, hy, hz, box, boy, boz):\n\n    '
        string = string + self._contains_point_source_code()
        string = string + MASK_CHUNK_CPU_BLOCK + '\n\n'
        string = string + 'def _mask_chunk_gpu(mask, ox, oy, oz, hx, hy, hz, box, boy, boz):\n\n    '
        string = string + self._contains_point_source_code()
        string = string + MASK_CHUNK_GPU_BLOCK
        return string

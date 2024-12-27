import rockverse._assert as _assert
from rockverse.region.region import Region

class Sphere(Region):

    """
    Defines spherical region of interest.

    .. note::
        For performance, region methods are dynamically created at class
        instantiation, and therefore the region parameters cannot be changed
        after the object is created.

    Parameters
    ----------
    p : 3-element tuple, list, or Numpy array
        Point defining the spatial coordinates (in voxel units) for
        center of the sphere.
    r : int or float
        Sphere radius in voxel units. Must be non-negative.
    region : {'inside', 'outside'}, optional
        Whether to consider the region inside or outside the sphere
        surface. Defaults to 'inside'.
    """

    def __init__(self, p, r, region='inside'):

        _assert.iterable.ordered_numbers('p', p)
        _assert.iterable.length('p', p, 3)
        self._p = tuple(p)

        _assert.condition.non_negative_integer_or_float('r', r)
        self._r = r

        _assert.in_group('region', region, ('inside', 'outside'))
        self._region = region.lower()

        super().__init__()

    #Because super().__init__() generates compiled optimized versions, parameters cannot be modified
    @property
    def p(self):
        ''' Point defining the center of the sphere.'''
        return self._p

    @property
    def r(self):
        '''Sphere radius.'''
        return self._r

    @property
    def region(self):
        '''Whether to consider the region inside or outside the cylinder surface.'''
        return self._region

    def __repr__(self):
        return f"Sphere(p={self._p}, r={self._r}, region='{self._region}')"

    def __str__(self):
        return ("Sphere:\n    "
                f"p: {self.p} (center location)\n    "
                f"r: {self.r} (radius)\n    "
                f"region: {self.region}")

    def _contains_point_source_code(self):
        string = f'''def _is_inside(x, y, z):
        d2 = (x-{float(self._p[0])})**2 + (y-{float(self._p[1])})**2 + (z-{float(self._p[2])})**2
        if d2 > {(float(self._r)**2)}:
            return {True if self._region=='outside' else False}
        return {False if self._region=='outside' else True}
        '''
        return string

# %%

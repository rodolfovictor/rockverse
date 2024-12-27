from rockverse.region.region import Region
import rockverse._assert as _assert

class Cylinder(Region):

    """
    Defines cylindrical region of interest.

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
        Cylinder radius in voxel units. Must be non-negative.
    v : 3-element tuple, list, Numpy array
        Vector components for the axis direction vector.
    l : int, float
        Cylinder length (must be non negative).
    region : {'inside', 'outside'}, optional
        Whether to consider the region inside or outside the cylinder
        surface. Defaults to 'inside'.
    """

    def __init__(self, p, v, r, l=float('inf'), region='inside'):

        _assert.iterable.ordered_numbers('p', p)
        _assert.iterable.length('p', p, 3)
        self._p = tuple(p)

        _assert.iterable.ordered_numbers('v', v)
        _assert.iterable.length('v', v, 3)
        if (v[0]**2 + v[1]**2 + v[2]**2) == 0.0:
            _assert.collective_raise(ValueError(f'Invalid direction vector v={v}'))

        self._v = tuple(v)

        _assert.condition.non_negative_integer_or_float('r', r)
        self._r = r

        _assert.condition.non_negative_integer_or_float('l', l)
        self._l = l

        _assert.in_group('region', region, ('inside', 'outside'))
        self._region = region.lower()

        super().__init__()

    #Because super().__init__() generates compiled optimized versions, parameters cannot be modified
    @property
    def p(self):
        ''' Point defining the center of the cylinder.'''
        return self._p

    @property
    def v(self):
        '''Vector components for the axis direction vector.'''
        return self._v

    @property
    def r(self):
        '''Cylinder radius.'''
        return self._r

    @property
    def l(self):
        '''Cylinder length.'''
        return self._l

    @property
    def region(self):
        '''Whether to consider the region inside or outside the cylinder surface.'''
        return self._region

    def __repr__(self):
        return f"Cylinder(p={self._p}, v={self._v}, r={self._r}, l={self._l}, region='{self._region}')"

    def __str__(self):
        return ("Cylinder:\n    "
                f"p: {self._p} (center location)\n    "
                f"v: {self._v} (axis direction vector)\n    "
                f"r: {self._r} (radius)\n    "
                f"l: {self._l} (length)\n    "
                f"region: {self._region}")

    def _contains_point_source_code(self):
        if (self.v[0]**2 + self.v[1]**2 + self.v[2]**2) == 0.0:
            _assert.collective_raise(ValueError(f'Cylinder cannot use v={self.v}'))

        string = f'''def _is_inside(x, y, z):
        #vector from ref_point to calc_point
        rx = x-{float(self._p[0])}
        ry = y-{float(self._p[1])}
        rz = z-{float(self._p[2])}
        #Projection r onto v
        r_dot_v = rx*{float(self._v[0])} + ry*{float(self._v[1])} + rz*{float(self._v[2])}
        v_dot_v = {float(self._v[0])**2 + float(self._v[1])**2 + float(self._v[2])**2}
        projx = r_dot_v/v_dot_v * {float(self._v[0])}
        projy = r_dot_v/v_dot_v * {float(self._v[1])}
        projz = r_dot_v/v_dot_v * {float(self._v[2])}
        '''
        if float(self._l) != float("inf"):
            string = string + f'''
        #distance along axis larger than l/2? Outside.
        if (projx**2 + projy**2 + projz**2) > {(float(self._l)/2)**2}:
            return {True if self._region == 'outside' else False}
         '''
        string = string + f'''
        #radial vector R = r - proj
        radialx = rx - projx
        radialy = ry - projy
        radialz = rz - projz
        #distance perpendicular to axis larger than r? Outside.
        if (radialx*radialx + radialy*radialy + radialz*radialz) > {self._r**2}:
            return {True if self._region == 'outside' else False}
        return {False if self._region == 'outside' else True}
        '''
        return string

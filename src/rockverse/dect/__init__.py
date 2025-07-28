"""
Provides classes and functions for managing
Dual Energy Computed Tomography (DECT) Monte Carlo processing. This module
is optimized for parallel computation across multiple CPUs or
GPUs using MPI (Message Passing Interface).

For details about the methods used in this module, please see reference:
    1. Original research paper: http://dx.doi.org/10.1002/2017JB014408


.. versionadded:: 0.3.0
    Initial release of the `rockverse.dualenergyct` module.

.. versionchanged:: 1.0.1
    Name changed to `rockverse.dect`

.. todo:
    - Get processing parameters from config

"""

### NOTE ###
# All functions are built expecting zarr LocalStore #

import copy
import numpy as np
import pandas as pd
import zarr
import hashlib
import shutil
from datetime import datetime
from scipy.special import erf
from mpi4py import MPI
from rockverse._utils import rvtqdm, datetimenow
from rockverse.configure import config
from rockverse.optimize import gaussian_fit, gaussian_val

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs

from rockverse.voxel_image import (
    VoxelImage,
    full_like)

from rockverse import _assert
from rockverse.errors import collective_raise, collective_only_rank0_runs
from rockverse._utils import collective_print
from rockverse.dect._periodic_table import ATOMIC_NUMBER_AND_MASS_DICT
from rockverse.dect._gpu_functions import (
    coeff_matrix_broad_search_gpu,
    reset_arrays_gpu,
    calc_rhoZ_arrays_gpu
    )
from rockverse.dect._corefunctions import make_index, error_value
from rockverse.dect._cpu_functions import (
    fill_coeff_matrix_cpu,
    calc_rhoZ_arrays_cpu
    )
from rockverse.dect._hash_functions import (
    hash_input_data,
    need_coefficient_matrices,
    hash_coefficient_matrices,
    need_output_array
    )

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

_STATUS = ['lowECT',
          'highECT',
          'segmentation',
          'mask',
          'calibration material 0: --- NOT CHECKED ---',
          'calibration material 1: --- NOT CHECKED ---',
          'calibration material 2: --- NOT CHECKED ---',
          'calibration material 3: --- NOT CHECKED ---',
          'calibration coefficient matrices: --- OUTDATED OR NOT SET. ---',
          'rho_min: --- NEEDS RESTART. ---',
          'rho_p25: --- NEEDS RESTART. ---',
          'rho_p50: --- NEEDS RESTART. ---',
          'rho_p75: --- NEEDS RESTART. ---',
          'rho_max: --- NEEDS RESTART. ---',
          'Z_min: --- NEEDS RESTART. ---',
          'Z_p25: --- NEEDS RESTART. ---',
          'Z_p50: --- NEEDS RESTART. ---',
          'Z_p75: --- NEEDS RESTART. ---',
          'Z_max: --- NEEDS RESTART. ---',
          'valid: --- NEEDS RESTART. ---']


class PeriodicTable():
    """
    Manages periodic table information for the Monte Carlo Dual Energy Computed
    Tomography (DECT) processing.
    This class allows for the retrieval and modification of atomic numbers and
    masses for elements used in DECT processing.

    .. note::
        This class is designed to be created and managed within the main
        :class:`DECTGroup`. It should not be called directly.

    Parameters
    ----------
    zgroup : zarr.hierarchy.Group
        The Zarr group where the main :class:`DECTGroup` is stored.

    Examples
    --------

    Create a dual energy CT group. The periodic table will be created using
    default values:

        >>> import rockverse as rv
        >>> dectgroup = rv.dect.create_group('/path/to/group/dir')
        >>> dectgroup.periodic_table

    You can get the values for a specific element using a key in the format
    '<element>/Z' for atomic number or '<element>/M' for atomic mass:

        >>> dectgroup.periodic_table['O/Z'] # Oxygen atomic number
        8
        >>> dectgroup.periodic_table['C/M'] # Carbon atomic mass
        12.011

    If you want to fine tune Z or M values, you can set new values just like
    you get current values:

        >>> dectgroup.periodic_table['C/M']  get current value
        12.011
        >>> dectgroup.periodic_table['C/M'] = 12.109999 # set new value
        >>> dectgroup.periodic_table['C/M'] # check...
        12.109999

    Exceptions will be raised if the element is not in the table, if value for
    Z is not a positive integer, or if value for M is not a positive integer or
    float.

    """

    def __init__(self, zgroup):
        """
        Initializes the PeriodicTable instance.

        Parameters
        ----------
        zgroup : zarr.hierarchy.Group
            The Zarr group where the main :class:`DECTGroup` is stored.
        """
        self.zgroup = zgroup

    def _get_full_table(self):
        """
        Retrieves the dictionary with the full periodic table from the Zarr group.
        """
        ZM_table = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                ZM_table = self.zgroup.attrs['ZM_table']
        ZM_table = comm.bcast(ZM_table, root=0)
        return ZM_table


    def _split_key(self, key):
        """
        Splits a key in the format '<element>/<property>' into its components.
        """
        error, element, prop = False, None, None
        if key.find('/') < 0:
            error = True
        else:
            element, prop = key.split('/')
        if error or prop not in ('Z', 'M'):
            collective_raise(KeyError(
                "Key must be in format '<element>/Z' for atomic number or "
                "'<element>/M' for atomic mass."))
        return element, prop


    def __getitem__(self, key):
        """
        Retrieves the value of a specific property for a given element.
        The key in the format '<element>/<property>'.
        """
        element, prop = self._split_key(key)
        ZM_table = self._get_full_table()
        if element not in ZM_table:
            collective_raise(KeyError(f"Element {element} not found in database."))
        return ZM_table[element][prop]


    def __setitem__(self, key, value):
        """
        Sets the value of a specific property for a given element.
        The key in the format '<element>/<property>'.
        Raises KeyError if the element is not found in the database.
        The value must be positive integer for Z and positive int or float for M
        Raises ValueError for wrong formats.
        """
        element, prop = self._split_key(key)
        if prop == 'Z' and (not isinstance(value, int) or value<=0):
            collective_raise(ValueError('Atomic number Z expects positive integer value.'))
        if prop == 'M' and (not isinstance(value, (int, float)) or value<=0):
            collective_raise(ValueError('Atomic mass M expects positive numeric value.'))
        ZM_table = self._get_full_table()
        if element not in ZM_table:
            collective_raise(KeyError(
                "New elements must be added using the add_element method."))
        ZM_table[element][prop] = value
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                self.zgroup.attrs['ZM_table'] = copy.deepcopy(ZM_table)

    def add_element(self, name, Z, M):
        """
        Adds a new element to the periodic table.

        Parameters
        ----------
        name : str
            The name of the element.
        Z : int or float
            The atomic number of the element.
        M : int or float
            The atomic mass of the element.

        Example
        -------

        Use this method to add new elements to the table:

        >>> dectgroup.periodic_table['Hi/M'] # This will raise KeyError
        ...
        KeyError: 'Element Hi not found in database.'
        >>> dectgroup.periodic_table.add_element(name='Hi', Z=123, M=244.345)
        >>> dectgroup.periodic_table['Hi/Z']
        123
        >>> dectgroup.periodic_table['Hi/M']
        244.345
        """
        if not isinstance(name, str):
            collective_raise(ValueError('Element name expects string.'))
        if not isinstance(Z, (int, float)):
            collective_raise(ValueError('Atomic number Z expects integer value.'))
        if not isinstance(M, (int, float)):
            collective_raise(ValueError('Atomic mass M expects numeric value.'))
        ZM_table = self._get_full_table()
        ZM_table[name] = {'Z': Z, 'M': M}
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                self.zgroup.attrs['ZM_table'] = copy.deepcopy(ZM_table)

    def as_dataframe(self):
        """
        Returns the periodic table as a pandas DataFrame.

        Example
        -------

            >>> import rockverse as rv
            >>> dectgroup = rv.dect.create_group('/path/to/group/dir')
            >>> dectgroup.periodic_table.as_dataframe()
                Z          M
            H     1    1.00797
            He    2    4.00260
            Li    3    6.94100
            Be    4    9.01218
            B     5   10.81000
            ..  ...        ...
            Sg  106  263.00000
            Bh  107  262.00000
            Hs  108  255.00000
            Mt  109  256.00000
            Ds  110  269.00000
        """
        ZM_table = self._get_full_table()
        index = list(ZM_table.keys())
        Z = [ZM_table[k]['Z'] for k in index]
        M = [ZM_table[k]['M'] for k in index]
        return pd.DataFrame({'Z': Z, 'M': M}, index=index).sort_values(by='Z')

    def as_dict(self):
        """
        Returns the periodic table as a dictionary.

        Example
        -------

            >>> import rockverse as rv
            >>> dectgroup = rv.dect.create_group('/path/to/group/dir')
            >>> dectgroup.periodic_table.as_dict()
            {'Ac': {'M': 227.0278, 'Z': 89},
            'Ag': {'M': 107.868, 'Z': 47},
            'Al': {'M': 26.98154, 'Z': 13},
            'Am': {'M': 243, 'Z': 95},
            .
            .
            .
            'Xe': {'M': 131.3, 'Z': 54},
            'Y': {'M': 88.9059, 'Z': 39},
            'Yb': {'M': 173.04, 'Z': 70},
            'Zn': {'M': 65.38, 'Z': 30},
            'Zr': {'M': 91.22, 'Z': 40}}

        """
        return self._get_full_table()


class CalibrationMaterial():
    """
    Manages calibration material information for the Monte Carlo Dual Energy
    Computed Tomography (DECT) processing.

    .. note::
        This class is designed to be created and managed within the main
        :class:`DECTGroup`. It should not be called directly.

        All attribute get and set operations are MPI collective. Make sure
        all your MPI processes call these functions when running within
        a parallel environment.


    Parameters
    ----------
    zgroup : zarr.hierarchy.Group
        The Zarr group where the main :class:`DECTGroup` is stored.
    index : {0, 1, 2, 3}
        The index of the calibration material.
    """

    __slots__ = ['index', 'zgroup']

    def __init__(self, zgroup, index):
        self.index = index
        self.zgroup = zgroup

    def _get_attribute(self, attr_name):
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                if attr_name in self.zgroup.attrs["calibration_material"][self.index]:
                    value = self.zgroup.attrs["calibration_material"][self.index][attr_name]
        value = comm.bcast(value, root=0)
        return value

    def _set_attribute(self, attr_name, attr_value):
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                attrs = self.zgroup.attrs.asdict()
                attrs["calibration_material"][self.index][attr_name] = attr_value
                self.zgroup.attrs.update(attrs)

    def _check_if_material_0(self, property):
        if self.index == '0':
            collective_raise(AttributeError('Calibration material 0 is empty space '
                                            f'(no {property} attribute).'))

    @property
    def description(self):
        """
        String with material name or description.

        Examples
        --------

        To get the current description:

            >>> material_description = calibration_material.description

        To set a new description:

            >>> calibration_material.description = "Water"

        """
        return self._get_attribute('description')

    @description.setter
    def description(self, v):
        _assert.instance('description', v, 'string', (str,))
        self._set_attribute('description', v)

    @property
    def bulk_density(self):
        """
        Bulk density for the calibration material
        in grams per cubic centimeter (g/cc).

        Examples
        --------

        To get the current bulk density:

            >>> density = calibration_material.bulk_density

        To set a new bulk density:

            >>> calibration_material.bulk_density = 1.0  # Setting density for water

        """
        self._check_if_material_0('bulk_density')
        return self._get_attribute('bulk_density')


    @bulk_density.setter
    def bulk_density(self, v):
        _assert.condition.positive_integer_or_float('bulk_density', v)
        self._check_if_material_0('bulk_density')
        self._set_attribute('bulk_density', float(v))

    @property
    def composition(self):
        """
        Dictionary defining the chemical composition of the calibration material.
        Must be set as ``key: value`` pairs where ``key`` is the element symbol
        and ``value`` is the proportionate number of atoms of each element.
        The element has to be a valid symbol in the :class:`PeriodicTable`.

        Examples:

        Water H\\ :sub:`2`\\ O:

        .. code-block:: python

            dectgroup.calibration_material[1].composition = {'H': 2, 'O': 1}

        Silica SiO\\ :sub:`2`:

        .. code-block:: python

            dectgroup.calibration_material[2].composition = {'Si': 1, 'O': 2}

        Dolomite CaMg(CO\\ :sub:`3`)\\ :sub:`2`:

        .. code-block:: python

            dectgroup.calibration_material[3].composition = {'Ca': 1, 'Mg': 1, 'C': 2, 'O': 6}

        Teflon (C\\ :sub:`2`\\ F\\ :sub:`4`\\ )\\ :sub:`n`:

        .. code-block:: python

            dectgroup.calibration_material[3].composition = {'C': 2, 'F': 4}
        """
        self._check_if_material_0('composition')
        return self._get_attribute('composition')

    @composition.setter
    def composition(self, v):
        _assert.instance('composition', v, 'dict', (dict,))
        self._check_if_material_0('composition')
        table = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                table = self.zgroup.attrs["ZM_table"]
        table = comm.bcast(table, root=0)
        elements = set(table.keys())
        for key, value in v.items():
            if key not in elements:
                collective_raise(ValueError(f"Setting {v}: element {key} not in periodic table."))
            if not isinstance(value, (int, float)):
                collective_raise(ValueError(f"Setting {v}: invalid proportionate number for {key}."))
        self._set_attribute('composition', v)


    def _get_pdf(self, name):
        new_array = None
        with collective_only_rank0_runs(id='getting'):
            array_name = f'{name}_standard{self.index}_pdf'
            if mpi_rank == 0:
                if array_name in self.zgroup:
                    new_array = self.zgroup[array_name][...]
        new_array = comm.bcast(new_array, root=0)
        if new_array is not None:
            return np.ascontiguousarray(new_array[:, 0]), np.ascontiguousarray(new_array[:, 1])
        return None


    def _set_pdf(self, name, value):
        if value is None:
            with collective_only_rank0_runs():
                if mpi_rank == 0:
                    array_name = f'{name}_standard{self.index}_pdf'
                    if array_name in self.zgroup:
                        path = self.zgroup.store.root / array_name
                        shutil.rmtree(path)
            return

        if (not isinstance(value, (tuple, list))
            or len(value) != 2
            or not isinstance(value[0], (list, tuple, np.ndarray))
            or not isinstance(value[1], (list, tuple, np.ndarray))
            or len(value[0]) != len(value[1])):
            collective_raise(ValueError(
                'Probability density function (pdf) must be a 2-element list '
                'or tuple containing equal size arrays with x and y pdf values.'))
        if any(value[1] < 0):
            collective_raise(ValueError(
                'Probability density function (pdf) values cannot be negative.'))
        if np.sum(value[1]) == 0:
            collective_raise(ValueError(
                'Probability density function (pdf) has to have at least one positive value.'))
        with collective_only_rank0_runs(id=f'{mpi_rank} aqui'):
            if mpi_rank == 0:
                # pdf normalization
                x = value[0].astype(float).copy()
                y = value[1].astype(float).copy()
                sum = 0.
                for k in range(1, len(x)):
                    sum += (y[k]+y[k-1])*(x[k]-x[k-1])*0.5
                z = zarr.create_array(store=self.zgroup.store,
                                      name=f'{name}_standard{self.index}_pdf',
                                      shape=(len(x), 2),
                                      chunks=(len(x), 2),
                                      dtype=float,
                                      overwrite=True)
                ind = np.argsort(x)
                z[:, 0] = x[ind]
                z[:, 1] = y[ind]/sum

    def _get_cdf(self, name):
        xy = self._get_pdf(name)
        if xy is None:
            return None
        x, y = xy
        sum = 0*x
        sum[0] = y[0] * (x[1]-x[0])
        for k in range(1, len(x)-1):
            sum[k] += sum[k-1] + y[k]*(x[k+1]-x[k-1])/2
        sum[-1] = sum[-2] + y[-1] * (x[-1]-x[-2])
        sum /= sum[-1]
        return np.ascontiguousarray(x), np.ascontiguousarray(sum)

    @property
    def lowE_pdf(self):
        """
        .. _dect_calibrationy_lowE:

        Low energy CT attenuation probability density function (PDF).

        This property is a two-element tuple containing:
        - x: The attenuation values.
        - y: The corresponding PDF values.

        When setting the (x, y) tuple, the values for y do not need to be normalized,
        as data normalization will occur before being stored.

        Examples
        --------
        To get the current lowE PDF:

            >>> x, y = calibration_material.lowE_pdf

        To set a new lowE PDF:

            >>> calibration_material.lowE_pdf = (new_x_values, new_y_values)

        Set to None to remove:

            >>> calibration_material.lowE_pdf = None

        """
        return self._get_pdf('lowE')

    @lowE_pdf.setter
    def lowE_pdf(self, v):
        return self._set_pdf('lowE', v)

    @property
    def lowE_cdf(self):
        """
        Low energy CT attenuation cumulative density function (CDF).

        This property is a two-element tuple containing:
        - x: The attenuation values.
        - y: The corresponding CDF values.

        This is a read-only property calculated from the `lowE_pdf` using
        numerical integration with the trapezoidal rule.

        Examples
        --------
        Get the current lowE CDF:

            >>> x, y = calibration_material.lowE_cdf
        """
        return self._get_cdf('lowE')

    @property
    def highE_pdf(self):
        """
        High energy CT attenuation probability density function (PDF).

        This property is a two-element tuple containing:
        - x: The attenuation values.
        - y: The corresponding PDF values.

        When setting the (x, y) tuple, the values for y do not need to be normalized,
        as data normalization will occur before being stored.

        Examples
        --------
        To get the current highE PDF:

            >>> x, y = calibration_material.highE_pdf

        To set a new highE PDF:

            >>> calibration_material.highE_pdf = (new_x_values, new_y_values)

        Set to None to remove:

            >>> calibration_material.highE_pdf = None

        """
        return self._get_pdf('highE')

    @highE_pdf.setter
    def highE_pdf(self, v):
        return self._set_pdf('highE', v)

    @property
    def highE_cdf(self):
        """
        High energy CT attenuation cumulative density function (CDF).

        This property is a two-element tuple containing:
        - x: The attenuation values.
        - y: The corresponding CDF values.

        This is a read-only property calculated from the `highE_pdf` using
        numerical integration with the trapezoidal rule.

        Examples
        --------
        Get the current highE CDF:

            >>> x, y = calibration_material.highE_cdf
        """
        return self._get_cdf('highE')

    def _check_gaussian_pdf_setter(self, v):
        conditions = [v is None,
                      isinstance(v, tuple) and len(v) == 2 and all(isinstance(k, (int, float)) for k in v)]
        if not any(conditions):
            collective_raise(ValueError(
                f"Gaussian pdf must be set as None or a two-element tuple of int for float."))

    @property
    def lowE_gaussian_pdf(self):
        """
        Low energy CT attenuation Gaussian probability density function (PDF).

        A tuple :math:`(\mu, \sigma)` with the mean and standard deviation values
        for a Gaussian (normal) probability density function model for
        low energy attenuation values:

        .. math::

            y(x) = \\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\\frac{1}{2}\left(\\frac{x-\mu}{\sigma}\\right)^2}.


        When set, gets precedence over the :obj:`lowE_pdf <rockverse.dect.CalibrationMaterial.lowE_pdf>` attribute.


        Examples
        --------
        To get the current lowE Gaussian PDF:

            >>> mean, std = calibration_material.lowE_gaussian_pdf

        To set a new lowE Gaussian PDF:

            >>> calibration_material.lowE_gaussian_pdf = (new_mean, new_std)

        To unset the lowE Gaussian PDF model:

            >>> calibration_material.lowE_pdf = None
        """
        return self._get_attribute('lowE_gaussian_pdf')

    @lowE_gaussian_pdf.setter
    def lowE_gaussian_pdf(self, v):
        self._check_gaussian_pdf_setter(v)
        self._set_attribute('lowE_gaussian_pdf', v)

    @property
    def highE_gaussian_pdf(self):
        """
        High energy CT attenuation Gaussian probability density function (PDF).


        A tuple :math:`(\mu, \sigma)` with the mean and standard deviation values
        for a Gaussian (normal) probability density function model for
        high energy attenuation values:

        .. math::

            y(x) = \\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\\frac{1}{2}\left(\\frac{x-\mu}{\sigma}\\right)^2}.


        When set, gets precedence over the :obj:`highE_pdf <rockverse.dect.CalibrationMaterial.highE_pdf>` attribute.

        Examples
        --------
        To get the current highE Gaussian PDF:

            >>> mean, std = calibration_material.highE_gaussian_pdf

        To set a new highE Gaussian PDF:

            >>> calibration_material.highE_gaussian_pdf = (new_mean, new_std)

        To unset the highE Gaussian PDF model:

            >>> calibration_material.lowE_pdf = None
        """
        return self._get_attribute('highE_gaussian_pdf')

    @highE_gaussian_pdf.setter
    def highE_gaussian_pdf(self, v):
        self._check_gaussian_pdf_setter(v)
        self._set_attribute('highE_gaussian_pdf', v)


    def as_dict(self):
        """
        Outputs a dictionary containing the class attributes:

        Examples
        --------
        Get the calibration material attributes as a dictionary:

            >>> attributes_dict = calibration_material.as_dict()
        """
        items = {'description': self.description}
        if self.index != '0':
            items['bulk_density'] = self.bulk_density
            items['composition'] = self.composition
        items['lowE_pdf'] = self.lowE_pdf
        items['lowE_cdf'] = self.lowE_cdf
        items['highE_pdf'] = self.highE_pdf
        items['highE_cdf'] = self.highE_cdf
        items['lowE_gaussian_pdf'] = self.lowE_gaussian_pdf
        items['highE_gaussian_pdf'] = self.highE_gaussian_pdf
        return items


    def fit_lowE_gaussian_pdf(self, order=1):
        '''
        Fit a Gaussian distribution to the PDF values provided in the
        :obj:`lowE_pdf <rockverse.dect.CalibrationMaterial.lowE_pdf>` attribute.

        Parameters
        ----------
        order : {int, float, inf, -inf}, optional
            The order of the error norm to be minimized (e.g., 2 for least squares, 1 for L1 norm).
        '''
        if self.lowE_pdf is None:
            collective_raise(Exception('You need to set the lowE_pdf attribute before calling fit_lowE_gaussian_pdf.'))
        c = gaussian_fit(*self.lowE_pdf, order=order)
        self.lowE_gaussian_pdf = tuple(c[1:])


    def fit_highE_gaussian_pdf(self, order=1):
        '''
        Fit a Gaussian distribution to the PDF values provided in the
        :obj:`highE_pdf <rockverse.dect.CalibrationMaterial.highE_pdf>` attribute.

        Parameters
        ----------
        order : {int, float, inf, -inf}, optional
            The order of the error norm to be minimized (e.g., 2 for least squares, 1 for L1 norm).
        '''
        if self.highE_pdf is None:
            collective_raise(Exception('You need to set the highE_pdf attribute before calling fit_highE_gaussian_pdf.'))
        c = gaussian_fit(*self.highE_pdf, order=order)
        self.highE_gaussian_pdf = tuple(c[1:])


    def hash(self):
        """
        Calculates the MD5 hash representation for the class instance.

        This hash value is used to verify the integrity of the calibration
        material's attributes and to track changes over time.

        Returns
        -------
        str
            The MD5 hash string representing the current state of the class instance.

        Examples
        --------
        To get the hash representation of the calibration material:

            >>> material_hash = calibration_material.hash()
        """
        md5 = hashlib.md5()
        if self.index != '0' and self.bulk_density is not None:
            md5.update(np.array([float(self.bulk_density),], dtype=float))
        if self.index != '0' and self.composition is not None:
            composition = self.composition
            for k in sorted(composition.keys()):
                md5.update(k.encode('ascii'))
                md5.update(np.array([float(composition[k]),], dtype=float))

        if self.lowE_gaussian_pdf is not None:
            md5.update(np.array(self.lowE_gaussian_pdf, dtype=float))
        elif self.lowE_pdf is not None:
            x, y = self.lowE_pdf
            md5.update(np.array(x, dtype=float))
            md5.update(np.array(y, dtype=float))

        if self.highE_gaussian_pdf is not None:
            md5.update(np.array(self.highE_gaussian_pdf, dtype=float))
        elif self.highE_pdf is not None:
            x, y = self.highE_pdf
            md5.update(np.array(x, dtype=float))
            md5.update(np.array(y, dtype=float))

        return md5.hexdigest()

    def check(self):
        """
        Checks for missing or incorrect properties in the calibration material.

        Returns
        -------
        str
            A summary of missing or incorrect properties, if any.
            If all properties are valid, returns an empty string.
        """
        msg = ""
        keys = ['description',]
        if self.index != '0':
            keys += ['bulk_density', 'composition']
        for key in keys:
            if self.__getattribute__(key) is None:
                msg = msg + f"\n    - =====> Missing {key}."
        if self.lowE_pdf is None and self.lowE_gaussian_pdf is None:
            msg = msg + f"\n    - =====> Needs lowE_pdf or lowE_gaussian_pdf"
        if self.highE_pdf is None and self.highE_gaussian_pdf is None:
            msg = msg + f"\n    - =====> Needs highE_pdf or highE_gaussian_pdf"
        if msg:
            return f"Calibration material {self.index}:" + msg
        return msg


    def _rhohat_Zn_values(self):
        """
        Calculate the electron density (rhohat) and atomic number values for the calibration material.
        """
        atomic_number_and_mass = self.zgroup.attrs['ZM_table']
        composition = self.composition
        missing = [k for k in composition.keys() if k not in atomic_number_and_mass]
        if len(missing) == 1:
            collective_raise(Exception(
                f"Element {missing[0]} not found in database."))
        elif len(missing) > 1:
            collective_raise(Exception(
                f"Elements ({', '.join(missing)}) not found in element database."))

        # Z, M, quantity
        values = np.zeros((len(composition), 3), dtype='f8')
        for i, (k, v) in enumerate(composition.items()):
            values[i][0] = atomic_number_and_mass[k]['Z']
            values[i][1] = atomic_number_and_mass[k]['M']
            values[i][2] = v
        sumZ = np.sum(values[:, 2]*values[:, 0])
        sumM = np.sum(values[:, 2]*values[:, 1])
        rhohat = np.float64(self.bulk_density*2.0*sumZ/sumM)
        return rhohat, values


class DECTGroup():
    """
    Manages Dual Energy Computed Tomography (DECT) processing.
    This class builds upon
    `Zarr groups <https://zarr.readthedocs.io/en/stable/user-guide/groups.html>`_
    and is adapted for MPI (Message Passing Interface) processing,
    enabling parallel computation across multiple CPUs or GPUs.

    For a detailed workflow, refer to the
    `original research paper <http://dx.doi.org/10.1002/2017JB014408>`_.

    .. note::
        This class should not be directly instantiated.
        Use the :func:`create_group` instead.

    Parameters
    ----------
    zgroup : zarr group
        An existing Zarr group.
    """

    # Process model master-slave commanded by rank 0 using zarr local store

    def _get_attribute(self, attr_name):
        value = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                value = self.zgroup.attrs[attr_name]
        value = comm.bcast(value, root=0)
        return value

    def _set_attribute(self, attr_name, attr_value):
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                attrs = self.zgroup.attrs.asdict()
                attrs[attr_name] = attr_value
                self.zgroup.attrs.update(attrs)

    def __init__(self, zgroup):
        self.zgroup = zgroup
        self._calibration_material = [CalibrationMaterial(self.zgroup, "0"),
                                      CalibrationMaterial(self.zgroup, "1"),
                                      CalibrationMaterial(self.zgroup, "2"),
                                      CalibrationMaterial(self.zgroup, "3")]
        self._periodic_table = PeriodicTable(self.zgroup)
        self.current_hashes = {'lowE': '',
                               'highE': '',
                               'mask': '',
                               'segmentation': '',
                               'calibration_material0': '',
                               'calibration_material1': '',
                               'calibration_material2': '',
                               'calibration_material3': ''}

    # Arrays ----------------------------------------------

    @property
    def lowECT(self):
        """
        The low energy computed tomography voxel image.
        Returns ``None`` if not set. Can only be changed through the
        :ref:`array creation methods <dect_array_creation>`.
        """
        if 'lowECT' in self.zgroup:
            return VoxelImage(self.zgroup['lowECT'])
        return None

    @property
    def highECT(self):
        """
        The high energy computed tomography voxel image.
        Returns ``None`` if not set. Can only be changed through the
        :ref:`array creation methods <dect_array_creation>`.
        """
        if 'highECT' in self.zgroup:
            return VoxelImage(self.zgroup['highECT'])
        return None

    @property
    def mask(self):
        """
        The mask voxel image. Returns ``None`` if not set. Masked voxels
        will be ignored during the Monte Carlo inversion.
        Can only be changed through the
        :ref:`array creation methods <dect_array_creation>`.
        """
        if 'mask' in self.zgroup:
            return VoxelImage(self.zgroup['mask'])
        return None


    @property
    def segmentation(self):
        """
        The segmentation voxel image. Returns ``None`` if not set.
        This array is used to locate the calibration materials in
        the image for the pre-processing steps.
        Can only be changed through the
        :ref:`array creation methods <dect_array_creation>`.
        """
        if 'segmentation' in self.zgroup:
            return VoxelImage(self.zgroup['segmentation'])
        return None

    @property
    def rho_min(self):
        """
        Voxel image with the minimum electron density per voxel.
        Minimum value is taken as the lower boxplot whisker boundary
        from the Monte Carlo inversion.
        """
        if 'rho_min' in self.zgroup:
            return VoxelImage(self.zgroup['rho_min'])
        return None

    @property
    def rho_p25(self):
        """
        Voxel image with the the first quartile (25th percentile) for
        the electron density values per voxel from the Monte Carlo
        inversion.
        """
        if 'rho_p25' in self.zgroup:
            return VoxelImage(self.zgroup['rho_p25'])
        return None

    @property
    def rho_p50(self):
        """
        Voxel image with the the median (50th percentile) values
        for the electron density per voxel from the Monte Carlo
        inversion.
        """
        if 'rho_p50' in self.zgroup:
            return VoxelImage(self.zgroup['rho_p50'])
        return None

    @property
    def rho_p75(self):
        """
        Voxel image with the the third quartile (75th percentile)
        for the electron density values per voxel from the Monte Carlo
        inversion.
        """
        if 'rho_p75' in self.zgroup:
            return VoxelImage(self.zgroup['rho_p75'])
        return None

    @property
    def rho_max(self):
        """
        Voxel image with the maximum electron density per voxel.
        Maximum value is taken as the upper boxplot whisker boundary
        from the Monte Carlo inversion.
        """
        if 'rho_max' in self.zgroup:
            return VoxelImage(self.zgroup['rho_max'])
        return None

    @property
    def Z_min(self):
        """
        Voxel image with the minimum effective atomic number per voxel.
        Minimum value is taken as the lower boxplot whisker boundary
        from the Monte Carlo inversion.
        """
        if 'Z_min' in self.zgroup:
            return VoxelImage(self.zgroup['Z_min'])
        return None

    @property
    def Z_p25(self):
        """
        Voxel image with the the first quartile (25th percentile) for the
        effective atomic number values per voxel from the Monte
        Carlo inversion.
        """
        if 'Z_p25' in self.zgroup:
            return VoxelImage(self.zgroup['Z_p25'])
        return None

    @property
    def Z_p50(self):
        """
        Voxel image with the the median values (50th percentile) for the
        effective atomic number per voxel from the Monte Carlo
        inversion.
        """
        if 'Z_p50' in self.zgroup:
            return VoxelImage(self.zgroup['Z_p50'])
        return None

    @property
    def Z_p75(self):
        """
        Voxel image with the the third quartile (75th percentile) for the
        effective atomic number values per voxel from the Monte
        Carlo inversion.
        """
        if 'Z_p75' in self.zgroup:
            return VoxelImage(self.zgroup['Z_p75'])
        return None

    @property
    def Z_max(self):
        """
        Voxel image with the maximum effective atomic number per voxel.
        Maximum value is taken as the upper boxplot whisker boundary
        from the Monte Carlo inversion.
        """
        if 'Z_max' in self.zgroup:
            return VoxelImage(self.zgroup['Z_max'])
        return None

    @property
    def valid(self):
        """
        Voxel image with the number of valid Monte Carlo results for each voxel.
        """
        if 'valid' in self.zgroup:
            return VoxelImage(self.zgroup['valid'])
        return None

    # Calibration materials and periodic table ----------------------

    @property
    def calibration_material(self):
        """
        A list containing four instances of the :class:`CalibrationMaterial`
        class, each representing a different calibration material used in
        Dual Energy Computed Tomography (DECT) processing.
        """
        return self._calibration_material

    @property
    def periodic_table(self):
        """
        An instance of the :class:`PeriodicTable` class that provides access to atomic
        number and atomic mass values for elements used in the Monte Carlo inversion.
        """
        return self._periodic_table

    @property
    def maxA(self):
        """
        Maximum value for inversion coefficient $A$ in the broad search algorithm
        used during Monte Carlo simulations.

        Examples
        --------
        To get the current value of maxA:

            >>> current_maxA = dectgroup.maxA

        To set a new value for maxA:

            >>> dectgroup.maxA = 2500.0  # Adjusting for specific data requirements
        """
        return self._get_attribute('maxA')

    @maxA.setter
    def maxA(self, v):
        _assert.instance('maxA', v, 'number', (float, int))
        self._set_attribute('maxA', v)

    @property
    def maxB(self):
        """
        Maximum value for inversion coefficient $B$ in the broad search algorithm
        used during Monte Carlo simulations.

        Examples
        --------
        To get the current value of maxA:

            >>> current_maxB = dectgroup.maxB

        To set a new value for maxA:

            >>> dectgroup.maxB = 500.0  # Adjusting for specific data requirements
        """
        return self._get_attribute('maxB')

    @maxB.setter
    def maxB(self, v):
        _assert.instance('maxB', v, 'number', (float, int))
        self._set_attribute('maxB', v)

    @property
    def maxn(self):
        """
        Maximum value for inversion coefficient $n$ in the broad search algorithm
        used during Monte Carlo simulations.

        Examples
        --------
        To get the current value of maxA:

            >>> current_maxn = dectgroup.maxn

        To set a new value for maxA:

            >>> dectgroup.maxn = 15  # Adjusting for specific data requirements
        """
        return self._get_attribute('maxn')

    @maxn.setter
    def maxn(self, v):
        _assert.instance('maxn', v, 'number', (float, int))
        self._set_attribute('maxn', v)

    @property
    def lowE_inversion_coefficients(self):
        """
        Pandas DataFrame with the valid realization sets for low energy inversion
        coefficients. Returns ``None`` if not calculated. Can be set to None
        to delete previous calculations.
        """
        tol = self.tol
        matrix = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                if 'matrixl' in self.zgroup:
                    matrix = self.zgroup['matrixl'][...]
        matrix = comm.bcast(matrix, root=0)
        if matrix is None:
            return None
        ind = matrix[:, -1]<=tol
        if not any(ind):
            return None
        matrix = matrix[ind, :]
        columns = ['CT_0', 'CT_1', 'CT_2', 'CT_3', 'Z_1', 'Z_2', 'Z_3', 'A', 'B', 'n', 'err']
        return pd.DataFrame(data=matrix, index=None, columns=columns, dtype='f8', copy=True)


    @lowE_inversion_coefficients.setter
    def lowE_inversion_coefficients(self, v):
        if v is not None:
            collective_raise(ValueError(
                "lowE_inversion_coefficients can only be directly set to None to delete previous calculations."
                " Use preprocess() or run() methods if you want to update the values."))
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                array_name = 'matrixl'
                if array_name in self.zgroup:
                    path = self.zgroup.store.root / array_name
                    shutil.rmtree(path)


    @property
    def highE_inversion_coefficients(self):
        """
        Pandas DataFrame with the valid realization sets for high energy inversion
        coefficients. Returns ``None`` if not calculated. Can be set to None
        to delete previous calculations.
        """
        tol = self.tol
        matrix = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                if 'matrixh' in self.zgroup:
                    matrix = self.zgroup['matrixh'][...]
        matrix = comm.bcast(matrix, root=0)
        if matrix is None:
            return None
        ind = matrix[:, -1]<=tol
        if not any(ind):
            return None
        matrix = matrix[ind, :]
        columns = ['CT_0', 'CT_1', 'CT_2', 'CT_3', 'Z_1', 'Z_2', 'Z_3', 'A', 'B', 'n', 'err']
        return pd.DataFrame(data=matrix, index=None, columns=columns, dtype='f8', copy=True)


    @highE_inversion_coefficients.setter
    def highE_inversion_coefficients(self, v):
        if v is not None:
            collective_raise(ValueError(
                "highE_inversion_coefficients can only be directly set to None to delete previous calculations."
                " Use preprocess() or run() methods if you want to update the values."))
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                array_name = 'matrixh'
                if array_name in self.zgroup:
                    path = self.zgroup.store.root / array_name
                    shutil.rmtree(path)


    @property
    def threads_per_block(self):
        """
        Number of threads per block when processing using GPUs.
        Use it for fine-tuning GPU performance based on your specific
        GPU capabilities.
        """
        return self._get_attribute('threads_per_block')

    @threads_per_block.setter
    def threads_per_block(self, v):
        _assert.instance('threads_per_block', v, 'int', (int,))
        self._set_attribute('threads_per_block', v)

    @property
    def tol(self):
        """
        Tolerance value for terminating the Newton-Raphson optimizations.
        """
        return self._get_attribute('tol')

    @tol.setter
    def tol(self, v):
        _assert.instance('tol', v, 'float', (float,))
        self._set_attribute('tol', v)

    @property
    def required_iterations(self):
        """
        The required number of valid Monte Carlo iterations for each voxel.
        """
        return self._get_attribute('required_iterations')

    @required_iterations.setter
    def required_iterations(self, v):
        _assert.instance('required_iterations', v, 'integer', (int,))
        self._set_attribute('required_iterations', v)

    @property
    def maximum_iterations(self):
        """
        The maximum number of trials to get valid Monte Carlo iterations per voxel.
        Recommended 10 times the required number of valid
        Monte Carlo iterations.
        """
        return self._get_attribute('maximum_iterations')

    @maximum_iterations.setter
    def maximum_iterations(self, v):
        _assert.instance('maximum_iterations', v, 'integer', (int,))
        self._set_attribute('maximum_iterations', v)

    @property
    def whis(self):
        """
        The boxplot whisker length for determining Monte Carlo outlier results.
        Minimum values will be at least :math:`Q_1-whis(Q_3-Q_2)` and maximum
        values will be at most :math:`Q_3+whis(Q_3-Q_2)`, where :math:`Q_1`,
        :math:`Q_2`, and :math:`Q_3` are the three quartiles for the Monte Carlo
        results. Default value is 1.5.
        """
        return self._get_attribute('whis')

    @whis.setter
    def whis(self, v):
        _assert.instance('whis', v, 'number', (int, float))
        self._set_attribute('whis', v)


    # Array creation ------------------------------------------------

    def create_mask(self, fill_value=False, overwrite=False,
                    field_name='mask', description='Mask voxel image', **kwargs):
        """
        Create a mask voxel image based on the lowECT one. This function enforces
        boolean dtype and the path '/mask' within the group store. These
        parameters will be ignored if passed as keyword arguments.

        Parameters
        ----------
        fill_value : bool, optional
            The value to fill the mask with (default is False).
        overwrite : bool, optional
            If True, overwrite existing mask (default is False).
        **kwargs
            Additional keyword arguments for the `rockverse.voxel_image.full_like`
            function used to create the mask array.
        """
        if self.lowECT is None:
            collective_raise(ValueError("lowECT must be set before creating mask."))
        kwargs['overwrite'] = overwrite
        kwargs['fill_value'] = fill_value
        kwargs['store'] = self.zgroup.store
        kwargs['path'] = 'mask'
        kwargs['dtype'] = 'b1'
        kwargs['field_name'] = field_name
        kwargs['description'] = description
        _ = full_like(self.lowECT, **kwargs)


    def delete_mask(self):
        """
        Remove the mask array from the structure.
        """
        with collective_only_rank0_runs():
            if mpi_rank == 0 and 'mask' in self.zgroup:
                zarr.storage.rmdir(self.zgroup.store, '/mask')
        comm.barrier()


    def create_segmentation(self, fill_value=0, dtype='u1', overwrite=False,
                            field_name='segmentation', description='Segmentation voxel image',
                            **kwargs):
        """
        Create segmentation voxel image based on the lowECT array. This function
        enforces unsigned integer dtype and the path '/segmentation' within
        the group store.

        Parameters
        ----------
        fill_value : bool, optional
            The value to fill the mask with.
        dtype : Numpy datatype
            Array data type. Must be unsigned integer.
        overwrite : bool, optional
            If True, overwrite existing segmentation.
        **kwargs
            Additional keyword arguments for the `rockverse.voxel_image.full_like` function used
            to create the segmentation array.
        """
        if self.lowECT is None:
            collective_raise(ValueError("lowECT must be set before creating segmentation."))
        kwargs['overwrite'] = overwrite
        kwargs['fill_value'] = fill_value
        kwargs['store'] = self.zgroup.store
        kwargs['path'] = 'segmentation'
        kwargs['dtype'] = dtype
        kwargs['field_name'] = field_name
        kwargs['description'] = description
        if np.dtype(kwargs['dtype']).kind != 'u':
            collective_raise(ValueError(
                "segmentation array dtype must be unsigned integer."))
        _ = full_like(self.lowECT, **kwargs)

    def delete_segmentation(self):
        """
        Remove the segmentation array from the structure.
        """
        with collective_only_rank0_runs():
            if mpi_rank == 0 and 'segmentation' in self.zgroup:
                zarr.storage.rmdir(self.zgroup.store, '/segmentation')
        comm.barrier()

    def copy_image(self, image, name, **kwargs):
        """
        Copy an existing voxel image into the DECT group.

        Parameters
        ----------
        image : VoxelImage
            The original voxel image to be copied.
        name : {'lowECT', 'highECT', 'mask', 'segmentation'}
            The path within the DECT group where the array will be stored.
        """
        _assert.rockverse_instance(image, 'image', ('VoxelImage',))
        _assert.in_group('path', name, ('lowECT', 'highECT', 'mask', 'segmentation'))
        if name == 'mask' and image.dtype.kind != 'b':
            collective_raise(ValueError("mask array dtype must be boolean."))
        if name == 'segmentation' and image.dtype.kind != 'u':
            collective_raise(ValueError("segmentation array dtype must be unsigned integer."))

        kwargs.update(**image.meta_data_as_dict)
        kwargs['store'] = self.zgroup.store
        kwargs['path'] = name
        _ = image.copy(**kwargs)


    def _check_input_data(self, status):
        """
        Check the structure for consistency and dependencies among arrays
        and processing parameters.
        """
        complete = True
        # Check if input datasets are set
        if self.lowECT is None:
            status[0] = "=====> lowECT array is not set."
            complete = False
        else:
            status[0] = "lowECT: " + self.lowECT.__repr__()

        if self.highECT is None:
            status[1] = "=====> highECT array is not set."
            complete = False
        else:
            status[1] = "highECT: " + self.highECT.__repr__()

        if self.segmentation is None:
            status[2] = "segmentation array is not set (optional)."
        else:
            status[2] = "segmentation: " + self.segmentation.__repr__()

        if self.mask is None:
            status[3] = "mask array is not set (optional)."
        else:
            status[3] = "mask: " + self.mask.__repr__()

        string = self.calibration_material[0].check()
        if string:
            status[4] = string
            complete = False
        else:
            status[4] = 'Calibration material 0 OK.'

        string = self.calibration_material[1].check()
        if string:
            status[5] = string
            complete = False
        else:
            status[5] = 'Calibration material 1 OK.'

        string = self.calibration_material[2].check()
        if string:
            status[6] = string
            complete = False
        else:
            status[6] = 'Calibration material 2 OK.'

        string = self.calibration_material[3].check()
        if string:
            status[7] = string
            complete = False
        else:
            status[7] = 'Calibration material 3 OK.'

        if not complete:
            return False

        # Check for shape, chunks and dtypes
        msg = ''
        shapes = [self.lowECT.shape, self.highECT.shape]
        chunks = [self.lowECT.chunks, self.highECT.chunks]
        tmp_msg = 'lowECT, highECT'
        if self.segmentation is not None:
            shapes.append(self.segmentation.shape)
            chunks.append(self.segmentation.chunks)
            tmp_msg = f'{tmp_msg}, segmentation'
        if self.mask is not None:
            shapes.append(self.mask.shape)
            chunks.append(self.mask.chunks)
            tmp_msg = f'{tmp_msg}, mask'
        if (any(shapes[k] != shapes[0] for k in range(len(shapes)))
            or any(chunks[k] != chunks[0] for k in range(len(chunks)))):
            msg = msg + "    - " + tmp_msg + " arrays must have same shape and chunk size.\n"
        if self.lowECT.dtype.kind not in 'uif':
            msg = msg + "    - lowECT dtype must be numeric.\n"
        if self.highECT.dtype.kind not in 'uif':
            msg = msg + "    - highECT dtype must be numeric.\n"
        if self.segmentation is not None and self.segmentation.dtype.kind != 'u':
            msg = msg + "    - segmentation dtype must be unsigned integer.\n"
        if self.mask is not None and self.mask.dtype.kind != 'b':
            msg = msg + "    - mask dtype must be boolean.\n"

        if msg:
            collective_raise(Exception('Invalid input arrays:\n'+msg))

        return True


    def _draw_coefficients(self):
        """
        Generate the calibration coefficient sets for low and high energy.
        """
        # Processing parameters -------------------------------------
        maximum, tol = self.maximum_iterations, self.tol

        # If Gaussian model, get mean and variance. Else, get cdf for interpolation
        if self.calibration_material[0].lowE_gaussian_pdf is not None:
            cdfxl0 = self.calibration_material[0].lowE_gaussian_pdf
            cdfyl0 = [0., 0.]
        else:
            cdfxl0, cdfyl0 = self.calibration_material[0].lowE_cdf

        if self.calibration_material[0].highE_gaussian_pdf is not None:
            cdfxh0 = self.calibration_material[0].highE_gaussian_pdf
            cdfyh0 = [0., 0.]
        else:
            cdfxh0, cdfyh0 = self.calibration_material[0].highE_cdf

        if self.calibration_material[1].lowE_gaussian_pdf is not None:
            cdfxl1 = self.calibration_material[1].lowE_gaussian_pdf
            cdfyl1 = [0., 0.]
        else:
            cdfxl1, cdfyl1 = self.calibration_material[1].lowE_cdf

        if self.calibration_material[1].highE_gaussian_pdf is not None:
            cdfxh1 = self.calibration_material[1].highE_gaussian_pdf
            cdfyh1 = [0., 0.]
        else:
            cdfxh1, cdfyh1 = self.calibration_material[1].highE_cdf

        if self.calibration_material[2].lowE_gaussian_pdf is not None:
            cdfxl2 = self.calibration_material[2].lowE_gaussian_pdf
            cdfyl2 = [0., 0.]
        else:
            cdfxl2, cdfyl2 = self.calibration_material[2].lowE_cdf

        if self.calibration_material[2].highE_gaussian_pdf is not None:
            cdfxh2 = self.calibration_material[2].highE_gaussian_pdf
            cdfyh2 = [0., 0.]
        else:
            cdfxh2, cdfyh2 = self.calibration_material[2].highE_cdf

        if self.calibration_material[3].lowE_gaussian_pdf is not None:
            cdfxl3 = self.calibration_material[3].lowE_gaussian_pdf
            cdfyl3 = [0., 0.]
        else:
            cdfxl3, cdfyl3 = self.calibration_material[3].lowE_cdf

        if self.calibration_material[3].highE_gaussian_pdf is not None:
            cdfxh3 = self.calibration_material[3].highE_gaussian_pdf
            cdfyh3 = [0., 0.]
        else:
            cdfxh3, cdfyh3 = self.calibration_material[3].highE_cdf

        # Init matrices ---------------------------------------------
        matrixl = None
        matrixh = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                matrixl = self.zgroup['matrixl'][...]
                matrixh = self.zgroup['matrixh'][...]
        matrixl = comm.bcast(matrixl, root=0)
        matrixh = comm.bcast(matrixh, root=0)

        # update error values
        rho1, Z1v = self.calibration_material[1]._rhohat_Zn_values()
        rho2, Z2v = self.calibration_material[2]._rhohat_Zn_values()
        rho3, Z3v = self.calibration_material[3]._rhohat_Zn_values()
        for k in range(matrixl.shape[0]):
            CT0, CT1, CT2, CT3, _, _, _, A, B, n = matrixl[k, :-1]
            matrixl[k, -1] = error_value(A, B, n, CT0, CT1, CT2, CT3, rho1, rho2, rho3, Z1v, Z2v, Z3v)
            CT0, CT1, CT2, CT3, _, _, _, A, B, n = matrixh[k, :-1]
            matrixh[k, -1] = error_value(A, B, n, CT0, CT1, CT2, CT3, rho1, rho2, rho3, Z1v, Z2v, Z3v)

        # Get indices for bad values
        indl = np.argwhere(~(matrixl[:, -1]<tol)).flatten() # ~ to get nans
        indh = np.argwhere(~(matrixh[:, -1]<tol)).flatten() # ~ to get nans

        # Split among  process
        indl_sub = indl[slice(mpi_rank, len(indl), mpi_nprocs)]
        indh_sub = indh[slice(mpi_rank, len(indh), mpi_nprocs)]
        sub_matrixl = matrixl[indl_sub, :].copy()
        sub_matrixh = matrixh[indh_sub, :].copy()

        # Fill by broad search
        maxA, maxB, maxn = self.maxA, self.maxB, self.maxn
        argsl = np.array([rho1, rho2, rho3, maxA, maxB, maxn, tol], dtype='f8')
        argsh = np.array([rho1, rho2, rho3, maxA, maxB, maxn, tol], dtype='f8')
        device_index = config.rank_select_gpu()
        if device_index is not None:
            with config._gpus[device_index]:
                dmatrixl = cuda.to_device(sub_matrixl)
                dmatrixh = cuda.to_device(sub_matrixh)
                dZ1v = cuda.to_device(Z1v)
                dZ2v = cuda.to_device(Z2v)
                dZ3v = cuda.to_device(Z3v)
                dargsl = cuda.to_device(argsl)
                dargsh = cuda.to_device(argsh)
                dcdfxl0 = cuda.to_device(cdfxl0)
                dcdfyl0 = cuda.to_device(cdfyl0)
                dcdfxl1 = cuda.to_device(cdfxl1)
                dcdfyl1 = cuda.to_device(cdfyl1)
                dcdfxl2 = cuda.to_device(cdfxl2)
                dcdfyl2 = cuda.to_device(cdfyl2)
                dcdfxl3 = cuda.to_device(cdfxl3)
                dcdfyl3 = cuda.to_device(cdfyl3)
                dcdfxh0 = cuda.to_device(cdfxh0)
                dcdfyh0 = cuda.to_device(cdfyh0)
                dcdfxh1 = cuda.to_device(cdfxh1)
                dcdfyh1 = cuda.to_device(cdfyh1)
                dcdfxh2 = cuda.to_device(cdfxh2)
                dcdfyh2 = cuda.to_device(cdfyh2)
                dcdfxh3 = cuda.to_device(cdfxh3)
                dcdfyh3 = cuda.to_device(cdfyh3)
                threadsperblock = self.threads_per_block
                blockspergrid = int(np.ceil(matrixl.shape[0]/threadsperblock))

                rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid,
                                                         seed=mpi_rank+int(datetime.now().timestamp()*1000))
                coeff_matrix_broad_search_gpu[blockspergrid, threadsperblock](
                    rng_states, dmatrixl, dZ1v, dZ2v, dZ3v, dargsl, dcdfxl0, dcdfyl0, dcdfxl1, dcdfyl1, dcdfxl2, dcdfyl2, dcdfxl3, dcdfyl3)
                sub_matrixl = dmatrixl.copy_to_host()

                rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid, seed=mpi_rank+int(datetime.now().timestamp()*1000))
                coeff_matrix_broad_search_gpu[blockspergrid, threadsperblock](
                    rng_states, dmatrixh, dZ1v, dZ2v, dZ3v, dargsh, dcdfxh0, dcdfyh0, dcdfxh1, dcdfyh1, dcdfxh2, dcdfyh2, dcdfxh3, dcdfyh3)
                sub_matrixh = dmatrixh.copy_to_host()
        else:
            fill_coeff_matrix_cpu(sub_matrixl, Z1v, Z2v, Z3v, argsl, cdfxl0, cdfyl0, cdfxl1, cdfyl1, cdfxl2, cdfyl2, cdfxl3, cdfyl3)
            fill_coeff_matrix_cpu(sub_matrixh, Z1v, Z2v, Z3v, argsh, cdfxh0, cdfyh0, cdfxh1, cdfyh1, cdfxh2, cdfyh2, cdfxh3, cdfyh3)
        comm.barrier()

        #Collect results
        matrixl[indl_sub, :] = sub_matrixl.copy()
        matrixh[indh_sub, :] = sub_matrixh.copy()
        for k in range(1, mpi_nprocs):
            if mpi_rank == k:
                comm.send(obj=np.ascontiguousarray(indl_sub), dest=0, tag=k)
                comm.send(obj=np.ascontiguousarray(indh_sub), dest=0, tag=k+mpi_nprocs)
                comm.send(obj=np.ascontiguousarray(sub_matrixl), dest=0, tag=k+2*mpi_nprocs)
                comm.send(obj=np.ascontiguousarray(sub_matrixh), dest=0, tag=k+3*mpi_nprocs)
            if mpi_rank == 0:
                indl_sub = comm.recv(None, source=k, tag=k)
                indh_sub = comm.recv(None, source=k, tag=k+mpi_nprocs)
                sub_matrixl = comm.recv(None, source=k, tag=k+2*mpi_nprocs)
                sub_matrixh = comm.recv(None, source=k, tag=k+3*mpi_nprocs)
                matrixl[indl_sub, :] = sub_matrixl.copy()
                matrixh[indh_sub, :] = sub_matrixh.copy()
            comm.barrier()
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                self.zgroup['matrixl'][...] = matrixl
                self.zgroup['matrixh'][...] = matrixh


    def _purge_coefficients(self):
        """
        Remove outliers from the calibration coefficient matrices.
        """
        tol = self.tol

        # Minimum and maximum for atomic numbers
        min_max_Z = np.zeros((3, 2), dtype='f8')
        for k in range(1, 4):
            temp = self.calibration_material[k]._rhohat_Zn_values()[1][:, 0]
            min_max_Z[k-1][0] = min(temp)
            min_max_Z[k-1][1] = max(temp)
        total = [0, 0]
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                for l, label in enumerate(('matrixl', 'matrixh')):
                    matrix = self.zgroup[label][...]
                    matrix = matrix[matrix[:, -1]<tol, :]
                    not_valid = set()
                    not_valid = not_valid.union(set(np.argwhere(matrix[:, 4] < min_max_Z[0, 0]).flatten())) # Remove Z1<Z1min
                    not_valid = not_valid.union(set(np.argwhere(matrix[:, 4] > min_max_Z[0, 1]).flatten())) # Remove Z1>Z1max
                    not_valid = not_valid.union(set(np.argwhere(matrix[:, 5] < min_max_Z[1, 0]).flatten())) # Remove Z2<Z2min
                    not_valid = not_valid.union(set(np.argwhere(matrix[:, 5] > min_max_Z[1, 1]).flatten())) # Remove Z2>Z2max
                    not_valid = not_valid.union(set(np.argwhere(matrix[:, 6] < min_max_Z[2, 0]).flatten())) # Remove Z3<Z3min
                    not_valid = not_valid.union(set(np.argwhere(matrix[:, 6] > min_max_Z[2, 1]).flatten())) # Remove Z3>Z3max
                    valid = sorted(set(range(matrix.shape[0])) - not_valid)
                    matrix = matrix[valid, :]
                    total[l] = matrix.shape[0]
                    self.zgroup[label][:total[l], :] = matrix
                    self.zgroup[label][total[l]:, :] = self.zgroup[label][total[l]:, :]*0+1
        total = comm.bcast(total, root=0)
        hexdigest = hash_coefficient_matrices(self)
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                self.zgroup['matrixl'].attrs['md5sum'] = hexdigest
                self.zgroup['matrixh'].attrs['md5sum'] = hexdigest
        comm.barrier()
        return total


    def _calc_rho_Z(self):
        """
        Calculate electron density (rho) and effective atomic number (Z) distributions
        using Monte Carlo simulations for each voxel in the CT images. This is an internal
        method and should not be called directly by users.
        """
        matrixl = None
        matrixh = None
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                matrixl = self.zgroup['matrixl'][...].astype('f8')
                matrixh = self.zgroup['matrixh'][...].astype('f8')
        matrixl = comm.bcast(matrixl, root=0)
        matrixh = comm.bcast(matrixh, root=0)

        tol = self.tol
        required_iterations = self.required_iterations
        whis = self.whis

        #get CT cutoff as percentile 2.5
        x, y = self.calibration_material[0].lowE_cdf
        ind = np.argwhere(y<0.025).flatten()[-1]
        lowE_CT_cutoff = x[ind]
        x, y = self.calibration_material[0].highE_cdf
        ind = np.argwhere(y<0.025).flatten()[-1]
        highE_CT_cutoff = x[ind]

        rho1, _ = self.calibration_material[1]._rhohat_Zn_values()
        rho2, _ = self.calibration_material[2]._rhohat_Zn_values()
        rho3, _ = self.calibration_material[3]._rhohat_Zn_values()
        nchunks = self.lowECT.nchunks

        device_index = config.rank_select_gpu()
        if device_index is not None:
            with config._gpus[device_index]:
                darray_rho = cuda.to_device(np.zeros(required_iterations, dtype='f8'))
                darray_Z = cuda.to_device(np.zeros(required_iterations, dtype='f8'))
                darray_error = cuda.to_device(np.zeros(required_iterations, dtype='f8'))
                dmatrixl = cuda.to_device(matrixl)
                dmatrixh = cuda.to_device(matrixh)
            threadsperblock = self.threads_per_block
            blockspergrid = int(np.ceil(required_iterations/threadsperblock))

        #Count total number of voxels to be processed
        voxels_needed = np.zeros(nchunks, dtype='u8')
        for block_id in rvtqdm(range(nchunks), desc='Counting voxels', unit='chunk'):
            chunk_indices = self.lowECT.chunk_slice_indices(block_id)
            if block_id % mpi_nprocs == mpi_rank:
                lowECT = self.lowECT[chunk_indices]
                highECT = self.highECT[chunk_indices]
                if self.mask is not None:
                    mask = self.mask[chunk_indices]
                else:
                    mask = np.zeros_like(lowECT, dtype=bool)
                rho_min = self.zgroup['rho_min'][chunk_indices]
                rho_p25 = self.zgroup['rho_p25'][chunk_indices]
                rho_p50 = self.zgroup['rho_p50'][chunk_indices]
                rho_p75 = self.zgroup['rho_p75'][chunk_indices]
                rho_max = self.zgroup['rho_max'][chunk_indices]
                Z_min = self.zgroup['Z_min'][chunk_indices]
                Z_p25 = self.zgroup['Z_p25'][chunk_indices]
                Z_p50 = self.zgroup['Z_p50'][chunk_indices]
                Z_p75 = self.zgroup['Z_p75'][chunk_indices]
                Z_max = self.zgroup['Z_max'][chunk_indices]
                valid = self.zgroup['valid'][chunk_indices]

                index = np.zeros_like(valid, dtype=int)-1
                make_index(index, lowECT, highECT, mask, valid, required_iterations,
                           lowE_CT_cutoff, highE_CT_cutoff, mpi_nprocs)
                voxels_needed[block_id] = len(np.argwhere(index>=0))
        voxels_needed = comm.allreduce(voxels_needed, op=MPI.SUM)

        #Process all
        totalvoxels = np.sum(voxels_needed)
        bar = rvtqdm(total=totalvoxels, unit='voxel')
        datetimestr = datetimenow()
        for block_id in range(nchunks):
            bar.set_description(f"{datetimestr} rho/Z inversion (chunk {block_id+1}/{nchunks})")
            chunk_indices = self.lowECT.chunk_slice_indices(block_id)
            if mpi_rank == 0:
                lowECT = self.lowECT[chunk_indices].copy()
                highECT = self.highECT[chunk_indices].copy()
                if self.mask is not None:
                    mask = self.mask[chunk_indices].copy()
                else:
                    mask = np.zeros_like(lowECT, dtype=bool)
                rho_min = self.zgroup['rho_min'][chunk_indices].copy()
                rho_p25 = self.zgroup['rho_p25'][chunk_indices].copy()
                rho_p50 = self.zgroup['rho_p50'][chunk_indices].copy()
                rho_p75 = self.zgroup['rho_p75'][chunk_indices].copy()
                rho_max = self.zgroup['rho_max'][chunk_indices].copy()
                Z_min = self.zgroup['Z_min'][chunk_indices].copy()
                Z_p25 = self.zgroup['Z_p25'][chunk_indices].copy()
                Z_p50 = self.zgroup['Z_p50'][chunk_indices].copy()
                Z_p75 = self.zgroup['Z_p75'][chunk_indices].copy()
                Z_max = self.zgroup['Z_max'][chunk_indices].copy()
                valid = self.zgroup['valid'][chunk_indices].copy()
            else:
                lowECT = None
                highECT = None
                mask = None
                rho_min = None
                rho_p25 = None
                rho_p50 = None
                rho_p75 = None
                rho_max = None
                Z_min = None
                Z_p25 = None
                Z_p50 = None
                Z_p75 = None
                Z_max = None
                valid = None
            lowECT = comm.bcast(lowECT, root=0)
            highECT = comm.bcast(highECT, root=0)
            mask = comm.bcast(mask, root=0)
            rho_min = comm.bcast(rho_min, root=0)
            rho_p25 = comm.bcast(rho_p25, root=0)
            rho_p50 = comm.bcast(rho_p50, root=0)
            rho_p75 = comm.bcast(rho_p75, root=0)
            rho_max = comm.bcast(rho_max, root=0)
            Z_min = comm.bcast(Z_min, root=0)
            Z_p25 = comm.bcast(Z_p25, root=0)
            Z_p50 = comm.bcast(Z_p50, root=0)
            Z_p75 = comm.bcast(Z_p75, root=0)
            Z_max = comm.bcast(Z_max, root=0)
            valid = comm.bcast(valid, root=0)

            index = np.zeros_like(valid, dtype=int)-1
            make_index(index, lowECT, highECT, mask, valid, required_iterations,
                        lowE_CT_cutoff, highE_CT_cutoff, mpi_nprocs)
            index = comm.bcast(index, root=0)
            totalvoxels = len(np.argwhere(index>=0))
            if totalvoxels == 0:
                continue

            nx, ny, nz = lowECT.shape
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if index[i, j, k] < 0:
                            continue
                        bar.update(1)
                        if index[i, j, k] != mpi_rank:
                            rho_min[i, j, k] = 0.0
                            rho_p25[i, j, k] = 0.0
                            rho_p50[i, j, k] = 0.0
                            rho_p75[i, j, k] = 0.0
                            rho_max[i, j, k] = 0.0
                            Z_min[i, j, k] = 0.0
                            Z_p25[i, j, k] = 0.0
                            Z_p50[i, j, k] = 0.0
                            Z_p75[i, j, k] = 0.0
                            Z_max[i, j, k] = 0.0
                            valid[i, j, k] = 0
                            continue

                        CTl = np.float64(lowECT[i, j, k])
                        CTh = np.float64(highECT[i, j, k])

                        if device_index is not None:
                            with config._gpus[device_index]:
                                reset_arrays_gpu[blockspergrid, threadsperblock](darray_rho, darray_Z, darray_error)
                                rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid, seed=mpi_rank+int(datetime.now().timestamp()*1000))
                                calc_rhoZ_arrays_gpu[blockspergrid, threadsperblock](
                                    darray_rho, darray_Z, darray_error,
                                    dmatrixl, dmatrixh, rng_states,
                                    CTl, CTh, rho1, rho2, rho3,
                                    required_iterations, tol)
                                array_rho = darray_rho.copy_to_host()
                                array_Z = darray_Z.copy_to_host()
                                array_error = darray_error.copy_to_host()
                        else:
                            array_rho, array_Z, array_error = calc_rhoZ_arrays_cpu(required_iterations, matrixl, matrixh, CTl, CTh, rho1, rho2, rho3, tol)

                        ind = array_error<tol
                        ind = np.logical_and(ind, ~np.isnan(array_error))
                        ind = np.logical_and(ind, ~np.isnan(array_Z))
                        ind = np.argwhere(ind).flatten()
                        if len(ind) == 0:
                            continue

                        valid[i, j, k] = len(ind)
                        array_rho = array_rho[ind]
                        array_Z = array_Z[ind]

                        Q1, Q2, Q3 = np.percentile(array_rho, [25, 50, 75])
                        whis1 = min(array_rho[array_rho>=(Q1-whis*(Q3-Q1))])
                        whis2 = max(array_rho[array_rho<=(Q3+whis*(Q3-Q1))])
                        rho_min[i, j, k] = whis1
                        rho_p25[i, j, k] = Q1
                        rho_p50[i, j, k] = Q2
                        rho_p75[i, j, k] = Q3
                        rho_max[i, j, k] = whis2

                        Q1, Q2, Q3 = np.percentile(array_Z, [25, 50, 75])
                        whis1 = min(array_Z[array_Z>=(Q1-whis*(Q3-Q1))])
                        whis2 = max(array_Z[array_Z<=(Q3+whis*(Q3-Q1))])
                        Z_min[i, j, k] = whis1
                        Z_p25[i, j, k] = Q1
                        Z_p50[i, j, k] = Q2
                        Z_p75[i, j, k] = Q3
                        Z_max[i, j, k] = whis2
            comm.barrier()
            rho_min = comm.reduce(rho_min, root=0, op=MPI.SUM)
            rho_p25 = comm.reduce(rho_p25, root=0, op=MPI.SUM)
            rho_p50 = comm.reduce(rho_p50, root=0, op=MPI.SUM)
            rho_p75 = comm.reduce(rho_p75, root=0, op=MPI.SUM)
            rho_max = comm.reduce(rho_max, root=0, op=MPI.SUM)
            Z_min = comm.reduce(Z_min, root=0, op=MPI.SUM)
            Z_p25 = comm.reduce(Z_p25, root=0, op=MPI.SUM)
            Z_p50 = comm.reduce(Z_p50, root=0, op=MPI.SUM)
            Z_p75 = comm.reduce(Z_p75, root=0, op=MPI.SUM)
            Z_max = comm.reduce(Z_max, root=0, op=MPI.SUM)
            valid = comm.reduce(valid, root=0, op=MPI.SUM)
            if mpi_rank == 0:
                self.zgroup['rho_min'][chunk_indices] = rho_min.copy()
                self.zgroup['rho_p25'][chunk_indices] = rho_p25.copy()
                self.zgroup['rho_p50'][chunk_indices] = rho_p50.copy()
                self.zgroup['rho_p75'][chunk_indices] = rho_p75.copy()
                self.zgroup['rho_max'][chunk_indices] = rho_max.copy()
                self.zgroup['Z_min'][chunk_indices] = Z_min.copy()
                self.zgroup['Z_p25'][chunk_indices] = Z_p25.copy()
                self.zgroup['Z_p50'][chunk_indices] = Z_p50.copy()
                self.zgroup['Z_p75'][chunk_indices] = Z_p75.copy()
                self.zgroup['Z_max'][chunk_indices] = Z_max.copy()
                self.zgroup['valid'][chunk_indices] = valid.copy()
            comm.barrier()
        bar.close()


    def check(self, *, verbose=True):
        """
        Check the group structure for consistency and dependencies among arrays
        and processing parameters. The method uses hash functions to detect changes
        in input data or intermediate results that might require reprocessing.

        Parameters
        ----------
        verbose : bool, optional
            If True (default), print detailed status information.
        """
        status = _STATUS.copy()
        if not(self._check_input_data(status)):
            collective_raise(ValueError(str('Group is not ready:\n- ' + '\n- '.join(status))))

        hash_input_data(self)

        # Check Monte Carlo calibration drawings
        if need_coefficient_matrices(self):
            status[8] = '====> calibration coefficient matrices outdated.'
        else:
            status[8] = 'calibration coefficient matrices: ready.'

        # Check Monte Carlo results
        for k, array in enumerate(('rho_min', 'rho_p25', 'rho_p50', 'rho_p75',
                                   'rho_max', 'Z_min', 'Z_p25', 'Z_p50', 'Z_p75',
                                   'Z_max', 'valid')):
            if array not in self.zgroup:
                status[9+k] = f"{array} <not created>"
            elif need_output_array(self, array):
                status[9+k] = f"====> {array}: failed checksum."
            else:
                status[9+k] = f"{array} OK."

        if verbose:
            collective_print("Group check:\n- " + '\n- '.join(status))



    def preprocess(self, restart=False):
        """
        Perform the preprocessing steps for Dual Energy Computed Tomography analysis:

        1. Hash input data (voxel images and calibration material attributes).
        2. Check hash depencies to guarantee simulation integrity.
        3. Generate Monte Carlo calibration coefficient matrices if needed.

        This method uses hash functions to detect changes in input data or intermediate
        results that might require reprocessing.

        Parameters
        ----------
        restart : bool, optional
            If True, force recalculation of all preprocessing steps, ignoring
            existing results. Default is False.
        """

        # Check and hash Input data
        status = _STATUS.copy()
        if not(self._check_input_data(status)):
            collective_raise(ValueError(str(
                'Missing input data:\n- '
                + '\n- '.join(status[:8]))))
        hash_input_data(self)

        # Monte Carlo calibration drawings
        if (not restart and need_coefficient_matrices(self)
            and any(k in self.zgroup for k in ('matrixl', 'matrixh'))):
            collective_raise(Exception(
                'Calibration coefficient matrices are outdated. '
                'Run with restart=True to restart the simulations from scratch.'))

        if not (restart or need_coefficient_matrices(self)):
            collective_print('Calibration coefficient matrices up to date.')
            return

        maximum = self.maximum_iterations
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                _ = self.zgroup.create_array(name='matrixl',
                                                shape=(maximum, 11),
                                                chunks=(maximum, 11),
                                                dtype='f8',
                                                fill_value=1,
                                                overwrite=True,
                                                attributes={'md5sum': ''})
                _ = self.zgroup.create_array(name='matrixh',
                                                shape=(maximum, 11),
                                                chunks=(maximum, 11),
                                                dtype='f8',
                                                fill_value=1,
                                                overwrite=True,
                                                attributes={'md5sum': ''})

        # fill in coefficients for new or incomplete matrices
        bar = rvtqdm(total=2*self.maximum_iterations,
                    desc='Generating inversion coefficients',
                    unit='',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
        current = 0
        for t in range(30):
            comm.barrier()
            self._draw_coefficients()
            new = self._purge_coefficients()
            bar.update(sum(new)-current)
            current = sum(new)
            if sum(new) >= 2*self.maximum_iterations:
                break
        bar.close()
        comm.barrier()
        hexdigest = hash_coefficient_matrices(self)
        with collective_only_rank0_runs():
            if mpi_rank == 0:
                self.zgroup['matrixl'].attrs['md5sum'] = hexdigest
                self.zgroup['matrixh'].attrs['md5sum'] = hexdigest
        comm.barrier()


    def run(self, restart=False):
        """
        Run the DECT analysis on the data in this group.

        This method performs the following steps:
        1. Calls ``preprocess`` to check input data and coefficient matrices.
        2. Checks if output arrays exist and are up-to-date.
        3. Creates or updates output arrays as necessary.
        4. Calculates the electron density (rho) and effective atomic number (Z)
        distributions using Monte Carlo simulations.

        Parameters
        ----------
        restart : bool, optional
            If True, force recalculation of all steps, ignoring existing results.
            Default is False.
        """
        self.preprocess(restart=restart)

        # Check Monte Carlo results
        groups = ('rho_min', 'rho_p25', 'rho_p50', 'rho_p75', 'rho_max',
                  'Z_min', 'Z_p25', 'Z_p50', 'Z_p75', 'Z_max', 'valid')
        exist = [k in self.zgroup for k in groups]
        hexdigest = hash_coefficient_matrices(self)

        create = False
        if restart:
            create = True
        elif all(exist):
            msg = ''
            if mpi_rank == 0:
                for gr in groups:
                    if self.zgroup[gr].attrs['dep_md5sum'] != hexdigest:
                        msg = msg + f"\n - {gr} array failed checksum."
            msg = comm.bcast(msg, root=0)
            if msg:
                collective_raise(Exception('\n'.join((
                    "Output arrays' hashes do not match:"+msg,
                    "Run with restart=True to restart the simulations from scratch."))))
        elif any(exist):
            collective_raise(Exception('Missing some output arrays. Run with restart=True to restart the simulations from scratch.'))
        else:
            create = True

        create = comm.bcast(create, root=0)
        if create:
            for k in rvtqdm(range(len(groups)), desc='Creating output images'):
                gr = groups[k]
                dtype = np.dtype('f8') if k<10 else np.dtype('u4')
                fill_value = 0.0 if k<10 else 0
                image = full_like(
                    self.lowECT,
                    overwrite=True,
                    fill_value=fill_value,
                    dtype=dtype,
                    store=self.zgroup.store,
                    path=gr,
                    field_name=gr,
                    field_unit='g/cc' if k<5 else '')
                with collective_only_rank0_runs():
                    if mpi_rank == 0:
                        image.zarray.attrs['dep_md5sum'] = hexdigest
        comm.barrier()

        self._calc_rho_Z()


    def view_pdfs(self, figsize=(8, 9), bins=30, percentile_interval=(0.1, 99.9), plot_pdfs=True, plot_cdfs=True):
        '''
        Convenience function for visualizing the probability density functions (PDFs) and cumulative
        density functions (CDFs) of calibration materials.

        Parameters
        ----------
        figsize : tuple of float, optional
            Size of the figure in inches, specified as `(width, height)`. Default is `(8, 9)`.

        bins : int, optional
            Number of bins used to create histograms for Monte Carlo PDFs. Default is `30`.

        percentile_interval : tuple of float, optional
            Percentile interval to adjust the x-axis limits, specified as `(percentile_min, percentile_max)`.
            Default is `(0.1, 99.9)`.

        plot_pdfs : bool, optional
            If `True`, the PDFs will be plotted on the graphs. Default is `True`.

        plot_cdfs : bool, optional
            If `True`, the CDFs will be plotted on the graphs. Default is `True`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `Figure` object containing the generated plots.

        axes : dict
            A dictionary containing the plot axes:
            - `'ax_pdf'`: The axes for the PDFs.
            - `'ax_cdf'`: The axes for the CDFs.
            - `'ax_legend'`: The axis containing the legend.

        Examples
        --------
        To visualize the PDFs and CDFs of the calibration materials:

        >>> dectgroup.view_pdfs()

        To customize the number of bins and disable the CDF plots:

        >>> dectgroup.view_pdfs(bins=50, plot_cdfs=False)
        '''

        def gaussfunc(m, s, x):
            return 1/np.sqrt(2*np.pi*s*s)*np.exp(-0.5*(((x-m)/s)**2))

        def cumgauss(m, s, x):
            return 0.5*(1+erf((x-m)/s/np.sqrt(2)))

        fig = plt.figure(figsize=figsize, layout='tight')
        gs = GridSpec(5, 2, fig, height_ratios=[0.2, 1, 1, 1, 1])
        ax_legend = fig.add_subplot(gs[0, :])
        ax_legend.set_axis_off()
        ax_pdf = np.empty((4, 2), dtype='object')
        ax_cdf = np.empty((4, 2), dtype='object')
        for i in range(4):
            for j in range(2):
                ax_pdf[i, j] = fig.add_subplot(gs[i+1, j])
                ax_cdf[i, j] = ax_pdf[i, j].twinx()

        mc_hist_plot = None
        mc_cum_hist_plot = None
        array_plot = None
        model_plot = None
        cum_model_plot = None

        for k in range(4):
            ax_pdf[k, 0].set_xlabel(f'Low E {self.calibration_material[k-1].description}')
            ax_pdf[k, 1].set_xlabel(f'High E {self.calibration_material[k-1].description}')
            if plot_pdfs:
                ax_pdf[k, 0].set_ylabel(f'PDF')
                ax_pdf[k, 1].set_ylabel(f'PDF')
            if plot_cdfs:
                ax_cdf[k, 0].set_ylabel(f'CDF')
                ax_cdf[k, 1].set_ylabel(f'CDF')
            for E in range(2):
                if E == 0:
                    pdfxy = self.calibration_material[k].lowE_pdf
                    cdfxy = self.calibration_material[k].lowE_cdf
                    gauss = self.calibration_material[k].lowE_gaussian_pdf
                    coefs = self.lowE_inversion_coefficients
                else:
                    pdfxy = self.calibration_material[k].highE_pdf
                    cdfxy = self.calibration_material[k].highE_cdf
                    gauss = self.calibration_material[k].highE_gaussian_pdf
                    coefs = self.highE_inversion_coefficients

                # Drawings
                if coefs is not None:
                    hy, hx = np.histogram(coefs[f'CT_{k}'], bins=bins, density=True)
                    hz = np.cumsum(hy*(hx[1:]-hx[:-1]))
                    if plot_pdfs:
                        mc_hist_plot = ax_pdf[k, E].hist(coefs[f'CT_{k}'], bins=bins, density=True,
                                                        facecolor='lightsteelblue', edgecolor='slategrey',
                                                        label='MC PDF')[2]
                    if plot_cdfs:
                        mc_cum_hist_plot = ax_cdf[k, E].step(hx[:-1], hz, color='purple', where='post', label='MC CDF')[0]

                pdfx, pdfy = None, None
                cdfx, cdfy = None, None
                if pdfxy is not None:
                    pdfx, pdfy = pdfxy
                    cdfx, cdfy = cdfxy

                # Array
                if gauss is None:
                    if pdfxy is not None:
                        if plot_pdfs:
                            array_plot = ax_pdf[k, E].plot(pdfx, pdfy, color='brown', marker='.', linestyle='', alpha=0.75, label='Array PDF')[0]
                            model_plot = ax_pdf[k, E].plot(pdfx, pdfy, color='brown', marker='', linestyle='-', alpha=0.75, label='Model PDF')[0]
                        if plot_cdfs:
                            cum_model_plot = ax_cdf[k, E].plot(cdfx, cdfy, color='orangered', marker='', linestyle='-', alpha=0.75, label='CDF model')[0]
                else:
                    m, s = gauss
                    x = np.linspace(m-4*s, m+4*s, 500)
                    y = gaussfunc(m, s, x)
                    if pdfxy is not None:
                        #Iterative reweighted least squares to set best amplitude match
                        ypdf = gaussfunc(m, s, pdfx)
                        f = ypdf.dot(pdfy) / pdfy.dot(pdfy)
                        for _ in range(10):
                            misfit = np.abs(f*pdfy-ypdf)
                            max_misfit = np.max(misfit)
                            misfit[misfit<0.01*max_misfit] = 0.01*max_misfit
                            w = 1/misfit
                            newf = (ypdf).dot(pdfy*w) / pdfy.dot(pdfy*w)
                            if abs(newf-f)/f < 1e-6:
                                break
                            f = newf
                        if plot_pdfs:
                            array_plot = ax_pdf[k, E].plot(pdfx, f*pdfy, color='brown', marker='.', linestyle='', alpha=0.75, label='Array PDF')[0]
                            ax_pdf[k, E].plot(pdfx, f*pdfy, color='darkblue', marker='', linestyle='-', alpha=0.25)[0]

                    if plot_pdfs:
                        model_plot = ax_pdf[k, E].plot(x, y, color='orangered', marker='', linestyle='-', label='Model PDF')[0]
                    if plot_cdfs:
                        cum_model_plot = ax_cdf[k, E].plot(x, cumgauss(m, s, x), color='green', marker='', linestyle='-', alpha=0.75, label='Model CDF')[0]

                if gauss is None and pdfxy is None:
                    xmin, xmax = 0, 1
                elif gauss is None:
                    xmin = np.max(cdfx[cdfy<=percentile_interval[0]/100])
                    xmax = np.min(cdfx[cdfy>=percentile_interval[1]/100])
                else:
                    xmin, xmax = m-4*s, m+4*s
                ax_pdf[k, E].set_xlim(xmin, xmax)
                ax_pdf[k, E].set_ylim(0, max(ax_pdf[k, E].get_ylim()))
                ax_cdf[k, E].set_ylim(0, 1.05)

        plots = [array_plot, model_plot, mc_hist_plot, cum_model_plot, mc_cum_hist_plot]
        labels = ['Array PDF', 'Model PDF', 'MC PDF', 'Model CDF', 'MC CDF']
        if plots:
            ax_legend.legend([k for k in plots if k is not None],
                            [v for k, v in zip(plots, labels) if k is not None],
                            loc='center', ncol=5)

        return fig, {'ax_pdf': ax_pdf, 'ax_cdf': ax_cdf, 'ax_legend': ax_legend}


    def view_inversion_coefs(self, figsize=(8, 5), **kwargs):
        '''
        Convenience function for visualizing the probability density functions (PDFs) for the
        Monte Carlo invertion coefficients.

        Parameters
        ----------
        figsize : tuple of float, optional
            Size of the figure in inches, specified as `(width, height)`. Default is `(8, 5)`.

        kwargs :
            Additional keyword arguments to be passed to the underlying Matplotlib hist function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `Figure` object containing the generated plots.

        ax : numpy.ndarray
            The axes for the PDFs.

        '''
        internal_kwargs = dict(bins=30, density=True, facecolor='lightsteelblue', edgecolor='slategrey')
        internal_kwargs.update(**kwargs)
        fig, ax = plt.subplots(2, 3, layout='constrained', figsize=figsize)
        fig.suptitle('Monte Carlo Inversion parameters')

        for i, (coef, mode) in enumerate(zip((self.lowE_inversion_coefficients,
                                              self.highE_inversion_coefficients),
                                              ('low', 'high'))):
            for j, xlb in enumerate(('A', 'B', 'n')):
                ax[i, j].hist(coef[f'{xlb}'], **internal_kwargs)
                ax[i, j].set_xlabel(f'{mode} $E$ ${xlb}$')
                ax[i, j].set_ylabel('PDF')
        return fig, ax


    def view_inversion_Zeff(self, figsize=(8, 5), **kwargs):
        '''
        Convenience function for visualizing the probability density functions (PDFs) for the
        Monte Carlo resulting effective atomic number for the calibration materials.

        Parameters
        ----------
        figsize : tuple of float, optional
            Size of the figure in inches, specified as `(width, height)`. Default is `(8, 5)`.

        kwargs :
            Additional keyword arguments to be passed to the underlying Matplotlib hist function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `Figure` object containing the generated plots.

        ax : numpy.ndarray
            The axes for the PDFs.

        '''
        internal_kwargs = dict(bins=30, density=True, facecolor='lightsteelblue', edgecolor='slategrey')
        internal_kwargs.update(**kwargs)
        fig, ax = plt.subplots(2, 3, layout='constrained', figsize=figsize)
        fig.suptitle('Monte Carlo effective atomic numbers')

        for i, (coef, mode) in enumerate(zip((self.lowE_inversion_coefficients,
                                              self.highE_inversion_coefficients),
                                              ('low', 'high'))):
            for j, xlb in enumerate(('Z_1', 'Z_2', 'Z_3')):
                ax[i, j].hist(coef[f'{xlb}'], **internal_kwargs)
                ax[i, j].set_xlabel(f'{mode} $E$ $Z_{{eff}}$')
                ax[i, j].set_title(f'{self.calibration_material[j+1].description}')
                ax[i, j].set_ylabel('PDF')
        return fig, ax


def create_group(store, *, path=None, overwrite=False, **kwargs):
    """
    .. _rockverse_dect_create_group:

    Create a new Dual Energy Computed Tomography (DECT) group in a specified Zarr store.

    Parameters
    ----------
    store : str
        The path to the Zarr store where the group will be created. This has to be local
        store or a path in the local file system.
    path : str, optional
        The path within the Zarr store where the new group will be created. If not provided,
        the group will be created at the root level of the store.
    overwrite : bool
        If True, any existing data at the specified path will be deleted before creating
        the new group. Default is False.
    **kwargs
        Additional keyword arguments to be passed to the underlying
        `Zarr group creation funtion <https://zarr.readthedocs.io/en/stable/api/zarr/index.html#zarr.create_group>`_.

    Returns
    -------
    DECTGroup
        An instance of the `DECTGroup` class representing the newly created group.

    Examples
    --------
    To create a new DECT group:

        >>> import rockverse as rv
        >>> dectgroup = rv.dect.create_group('/path/to/dect/store', path='group1', overwrite=True)

    """

    _assert.zarr_localstore('store', store)
    if path is not None:
        _assert.instance('path', path, 'string', (str,))
    _assert.boolean('overwrite', overwrite)
    kwargs['store'] = store
    kwargs['path'] = path
    kwargs['zarr_format'] = 3
    kwargs['overwrite'] = overwrite
    kwargs['attributes'] = {
        '_ROCKVERSE_DATATYPE': 'DECTGroup',
        'tol': 1e-12,
        'whis': 1.5,
        'required_iterations': 5000,
        'maximum_iterations': 50000,
        'maxA': 2000,
        'maxB': 1500,
        'maxn': 30,
        'threads_per_block': 4,
        'calibration_material': {
            "0": {
                'description': None,
                },
            "1": {
                'description': None,
                'composition': None,
                'bulk_density': None,
                },
            "2": {
                'description': None,
                'composition': None,
                'bulk_density': None,
                },
            "3": {
                'description': None,
                'composition': None,
                'bulk_density': None,
                }
            },
            'ZM_table': copy.deepcopy(ATOMIC_NUMBER_AND_MASS_DICT)
        }
    with collective_only_rank0_runs():
        if mpi_rank == 0:
            with zarr.config.set({'array.order': 'C'}):
                _ = zarr.create_group(**kwargs)
    with zarr.config.set({'array.order': 'C'}):
        #Open in sequence to concurrent reading
        for k in range(mpi_nprocs):
            if k == mpi_rank:
                z = zarr.open_group(store=kwargs['store'], path=kwargs['path'], mode='r+')
            comm.barrier()
    return DECTGroup(z)

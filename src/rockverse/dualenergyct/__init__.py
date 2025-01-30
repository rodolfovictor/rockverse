"""
The ``rockverse.dualenergyct`` module provides classes and functions for managing
Dual Energy Computed Tomography (DECT) Monte Carlo processing. This module
is optimized for parallel computation across multiple CPUs or
GPUs using MPI (Message Passing Interface).

For details about the methods used in this module, please see reference:
    1. Original research paper: http://dx.doi.org/10.1002/2017JB014408

Classes:
    - :class:`rockverse.dualenergyct.DualEnergyCTGroup`: Main class for managing DECT processing,
      including data import, calibration, and Monte Carlo simulations.
    - :class:`PeriodicTable`: Manages periodic table information, allowing for the retrieval
      and modification of atomic numbers and masses.
    - :class:`CalibrationMaterial`: Handles calibration material information, including
      description, bulk density, and chemical composition.

.. versionadded:: 0.3.0
    Initial release of the `rockverse.dualenergyct` module.

.. todo:
    - Get processing parameters from rcparams
    - Print gaussian coeficients by material name
    - Enforce directory store
"""

import os
import copy
import numpy as np
import pandas as pd
import zarr
from datetime import datetime
from rockverse._utils import rvtqdm
from rockverse.config import config

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

from rockverse.voxel_image import (
    VoxelImage,
    from_array,
    full_like)

from rockverse.voxel_image.histogram import Histogram
from rockverse.optimize import gaussian_fit
from rockverse import _assert
from rockverse._utils import collective_print
from rockverse.dualenergyct._periodic_table import ATOMIC_NUMBER_AND_MASS_DICT
from rockverse.dualenergyct._gpu_functions import (
    _coeff_matrix_broad_search_gpu,
    _reset_arrays_gpu,
    _calc_rhoZ_arrays_gpu
    )
from rockverse.dualenergyct._corefunctions import _make_index
from rockverse.dualenergyct._cpu_functions import (
    _fill_coeff_matrix_cpu,
    _calc_rhoZ_arrays_cpu
    )
from rockverse.dualenergyct._hash_functions import (
    hash_input_data,
    hash_histogram,
    need_histogram,
    hash_calibration_gaussian_coefficients,
    need_calibration_gaussian_coefficients,
    need_coefficient_matrices,
    hash_coefficient_matrices,
    hash_pre_process,
    need_output_array
    )

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

_STATUS = ['lowECT',
          'highECT',
          'segmentation',
          'mask',
          'histogram_bins',
          'calibration material 0: --- NOT CHECKED ---',
          'calibration material 1: --- NOT CHECKED ---',
          'calibration material 2: --- NOT CHECKED ---',
          'calibration material 3: --- NOT CHECKED ---',
          'lowEhistogram: --- OUTDATED OR NOT SET. ---',
          'highEhistogram: --- OUTDATED OR NOT SET. ---',
          'calibration gaussian coefficients: --- OUTDATED OR NOT SET. ---',
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
        :class:`Group`. It should not be called directly.

    Parameters
    ----------
    zgroup : zarr.hierarchy.Group
        The Zarr group where the main :class:`Group` is stored.

    Examples
    --------

    Create a dual energy CT group. The periodic table will be created using
    default values:

        >>> import drp
        >>> dectgroup = drp.dualenergyct.create_group('/path/to/group/dir')
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
            The Zarr group where the main :class:`Group` is stored.
        """
        self.zgroup = zgroup
        if mpi_rank == 0 and 'ZM_table' not in self.zgroup.attrs:
            self.zgroup.attrs['ZM_table'] = copy.deepcopy(ATOMIC_NUMBER_AND_MASS_DICT)
        comm.barrier()


    def _get_full_table(self):
        """
        Retrieves the dictionary with the full periodic table from the Zarr group.
        """
        ZM_table = None
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
            _assert.collective_raise(KeyError(
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
            _assert.collective_raise(KeyError(f"Element {element} not found in database."))
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
            _assert.collective_raise(ValueError('Atomic number Z expects positive integer value.'))
        if prop == 'M' and (not isinstance(value, (int, float)) or value<=0):
            _assert.collective_raise(ValueError('Atomic mass M expects positive numeric value.'))
        ZM_table = self._get_full_table()
        if element not in ZM_table:
            _assert.collective_raise(KeyError(
                "New elements must be added using the add_element method."))
        ZM_table[element][prop] = value
        if mpi_rank == 0:
            self.zgroup.attrs['ZM_table'] = copy.deepcopy(ZM_table)
        comm.barrier()


    def add_element(self, name, Z, M):
        """
        Adds a new element to the periodic table.

        Parameters
        ----------
        name : str
            The name of the element.
        Z : int
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
            _assert.collective_raise(ValueError('Element name expects string.'))
        if not isinstance(Z, int):
            _assert.collective_raise(ValueError('Atomic number Z expects integer value.'))
        if not isinstance(M, (int, float)):
            _assert.collective_raise(ValueError('Atomic mass M expects numeric value.'))
        ZM_table = self._get_full_table()
        ZM_table[name] = {'Z': Z, 'M': M}
        if mpi_rank == 0:
            self.zgroup.attrs['ZM_table'] = copy.deepcopy(ZM_table)
        comm.barrier()

    def as_dataframe(self):
        """
        Returns the periodic table as a pandas DataFrame.

        Example
        -------

            >>> import drp
            >>> dectgroup = drp.dualenergyct.create_group('/path/to/group/dir')
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

            >>> import drp
            >>> dectgroup = drp.dualenergyct.create_group('/path/to/group/dir')
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
    Computed Tomography (DECT) processing. This class handles various properties
    such as description, bulk density, and chemical composition of the calibration
    materials.

    .. note::
        This class is designed to be created and managed within the main
        :class:`Group`. It should not be called directly.

    Parameters
    ----------
    zgroup : zarr.hierarchy.Group
        The Zarr group where the main :class:`Group` is stored.
    index : {0, 1, 2, 3}
        The index of the calibration material.
    """

    def _getattrs(self):
        sattrs = None
        if mpi_rank == 0:
            sattrs = self.zgroup.attrs['calibration_materials'][self.index]
        sattrs = comm.bcast(sattrs, root=0)
        return sattrs

    def __init__(self, zgroup, index):
        self.index = index
        self.zgroup = zgroup

    def __getitem__(self, key):
        sattrs = self._getattrs()
        return sattrs[key]

    def __setitem__(self, key, value):
        sattrs = None
        if mpi_rank == 0:
            sattrs = self.zgroup.attrs['calibration_materials'][self.index]
        sattrs = comm.bcast(sattrs, root=0)
        if key not in sattrs:
            _assert.collective_raise(KeyError(f'{key} (new keys are not allowed).'))
        if key == 'description':
            _assert.instance('description', value, 'string', (str,))
        if key == 'bulk_density':
            _assert.instance('bulk_density', value, 'number', (int, float))
        if key == 'segmentation_phase':
            _assert.instance('segmentation_phase', value, 'integer', (int,))
        sattrs[key] = value
        attrs = None
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
        attrs = comm.bcast(attrs, root=0)
        attrs['calibration_materials'][self.index] = sattrs
        if mpi_rank == 0:
            self.zgroup.attrs.update(**attrs)
        comm.barrier()

    def keys(self):
        """
        Get the keys of the calibration material dictionary.
        """
        sattrs = self._getattrs()
        return sattrs.keys()

    def values(self):
        """
        Get the values of the calibration material dictionary.
        """
        sattrs = self._getattrs()
        return sattrs.values()

    def items(self):
        """
        Get the key-value pairs of the calibration material properties.
        """
        sattrs = self._getattrs()
        return sattrs.items()

    def __str__(self):
        str_ = None
        if mpi_rank == 0:
            str_ = self.zgroup.attrs['calibration_materials'][self.index].__str__()
        str_ = comm.bcast(str_, root=0)
        return str_

    def __repr__(self):
        repr_ = None
        if mpi_rank == 0:
            repr_ = self.zgroup.attrs['calibration_materials'][self.index].__repr__()
        repr_ = comm.bcast(repr_, root=0)
        return repr_

    def check(self):
        """
        Checks for missing or incorrect properties in the calibration material.
        """
        msg = ""
        for key, value in self.items():
            if value is None:
                msg = msg + f"\n    - =====> Missing {key}."
        if msg:
            return f"Calibration material {self.index}:" + msg
        return msg


    def _rhohat_Zn_values(self):
        """
        Calculate the electron density (rhohat) and atomic number values for the calibration material.
        """
        atomic_number_and_mass = self.zgroup.attrs['ZM_table']
        composition = self['composition']
        missing = [k for k in composition.keys() if k not in atomic_number_and_mass]
        if len(missing) == 1:
            _assert.collective_raise(Exception(
                f"Element {missing[0]} not found in database."))
        elif len(missing) > 1:
            _assert.collective_raise(Exception(
                f"Elements ({', '.join(missing)}) not found in element database."))

        # Z, M, quantity
        values = np.zeros((len(composition), 3), dtype='f8')
        for i, (k, v) in enumerate(composition.items()):
            values[i][0] = atomic_number_and_mass[k]['Z']
            values[i][1] = atomic_number_and_mass[k]['M']
            values[i][2] = v
        sumZ = np.sum(values[:, 2]*values[:, 0])
        sumM = np.sum(values[:, 2]*values[:, 1])
        rhohat = np.float64(self['bulk_density']*2.0*sumZ/sumM)
        return rhohat, values



class DualEnergyCTGroup():
    """
    :bdg-info:`Parallel`
    :bdg-info:`CPU`
    :bdg-info:`GPU`

    Manages Dual Energy Computed Tomography (DECT) processing.
    The workflow is described in the
    `original research paper <http://dx.doi.org/10.1002/2017JB014408>`_.
    This class builds upon
    `Zarr groups <https://zarr.readthedocs.io/en/stable/api/hierarchy.html>`_
    and is adapted for MPI (Message Passing Interface) processing,
    allowing for parallel computation across multiple CPUs or GPUs.

    Parameters
    ----------
    store : str
        Zarr store where the underlying Zarr group will be created.
    overwrite : bool
        If True, deletes all data under ``store`` and creates a new group.
    **kwargs
        Additional keyword arguments for
        `Zarr group creation <https://zarr.readthedocs.io/en/stable/api/hierarchy.html>`_.
    """

    # Process model master-slave commanded by rank 0

    def __init__(self,
                 store,
                 overwrite=False,
                 **kwargs):
        kwargs['overwrite'] = overwrite
        if mpi_rank == 0:
            try:
                z = zarr.hierarchy.open_group(store, 'r')
            except Exception:
                kwargs['overwrite'] = True
            if 'overwrite' in kwargs and kwargs['overwrite']:
                kwargs['cache_attrs'] = False
                z = zarr.group(store=store, **kwargs)
                z.attrs['_ROCKVERSE_DATATYPE'] = 'DualEnergyCTGroup'
                z.attrs['tol'] = 1e-12
                z.attrs['whis'] = 1.5
                z.attrs['required_iterations'] = 5000
                z.attrs['maximum_iterations'] = 50000
                z.attrs['maxA'] = 2000
                z.attrs['maxB'] = 1500
                z.attrs['maxn'] = 30
                z.attrs['threads_per_block'] = 4
                z.attrs['hash_buffer_size'] = 100
                z.attrs['calibration_materials'] = {
                    "lowEHistogram": "",
                    "highEHistogram": "",
                    "0": {
                        'description': None,
                        'segmentation_phase': None,
                        'lowE_gaussian_center_bounds': None,
                        'highE_gaussian_center_bounds': None,
                        },
                    "1": {
                        'description': None,
                        'segmentation_phase': None,
                        'composition': None,
                        'bulk_density': None,
                        'lowE_gaussian_center_bounds': None,
                        'highE_gaussian_center_bounds': None,
                        },
                    "2": {
                        'description': None,
                        'segmentation_phase': None,
                        'composition': None,
                        'bulk_density': None,
                        'lowE_gaussian_center_bounds': None,
                        'highE_gaussian_center_bounds': None,
                        },
                    "3": {
                        'description': None,
                        'segmentation_phase': None,
                        'composition': None,
                        'bulk_density': None,
                        'lowE_gaussian_center_bounds': None,
                        'highE_gaussian_center_bounds': None,
                        }
                }
                self._histogram_bins = 256
        comm.barrier()
        self.zgroup = zarr.open_group(store,
                                      mode='r+',
                                      cache_attrs=False,
                                      )
        self._calibration_material0 = CalibrationMaterial(self.zgroup, '0')
        self._calibration_material1 = CalibrationMaterial(self.zgroup, '1')
        self._calibration_material2 = CalibrationMaterial(self.zgroup, '2')
        self._calibration_material3 = CalibrationMaterial(self.zgroup, '3')
        self._periodic_table = PeriodicTable(self.zgroup)
        self.current_hashes = {'lowE': '',
                               'highE': '',
                               'mask': '',
                               'segmentation': '',}


    # Arrays ----------------------------------------------

    @property
    def lowECT(self):
        """
        The low energy computed tomography voxel image.
        Returns ``None`` if not set. Can only be changed through the
        :ref:`array creation methods <dect_array_creation>`.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/lowECT'):
            return VoxelImage(store=self.zgroup.store, path='/lowECT')
        return None

    @property
    def highECT(self):
        """
        The high energy computed tomography voxel image.
        Returns ``None`` if not set. Can only be changed through the
        :ref:`array creation methods <dect_array_creation>`.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/highECT'):
            return VoxelImage(store=self.zgroup.store, path='/highECT')
        return None

    @property
    def mask(self):
        """
        The mask voxel image. Returns ``None`` if not set. Masked voxels
        will be ignored during the Monte Carlo inversion and histogram calculations.
        Can only be changed through the
        :ref:`array creation methods <dect_array_creation>`.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/mask'):
            return VoxelImage(store=self.zgroup.store, path='/mask')
        return None

    @property
    def segmentation(self):
        """
        The segmentation voxel image. Returns ``None`` if not set. This array
        is used to locate the calibration materials in the image and calculate
        histograms. Can only be changed through the
        :ref:`array creation methods <dect_array_creation>`.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/segmentation'):
            return VoxelImage(store=self.zgroup.store, path='/segmentation')
        return None

    @property
    def rho_min(self):
        """
        Voxel image with the minimum electron density per voxel.
        Minimum value is taken as the lower boxplot whisker boundary
        from the Monte Carlo inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/rho_min'):
            return VoxelImage(store=self.zgroup.store, path='/rho_min')
        return None

    @property
    def rho_p25(self):
        """
        Voxel image with the the first quartile for the electron density
        values per voxel (Q1 or 25th percentile) from the Monte Carlo
        inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/rho_p25'):
            return VoxelImage(store=self.zgroup.store, path='/rho_p25')
        return None

    @property
    def rho_p50(self):
        """
        Voxel image with the the median values for the electron density
        per voxel (Q2 or 50th percentile) from the Monte Carlo
        inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/rho_p50'):
            return VoxelImage(store=self.zgroup.store, path='/rho_p50')
        return None

    @property
    def rho_p75(self):
        """
        Voxel image with the the third quartile for the electron density
        values per voxel (Q3 or 75th percentile) from the Monte Carlo
        inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/rho_p75'):
            return VoxelImage(store=self.zgroup.store, path='/rho_p75')
        return None

    @property
    def rho_max(self):
        """
        Voxel image with the maximum electron density per voxel.
        Maximum value is taken as the upper boxplot whisker boundary
        from the Monte Carlo inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/rho_max'):
            return VoxelImage(store=self.zgroup.store, path='/rho_max')
        return None

    @property
    def Z_min(self):
        """
        Voxel image with the minimum effective atomic number per voxel.
        Minimum value is taken as the lower boxplot whisker boundary
        from the Monte Carlo inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/Z_min'):
            return VoxelImage(store=self.zgroup.store, path='/Z_min')
        return None

    @property
    def Z_p25(self):
        """
        Voxel image with the the first quartile for the effective atomic
        number values per voxel (Q1 or 25th percentile) from the Monte
        Carlo inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/Z_p25'):
            return VoxelImage(store=self.zgroup.store, path='/Z_p25')
        return None

    @property
    def Z_p50(self):
        """
        Voxel image with the the median values for the effective atomic
        number per voxel (Q2 or 50th percentile) from the Monte Carlo
        inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/Z_p50'):
            return VoxelImage(store=self.zgroup.store, path='/Z_p50')
        return None

    @property
    def Z_p75(self):
        """
        Voxel image with the the third quartile for the effective atomic
        number values per voxel (Q3 or 75th percentile) from the Monte
        Carlo inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/Z_p75'):
            return VoxelImage(store=self.zgroup.store, path='/Z_p75')
        return None

    @property
    def Z_max(self):
        """
        Voxel image with the maximum effective atomic number per voxel.
        Maximum value is taken as the upper boxplot whisker boundary
        from the Monte Carlo inversion.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/Z_max'):
            return VoxelImage(store=self.zgroup.store, path='/Z_max')
        return None

    @property
    def valid(self):
        """
        Voxel image with the number of valid Monte Carlo results for each voxel.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/valid'):
            return VoxelImage(store=self.zgroup.store, path='/valid')
        return None


    # Calibration materials and periodic table ----------------------

    @property
    def calibration_material0(self):
        """Dictionary-like object with the properties for `calibration_material0`
        (empty region). The following keys must be set:

        * ``'description'`` - string describing `calibration_material0`. Will be
          used in the default plots.

        * ``'segmentation_phase'`` - segmentation phase in segmentation array
          defining the voxels belonging to `calibration_material0`.

        * ``'lowE_gaussian_center_bounds'`` - two-element list with lower and upper
          bounds for the gaussian center in the lowECT image histogram.

        * ``'highE_gaussian_center_bounds'`` - two-element list with lower and upper
          bounds for the gaussian center in the highECT image histogram.
        """
        return self._calibration_material0

    @property
    def calibration_material1(self):
        """Dictionary-like object with the properties for `calibration_material1`.
        The keys are the same as in `calibration_material0`, plus these additional
        ones:

        * ``'bulk_density'`` - bulk density for the calibration material, in g/cc.

        * ``'composition'`` - dictionary defining the chemical composition. Must
          be set as `key: value` pairs where `key` is the element symbol and
          `value` is the proportionate number of atoms of each element.

        Examples:

        Water H\\ :sub:`2`\\ O:

        .. code-block:: python

            path = r'/path/to/my/working/dir/dect'
            dectgroup = rockverse.dualenergyct.create_group(path)
            dectgroup.calibration_material1['description'] = 'Water'
            dectgroup.calibration_material1['bulk_density'] = 1
            dectgroup.calibration_material1['composition'] = {'H': 2, 'O': 1}

        Silica SiO\\ :sub:`2`:

        .. code-block:: python

            path = r'/path/to/my/working/dir/dect.zarr'
            dectgroup = rockverse.dualenergyct.create_group(path)
            dectgroup.calibration_material1['description'] = 'Silica'
            dectgroup.calibration_material1['bulk_density'] = 2.2
            dectgroup.calibration_material1['composition'] = {'Si': 1, 'O': 2}

        Dolomite CaMg(CO\\ :sub:`3`)\\ :sub:`2`:

        .. code-block:: python

            path = r'/path/to/my/working/dir/dect.zarr'
            dectgroup = rockverse.dualenergyct.create_group(path)
            dectgroup.calibration_material1['description'] = 'Dolomite'
            dectgroup.calibration_material1['bulk_density'] = 2.84
            dectgroup.calibration_material1['composition'] = {'Ca': 1, 'Mg': 1, 'C': 2, 'O': 6}

        Teflon (C\\ :sub:`2`\\ F\\ :sub:`4`\\ )\\ :sub:`n`:

        .. code-block:: python

            path = r'/path/to/my/working/dir/dect.zarr'
            dectgroup = rockverse.dualenergyct.create_group(path)
            dectgroup.calibration_material1['description'] = 'Teflon'
            dectgroup.calibration_material1['composition'] = {'C': 2, 'F': 4}
            dectgroup.calibration_material1['bulk_density'] = 2.2
        """
        return self._calibration_material1

    @property
    def calibration_material2(self):
        """Dictionary-like object with the properties for `calibration_material2`.
        See `calibration_material1` for details.
        """
        return self._calibration_material2

    @property
    def calibration_material3(self):
        """Dictionary-like object with the properties for `calibration_material3`.
        See `calibration_material1` for details.
        """
        return self._calibration_material3

    @property
    def periodic_table(self):
        """
        Class with atomic number and atomic mass values to be used in the
        Monte Carlo inversion. See the details in :class:`PeriodicTable` documentation.
        """
        return self._periodic_table

    @property
    def histogram_bins(self):
        '''
        Number of equal-width histogram bins for the CT images.
        Must be a positive integer.
        '''
        bins = None
        if mpi_rank == 0 and 'histogram_bins' in self.zgroup.attrs:
            bins = self.zgroup.attrs['histogram_bins']
        bins = comm.bcast(bins, root=0)
        return bins

    @histogram_bins.setter
    def histogram_bins(self, v):
        _assert.condition.positive_integer('histogram_bins', v)
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['histogram_bins'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()


    @property
    def maxA(self):
        "Maximum value for coefficient A in broad search."
        maxA_ = None
        if mpi_rank == 0:
            maxA_ = self.zgroup.attrs['maxA']
        maxA_ = comm.bcast(maxA_, root=0)
        return maxA_

    @maxA.setter
    def maxA(self, v):
        _assert.instance('maxA', v, 'number', (float, int))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['maxA'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()

    @property
    def maxB(self):
        "Maximum value for coefficient B in broad search."
        maxB_ = None
        if mpi_rank == 0:
            maxB_ = self.zgroup.attrs['maxB']
        maxB_ = comm.bcast(maxB_, root=0)
        return maxB_

    @maxB.setter
    def maxB(self, v):
        _assert.instance('maxB', v, 'number', (float, int))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['maxB'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()

    @property
    def maxn(self):
        "Maximum value for coefficient n in broad search."
        maxn_ = None
        if mpi_rank == 0:
            maxn_ = self.zgroup.attrs['maxn']
        maxn_ = comm.bcast(maxn_, root=0)
        return maxn_

    @maxn.setter
    def maxn(self, v):
        _assert.instance('maxn', v, 'number', (float, int))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['maxn'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()

    @property
    def lowEhistogram(self):
        """
        Histogram count for the low energy image as a Pandas DataFrame.
        Return ``None`` if lowECT is not set.
        """
        hist = 'None'
        if mpi_rank == 0 and 'lowEHistogram' in self.zgroup:
            hist = {'data': self.zgroup['lowEHistogram'][...],
                    'columns': ['bin_centers',
                                *self.zgroup['lowEHistogram'].attrs['columns']]}
        hist = comm.bcast(hist, root=0)
        if hist == 'None':
            return None
        return pd.DataFrame(data=hist['data'], columns=hist['columns'])


    @property
    def highEhistogram(self):
        """
        Histogram count for the high energy image as a Pandas DataFrame.
        Return ``None`` if highECT is not set.
        """
        hist = 'None'
        if mpi_rank == 0 and 'highEHistogram' in self.zgroup:
            hist = {'data': self.zgroup['highEHistogram'][...],
                    'columns': ['bin_centers', *self.zgroup['highEHistogram'].attrs['columns']]}
        hist = comm.bcast(hist, root=0)
        if hist == 'None':
            return None
        return pd.DataFrame(data=hist['data'], columns=hist['columns'])


    @property
    def lowE_inversion_coefficients(self):
        """
        Pandas DataFrame with the realization sets for low energy inversion
        coefficients. Returns ``None`` if not calculated.
        """
        matrix = None
        if mpi_rank == 0:
            if zarr.storage.contains_array(self.zgroup.store, '/matrixl'):
                matrix = self.zgroup['matrixl'][...]
        matrix = comm.bcast(matrix, root=0)
        if matrix is None:
            return None
        columns = ['CT_0', 'CT_1', 'CT_2', 'CT_3', 'Z_1', 'Z_2', 'Z_3', 'A', 'B', 'n', 'err']
        return pd.DataFrame(data=matrix, index=None, columns=columns, dtype='f8', copy=True)

    @property
    def highE_inversion_coefficients(self):
        """
        Pandas DataFrame with the realization sets for high energy inversion
        coefficients. Returns ``None`` if not calculated.
        """
        matrix = None
        if mpi_rank == 0:
            if zarr.storage.contains_array(self.zgroup.store, '/matrixh'):
                matrix = self.zgroup['matrixh'][...]
        matrix = comm.bcast(matrix, root=0)
        if matrix is None:
            return None
        columns = ['CT_0', 'CT_1', 'CT_2', 'CT_3', 'Z_1', 'Z_2', 'Z_3', 'A', 'B', 'n', 'err']
        return pd.DataFrame(data=matrix, index=None, columns=columns, dtype='f8', copy=True)


    # Inversion parameters ------------------------------------------

    @property
    def threads_per_block(self):
        """
        Number of threads per block when processing using GPUs.
        Use it for fine-tuning GPU performance based on your specific
        GPU capabilities. Default value is 4.
        """
        threads_per_block_ = None
        if mpi_rank == 0:
            threads_per_block_ = self.zgroup.attrs['threads_per_block']
        threads_per_block_ = comm.bcast(threads_per_block_, root=0)
        return threads_per_block_


    @threads_per_block.setter
    def threads_per_block(self, v):
        _assert.instance('threads_per_block', v, 'int', (int,))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['threads_per_block'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()


    @property
    def hash_buffer_size(self):
        """
        (Integer) Number of array slices processed at once when calculating hashes.

        The array hashes are not calculated all at once, but updated using a number of slices
        at a time, given by the `hash_buffer_size` parameter. Higher values for `hash_buffer_size`
        will speed up the hashing operation but require more RAM, while lower values require
        less memory but slow down the hashing operation due to the need to access the file
        system more times. Default value of 100 should be a good compromise in most cases.
        """
        hash_buffer_size_ = None
        if mpi_rank == 0:
            hash_buffer_size_ = self.zgroup.attrs['hash_buffer_size']
        hash_buffer_size_ = comm.bcast(hash_buffer_size_, root=0)
        return hash_buffer_size_


    @hash_buffer_size.setter
    def hash_buffer_size(self, v):
        _assert.instance('hash_buffer_size', v, 'int', (int,))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['hash_buffer_size'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()


    @property
    def tol(self):
        """
        Tolerance value for terminating the Newton-Raphson optimizations.
        Default value is 1e-12.
        """
        tol_ = None
        if mpi_rank == 0:
            tol_ = self.zgroup.attrs['tol']
        tol_ = comm.bcast(tol_, root=0)
        return tol_

    @tol.setter
    def tol(self, v):
        _assert.instance('tol', v, 'float', (float,))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['tol'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()

    @property
    def required_iterations(self):
        """
        The required number of valid Monte Carlo iterations for each voxel.
        Default value is 5000.
        """
        required_iterations_ = None
        if mpi_rank == 0:
            required_iterations_ = self.zgroup.attrs['required_iterations']
        required_iterations_ = comm.bcast(required_iterations_, root=0)
        return required_iterations_

    @required_iterations.setter
    def required_iterations(self, v):
        _assert.instance('required_iterations', v, 'integer', (int,))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['required_iterations'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()

    @property
    def maximum_iterations(self):
        """
        The maximum number of trials to get valid Monte Carlo iterations per voxel.
        Default value is 50000. Recommended 10 times the required number of valid
        Monte Carlo iterations.
        """
        maximum_iterations_ = None
        if mpi_rank == 0:
            maximum_iterations_ = self.zgroup.attrs['maximum_iterations']
        maximum_iterations_ = comm.bcast(maximum_iterations_, root=0)
        return maximum_iterations_

    @maximum_iterations.setter
    def maximum_iterations(self, v):
        _assert.instance('maximum_iterations', v, 'integer', (int,))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['maximum_iterations'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()

    @property
    def whis(self):
        """
        The boxplot whisker length for determining Monte Carlo outlier results.
        Minimum values will be at least :math:`Q_1-whis(Q_3-Q_2)` and maximum
        values will be at most :math:`Q_3+whis(Q_3-Q_2)`, where :math:`Q_1`,
        :math:`Q_2`, and :math:`Q_3` are the three quartiles for the Monte Carlo
        results. Default value is 1.5.
        """
        whis_ = None
        if mpi_rank == 0:
            whis_ = self.zgroup.attrs['whis']
        whis_ = comm.bcast(whis_, root=0)
        return whis_

    @whis.setter
    def whis(self, v):
        _assert.instance('whis', v, 'number', (int, float))
        if mpi_rank == 0:
            attrs = self.zgroup.attrs.asdict()
            attrs['whis'] = v
            self.zgroup.attrs.update(attrs)
        comm.barrier()

    @property
    def calibration_gaussian_coefficients(self):
        """
        A Pandas DataFrame with the fitting coefficients ``[A, mu, sigma]``
        for the calibration materials in the low and high energy CT images,
        according to the model

        .. math:: y = Ae^{-\\frac{1}{2}\\bigg(\\frac{x-\\mu}{\\sigma}\\bigg)^2}.
            :label: gaussian
        """
        coeff = None
        if mpi_rank == 0 and 'CalibrationGaussianCoefficients' in self.zgroup:
            coeff = {'data': self.zgroup['CalibrationGaussianCoefficients'][...],
                     'index': self.zgroup['CalibrationGaussianCoefficients'].attrs['index'],
                     'columns': self.zgroup['CalibrationGaussianCoefficients'].attrs['columns']}
        coeff = comm.bcast(coeff, root=0)
        return pd.DataFrame(data=coeff['data'], index=coeff['index'],
                            columns=coeff['columns'])

    @property
    def _calibration_gaussian_coefficient_values(self):
        """
        See calibration_gaussian_coefficients.
        """
        coeff = None
        if mpi_rank == 0 and 'CalibrationGaussianCoefficients' in self.zgroup:
            coeff = self.zgroup['CalibrationGaussianCoefficients'][...]
        coeff = comm.bcast(coeff, root=0)
        return coeff









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
            Additional keyword arguments for the `drp.full_like` function used
            to create the mask array.
        """
        if self.lowECT is None:
            _assert.collective_raise(ValueError("lowECT must be set before creating mask."))
        kwargs['overwrite'] = overwrite
        kwargs['fill_value'] = fill_value
        kwargs['store'] = os.path.join(self.zgroup.store.path, 'mask')
        kwargs['dtype'] = 'b1'
        kwargs['field_name'] = field_name
        kwargs['description'] = description
        _ = full_like(self.lowECT, **kwargs)


    def delete_mask(self):
        """
        Remove the mask array from the structure.
        """
        if zarr.storage.contains_array(self.zgroup.store, '/mask') and mpi_rank == 0:
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
            Additional keyword arguments for the `drp.full_like` function used
            to create the segmentation array.
        """
        if self.lowECT is None:
            _assert.collective_raise(ValueError("lowECT must be set before creating segmentation."))
        kwargs['overwrite'] = overwrite
        kwargs['fill_value'] = fill_value
        kwargs['store'] = os.path.join(self.zgroup.store.path, 'segmentation')
        kwargs['dtype'] = dtype
        kwargs['field_name'] = field_name
        kwargs['description'] = description
        if np.dtype(kwargs['dtype']).kind != 'u':
            _assert.collective_raise(ValueError(
                "segmentation array dtype must be unsigned integer."))
        _ = full_like(self.lowECT, **kwargs)


    def copy_image(self, image, path, **kwargs):
        """
        Copy an existing voxel image into the DECT group.

        Parameters
        ----------
        image : VoxelImage
            The original voxel image to be copied.
        path : {'lowECT', 'highECT', 'mask', 'segmentation'}
            The path within the DECT group where the array will be stored.
        """
        _assert.rockverse_instance(image, 'image', ('VoxelImage',))
        _assert.in_group('path', path, ('lowECT', 'highECT', 'mask', 'segmentation'))
        if path == 'mask' and image.dtype.kind != 'b':
            _assert.collective_raise(ValueError("mask array dtype must be boolean."))
        if path == 'segmentation' and array.dtype.kind != 'u':
            _assert.collective_raise(ValueError("segmentation array dtype must be unsigned integer."))

        kwargs.update(**image.meta_data_as_dict)
        kwargs['store'] = os.path.join(self.zgroup.store.path, path)
        z = from_array(image, **kwargs)


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
            status[2] = "=====> segmentation array is not set."
            complete = False
        else:
            status[2] = "segmentation: " + self.segmentation.__repr__()


        if self.mask is None:
            status[3] = "mask array is not set (optional)."
        else:
            status[3] = "mask: " + self.mask.__repr__()


        if self.histogram_bins is None:
            status[4] = "=====> histogram_bins is not set."
            complete = False
        else:
            status[4] = f"histogram_bins = {self.histogram_bins}"


        string = self.calibration_material0.check()
        if string:
            status[5] = string
            complete = False
        else:
            status[5] = 'Calibration material 0 OK.'

        string = self.calibration_material1.check()
        if string:
            status[6] = string
            complete = False
        else:
            status[6] = 'Calibration material 1 OK.'

        string = self.calibration_material2.check()
        if string:
            status[7] = string
            complete = False
        else:
            status[7] = 'Calibration material 2 OK.'

        string = self.calibration_material3.check()
        if string:
            status[8] = string
            complete = False
        else:
            status[8] = 'Calibration material 3 OK.'

        if not complete:
            return False

        # Check for shape, chunks and dtypes
        msg = ''
        shapes = [self.lowECT.shape, self.highECT.shape, self.segmentation.shape]
        chunks = [self.lowECT.chunks, self.highECT.chunks, self.segmentation.chunks]
        tmp_msg = 'lowECT, highECT, and segmentation'
        if self.mask is not None:
            shapes.append(self.mask.shape)
            chunks.append(self.mask.chunks)
            tmp_msg = 'lowECT, highECT, segmentation, and mask'
        if (any(shapes[k] != shapes[0] for k in range(len(shapes)))
            or any(chunks[k] != chunks[0] for k in range(len(chunks)))):
            msg = msg + "    - " + tmp_msg + " arrays must have same shape and chunk size.\n"
        if self.lowECT.dtype.kind not in 'uif':
            msg = msg + "    - lowECT dtype must be numeric.\n"
        if self.highECT.dtype.kind not in 'uif':
            msg = msg + "    - highECT dtype must be numeric.\n"
        if self.segmentation.dtype.kind != 'u':
            msg = msg + "    - segmentation dtype must be unsigned integer.\n"
        if self.mask is not None and self.mask.dtype.kind != 'b':
            msg = msg + "    - mask dtype must be boolean.\n"

        if msg:
            _assert.collective_raise(Exception('Invalid input arrays:\n'+msg))

        return True


    # Histograms ----------------------------------------------------
    def _calc_histogram(self, ct):
        """
        Calculate histograms for the low and high energy CT data, using the
        segmentation and mask arrays, as well as the specified number of bins.

        The method uses the `lowECT` or `highECT` array (depending on the `ct`
        parameter), applies the `mask` (if available), and segments the data
        according to the `segmentation` array. The histogram is calculated using
        the number of bins specified in `histogram_bins`.

        The resulting histogram is stored in the Zarr group as 'lowEHistogram' or
        'highEHistogram', depending on the `ct` parameter.

        Parameters
        ----------
        ct : {'low', 'high'}
            Specifies whether to calculate the histogram for low or high energy CT data.

        Note
        ----
        This is an internal method and should not be called directly by users.
        """
        _assert.in_group('ct', ct, ('low', 'high'))
        image = self.lowECT if ct == 'low' else self.highECT
        new_hist = Histogram(image=image,
                             bins=self.histogram_bins,
                             region=None,
                             mask=self.mask,
                             segmentation=self.segmentation)
        shape = list(new_hist.count.values.shape)
        shape[1] += 1
        values = np.zeros(shape, dtype='f8')
        values[:, 0] = new_hist.bin_centers
        values[:, 1:] = new_hist.count.values
        columns = [str(k) for k in new_hist.count.columns]
        if mpi_rank == 0:
            _ = self.zgroup.create_dataset(name=ct+'EHistogram',
                                           data=values,
                                           overwrite=True)
            self.zgroup[ct+'EHistogram'].attrs['columns'] = columns
        comm.barrier()
        hexdigest = hash_histogram(self, ct)
        if mpi_rank == 0:
            self.zgroup[ct+'EHistogram'].attrs['md5sum'] = hexdigest
        comm.barrier()


    def _fit_histograms(self):
        """
        Fit Gaussian distributions to the histograms of calibration materials
        for both low and high energy CT data.

        The Gaussian model used is:
            y = A * exp(-0.5 * ((x - mu) / sigma)^2)

        where:
            y is the histogram count
            x is the center of the histogram bin
            A is the amplitude
            mu is the mean
            sigma is the standard deviation

        Results are stored as a 2d array in the Zarr group as
        'CalibrationGaussianCoefficients', with the configuration

                                'A_lowE', 'mu_lowE', 'sigma_lowE', 'A_highE', 'mu_highE', 'sigma_highE'
        Calibration material 0      .         .            .            .         .           .
        Calibration material 1      .         .            .            .         .           .
        Calibration material 2      .         .            .            .         .           .
        Calibration material 3      .         .            .            .         .           .

        Note
        ----
        This is an internal method and should not be called directly by users.
        It assumes that the histograms have already been calculated and stored.
        """
        columns = ['A_lowE', 'mu_lowE', 'sigma_lowE',
                   'A_highE', 'mu_highE', 'sigma_highE']
        index = ['Calibration material 0', 'Calibration material 1',
                 'Calibration material 2', 'Calibration material 3']
        coefs = np.zeros((4, 6), dtype='f8')
        for i, calibration_material in enumerate([self.calibration_material0,
                                                  self.calibration_material1,
                                                  self.calibration_material2,
                                                  self.calibration_material3]):
            phase = str(calibration_material['segmentation_phase'])
            for j, (hist, label) in enumerate(zip((self.lowEhistogram,
                                                   self.highEhistogram),
                                                  ('lowE', 'highE'))):
                x = hist.bin_centers.values
                y = hist[phase].values
                bounds = calibration_material[f'{label}_gaussian_center_bounds']
                try:
                    c = list(gaussian_fit(x, y, center_bounds=bounds))
                    coefs[i, 0+3*j] = c[0]
                    coefs[i, 1+3*j] = c[1]
                    coefs[i, 2+3*j] = c[2]
                except Exception as e:
                    e.add_note(f'On calibration material {i}, {label}')
                    _assert.collective_raise(e)
        coefs = comm.bcast(coefs, root=0)
        if mpi_rank == 0:
            acoefs = self.zgroup.array(name='CalibrationGaussianCoefficients',
                                       data=coefs, dtype='f8', overwrite=True)
            acoefs.attrs['columns'] = columns
            acoefs.attrs['index'] = index
        comm.barrier()
        hexdigest = hash_calibration_gaussian_coefficients(self)
        if mpi_rank == 0:
            acoefs.attrs['md5sum'] = hexdigest
        comm.barrier()




    def _draw_coefficients(self):
        """
        Generate the calibration coefficient sets for low and high energy.
        """
        # Processing parameters
        maximum, tol = self.maximum_iterations, self.tol
        coefs = self.calibration_gaussian_coefficients

        # Init matrices ----------------------------------------------
        matrixl = np.ones((maximum, 11), dtype='f8')
        matrixh = np.ones((maximum, 11), dtype='f8')
        if mpi_rank == 0:
            temp = self.zgroup['matrixl'][...]
            temp = temp[temp[:, -1]<tol, :]
            length = min(matrixl.shape[0], temp.shape[0])
            matrixl[:length, :] = temp[:length, :]
            temp = self.zgroup['matrixh'][...]
            temp = temp[temp[:, -1]<tol, :]
            length = min(matrixh.shape[0], temp.shape[0])
            matrixh[:length, :] = temp[:length, :]
        matrixl = comm.bcast(matrixl, root=0)
        matrixh = comm.bcast(matrixh, root=0)

        # Split and process
        ind = slice(mpi_rank, maximum, mpi_nprocs)
        rho1, Z1v = self.calibration_material1._rhohat_Zn_values()
        rho2, Z2v = self.calibration_material2._rhohat_Zn_values()
        rho3, Z3v = self.calibration_material3._rhohat_Zn_values()
        m0l = coefs.loc['Calibration material 0', 'mu_lowE']
        s0l = coefs.loc['Calibration material 0', 'sigma_lowE']
        m1l = coefs.loc['Calibration material 1', 'mu_lowE']
        s1l = coefs.loc['Calibration material 1', 'sigma_lowE']
        m2l = coefs.loc['Calibration material 2', 'mu_lowE']
        s2l = coefs.loc['Calibration material 2', 'sigma_lowE']
        m3l = coefs.loc['Calibration material 3', 'mu_lowE']
        s3l = coefs.loc['Calibration material 3', 'sigma_lowE']
        m0h = coefs.loc['Calibration material 0', 'mu_highE']
        s0h = coefs.loc['Calibration material 0', 'sigma_highE']
        m1h = coefs.loc['Calibration material 1', 'mu_highE']
        s1h = coefs.loc['Calibration material 1', 'sigma_highE']
        m2h = coefs.loc['Calibration material 2', 'mu_highE']
        s2h = coefs.loc['Calibration material 2', 'sigma_highE']
        m3h = coefs.loc['Calibration material 3', 'mu_highE']
        s3h = coefs.loc['Calibration material 3', 'sigma_highE']
        comm.barrier()

        # Fill broad search
        maxA, maxB, maxn = self.maxA, self.maxB, self.maxn
        matrixl = matrixl[ind, :].copy()
        matrixh = matrixh[ind, :].copy()
        argsl = np.array([m0l, s0l, m1l, s1l, m2l, s2l, m3l, s3l, rho1, rho2, rho3, maxA, maxB, maxn, tol], dtype='f8')
        argsh = np.array([m0h, s0h, m1h, s1h, m2h, s2h, m3h, s3h, rho1, rho2, rho3, maxA, maxB, maxn, tol], dtype='f8')
        device_index = config.rank_select_gpu()
        if device_index is not None:
            with config._gpus[device_index]:
                dmatrixl = cuda.to_device(matrixl)
                dmatrixh = cuda.to_device(matrixh)
                dZ1v = cuda.to_device(Z1v)
                dZ2v = cuda.to_device(Z2v)
                dZ3v = cuda.to_device(Z3v)
                dargsl = cuda.to_device(argsl)
                dargsh = cuda.to_device(argsh)
                threadsperblock = self.threads_per_block
                blockspergrid = int(np.ceil(matrixl.shape[0]/threadsperblock))

                rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid, seed=mpi_rank+int(datetime.now().timestamp()*1000))
                _coeff_matrix_broad_search_gpu[blockspergrid, threadsperblock](
                        rng_states, dmatrixl, dZ1v, dZ2v, dZ3v, dargsl)
                matrixl = dmatrixl.copy_to_host()

                rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid, seed=mpi_rank+int(datetime.now().timestamp()*1000))
                _coeff_matrix_broad_search_gpu[blockspergrid, threadsperblock](
                    rng_states, dmatrixh, dZ1v, dZ2v, dZ3v, dargsh)
                matrixh = dmatrixh.copy_to_host()
        else:
            _fill_coeff_matrix_cpu(matrixl, Z1v, Z2v, Z3v, argsl)
            _fill_coeff_matrix_cpu(matrixh, Z1v, Z2v, Z3v, argsh)
        comm.barrier()
        for k in range(mpi_nprocs):
            if k == mpi_rank:
                self.zgroup['matrixl'][ind, :] = matrixl
                self.zgroup['matrixh'][ind, :] = matrixh
            comm.barrier()
        comm.barrier()



    def _purge_coefficients(self):
        """
        Remove outliers from the calibration coefficient matrices.
        """
        tol, whis = self.tol, self.whis
        total = [0, 0]
        if mpi_rank == 0:
            for l, label in enumerate(('matrixl', 'matrixh')):
                matrix = self.zgroup[label][...]
                matrix = matrix[matrix[:, -1]<tol, :]
                not_valid = set()
                for k in range(11):
                    data = matrix[:, k]
                    Q1, Q3 = np.percentile(data, [25, 75])
                    not_valid = not_valid.union(set(np.argwhere(data<Q1-whis*(Q3-Q1)).flatten()))
                    not_valid = not_valid.union(set(np.argwhere(data>Q3+whis*(Q3-Q1)).flatten()))
                valid = sorted(set(range(matrix.shape[0])) - not_valid)
                matrix = matrix[valid, :]
                total[l] = matrix.shape[0]
                self.zgroup[label][:total[l], :] = matrix
                self.zgroup[label][total[l]:, :] = self.zgroup[label][total[l]:, :]*0+1
        total = comm.bcast(total, root=0)
        hexdigest = hash_coefficient_matrices(self)
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
        if mpi_rank == 0:
            matrixl = self.zgroup['matrixl'][...].astype('f8')
            matrixh = self.zgroup['matrixh'][...].astype('f8')
        matrixl = comm.bcast(matrixl, root=0)
        matrixh = comm.bcast(matrixh, root=0)

        tol = self.tol
        required_iterations = self.required_iterations
        whis = self.whis
        coefs = self.calibration_gaussian_coefficients
        m0l = coefs.loc['Calibration material 0', 'mu_lowE']
        s0l = coefs.loc['Calibration material 0', 'sigma_lowE']
        m0h = coefs.loc['Calibration material 0', 'mu_highE']
        s0h = coefs.loc['Calibration material 0', 'sigma_highE']
        rho1, _ = self.calibration_material1._rhohat_Zn_values()
        rho2, _ = self.calibration_material2._rhohat_Zn_values()
        rho3, _ = self.calibration_material3._rhohat_Zn_values()
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

        for block_id in rvtqdm(range(nchunks), desc='rho/Z inversion', unit='chunk'):
            box, bex, boy, bey, boz, bez = self.lowECT.chunk_slice_indices(block_id)
            if mpi_rank == 0:
                lowECT = self.lowECT[box:bex, boy:bey, boz:bez].copy()
                highECT = self.highECT[box:bex, boy:bey, boz:bez].copy()
                mask = self.mask[box:bex, boy:bey, boz:bez].copy()
                rho_min = self.zgroup['rho_min'][box:bex, boy:bey, boz:bez].copy()
                rho_p25 = self.zgroup['rho_p25'][box:bex, boy:bey, boz:bez].copy()
                rho_p50 = self.zgroup['rho_p50'][box:bex, boy:bey, boz:bez].copy()
                rho_p75 = self.zgroup['rho_p75'][box:bex, boy:bey, boz:bez].copy()
                rho_max = self.zgroup['rho_max'][box:bex, boy:bey, boz:bez].copy()
                Z_min = self.zgroup['Z_min'][box:bex, boy:bey, boz:bez].copy()
                Z_p25 = self.zgroup['Z_p25'][box:bex, boy:bey, boz:bez].copy()
                Z_p50 = self.zgroup['Z_p50'][box:bex, boy:bey, boz:bez].copy()
                Z_p75 = self.zgroup['Z_p75'][box:bex, boy:bey, boz:bez].copy()
                Z_max = self.zgroup['Z_max'][box:bex, boy:bey, boz:bez].copy()
                valid = self.zgroup['valid'][box:bex, boy:bey, boz:bez].copy()
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
            _make_index(index, lowECT, highECT, mask, valid, required_iterations,
                       m0l, s0l, m0h, s0h, mpi_nprocs)
            index = comm.bcast(index, root=0)
            totalvoxels = len(np.argwhere(index>=0))
            if totalvoxels == 0:
                continue
            bar = rvtqdm(total=totalvoxels, position=1, unit='voxel',
                         desc=f'rho/Z inversion chunk {block_id}/{nchunks}')
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
                                _reset_arrays_gpu[blockspergrid, threadsperblock](darray_rho, darray_Z, darray_error)
                                rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid, seed=mpi_rank+int(datetime.now().timestamp()*1000))
                                _calc_rhoZ_arrays_gpu[blockspergrid, threadsperblock](
                                    darray_rho, darray_Z, darray_error,
                                    dmatrixl, dmatrixh, rng_states,
                                    CTl, CTh, rho1, rho2, rho3,
                                    required_iterations, tol)
                                array_rho = darray_rho.copy_to_host()
                                array_Z = darray_Z.copy_to_host()
                                array_error = darray_error.copy_to_host()
                        else:
                            array_rho, array_Z, array_error = _calc_rhoZ_arrays_cpu(required_iterations, matrixl, matrixh, CTl, CTh, m0l, s0l, m0h, s0h, rho1, rho2, rho3, tol)

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
            bar.close()
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
                self.zgroup['rho_min'][box:bex, boy:bey, boz:bez] = rho_min.copy()
                self.zgroup['rho_p25'][box:bex, boy:bey, boz:bez] = rho_p25.copy()
                self.zgroup['rho_p50'][box:bex, boy:bey, boz:bez] = rho_p50.copy()
                self.zgroup['rho_p75'][box:bex, boy:bey, boz:bez] = rho_p75.copy()
                self.zgroup['rho_max'][box:bex, boy:bey, boz:bez] = rho_max.copy()
                self.zgroup['Z_min'][box:bex, boy:bey, boz:bez] = Z_min.copy()
                self.zgroup['Z_p25'][box:bex, boy:bey, boz:bez] = Z_p25.copy()
                self.zgroup['Z_p50'][box:bex, boy:bey, boz:bez] = Z_p50.copy()
                self.zgroup['Z_p75'][box:bex, boy:bey, boz:bez] = Z_p75.copy()
                self.zgroup['Z_max'][box:bex, boy:bey, boz:bez] = Z_max.copy()
                self.zgroup['valid'][box:bex, boy:bey, boz:bez] = valid.copy()
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
        def raise_not_complete(status):
            _assert.collective_raise(ValueError(str(
                'Group is not ready:\n- ' + '\n- '.join(status))))

        status = _STATUS.copy()
        if not(self._check_input_data(status)):
            raise_not_complete(status)

        hash_input_data(self)

        # Histogramas
        needlow = need_histogram(self, 'low')
        if not needlow:
            status[9] = 'lowEhistogram: ready.'
        needhigh = need_histogram(self, 'high')
        if not needhigh:
            status[10] = 'highEhistogram: ready.'
        if needlow or needhigh:
            raise_not_complete(status)

        # Gaussian coefficients
        if need_calibration_gaussian_coefficients(self):
            raise_not_complete(status)
        status[11] = 'calibration gaussian coefficients: ready.'

        # Check Monte Carlo calibration drawings
        if need_coefficient_matrices(self):
            status[12] = '====> calibration coefficient matrices outdated.'
        else:
            status[12] = 'calibration coefficient matrices: ready.'

        # Check Monte Carlo results
        for k, array in enumerate(('rho_min', 'rho_p25', 'rho_p50', 'rho_p75',
                                   'rho_max', 'Z_min', 'Z_p25', 'Z_p50', 'Z_p75',
                                   'Z_max', 'valid')):
            if array not in self.zgroup:
                status[13+k] = "<not created>"
            elif need_output_array(self, array):
                status[13+k] = f"====> {array}: failed checksum."
            else:
                status[13+k] = f"{array} OK."

        if verbose:
            collective_print("Group check:\n- " + '\n- '.join(status))



    def preprocess(self, restart=False):
        """
        Perform preprocessing steps for Dual Energy Computed Tomography analysis:

        1. Check and hash input data for consistency.
        2. Calculate histograms for low and high energy CT data if needed.
        3. Fit Gaussian distributions to calibration material histograms.
        4. Generate Monte Carlo calibration coefficient matrices.

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
            _assert.collective_raise(ValueError(str(
                'Missing input data:\n- '
                + '\n- '.join(status[:8]))))
        hash_input_data(self)


        # Histogramas
        for ct in ('low', 'high'):
            if need_histogram(self, ct) or restart:
                self._calc_histogram(ct)
            else:
                collective_print(f'{ct}Ehistogram up to date.')


        # Gaussian coefficients
        if need_calibration_gaussian_coefficients(self):
            self._fit_histograms()
        collective_print('Gaussian coefficients for calibration histograms:')
        collective_print(self.calibration_gaussian_coefficients.to_string(),
                         print_time=False)


        # Monte Carlo calibration drawings
        if (not restart
            and need_coefficient_matrices(self)
            and any(k in self.zgroup for k in ('matrixl', 'matrixh'))):
            _assert.collective_raise(Exception(
                'Calibration coefficient matrices are outdated. '
                'Run with restart=True to restart the simulations from scratch.'))

        if need_coefficient_matrices(self) or restart:
            maximum = self.maximum_iterations
            if mpi_rank == 0:
                _ = self.zgroup.create_dataset('matrixl', shape=(maximum, 11),
                                               chunks=False, dtype='f8',
                                               fill_value=1, overwrite=True)
                _ = self.zgroup.create_dataset('matrixh', shape=(maximum, 11),
                                               chunks=False, dtype='f8',
                                               fill_value=1, overwrite=True)
            comm.barrier()
            bar = rvtqdm(total=2*self.maximum_iterations,
                         desc='Generating inversion coefficients',
                         unit='',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
            current = 0
            for _ in range(30):
                self._draw_coefficients()
                new = self._purge_coefficients()
                bar.update(sum(new)-current)
                current = sum(new)
                if sum(new) >= 2*self.maximum_iterations:
                    break
            bar.close()
            comm.barrier()
            hexdigest = hash_coefficient_matrices(self)
            if mpi_rank == 0:
                self.zgroup['matrixl'].attrs['md5sum'] = hexdigest
                self.zgroup['matrixh'].attrs['md5sum'] = hexdigest
            comm.barrier()
        else:
            collective_print('Calibration matrices up to date.')



    def run(self, restart=False):
        """
        Run the DECT analysis on the data in this group.

        This method performs the following steps:
        1. Runs preprocessing steps (histograms, Gaussian fitting, coefficient matrices).
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
        hexdigest = hash_pre_process(self)

        create = False
        if restart:
            create = True
        elif all(exist):
            msg = ''
            if mpi_rank == 0:
                for gr in groups:
                    if self.zgroup[gr].attrs['md5sum'] != hexdigest:
                        msg = msg + f"\n - {gr} array failed checksum."
            msg = comm.bcast(msg, root=0)
            if msg:
                _assert.collective_raise(Exception('\n'.join((
                    "Output arrays' hashes do not match:"+msg,
                    "Run with restart=True to restart the simulations from scratch."))))
        elif any(exist):
            _assert.collective_raise(Exception('Missing some output arrays. Run with restart=True to restart the simulations from scratch.'))
        else:
            create = True

        create = comm.bcast(create, root=0)
        if create:
            for k, gr in enumerate(groups):
                dtype = 'f8' if k<10 else 'u4'
                fill_value = 0.0 if k<10 else 0
                if k % mpi_nprocs == mpi_rank:
                    self.zgroup[gr] = zarr.full_like(self.lowECT,
                                                     overwrite=True,
                                                     fill_value=fill_value,
                                                     dtype=dtype)
                    self.zgroup[gr].attrs.update(self.lowECT.attrs.asdict())
                    self.zgroup[gr].attrs['field_name'] = gr
                    self.zgroup[gr].attrs['field_unit'] = 'g/cc' if k<5 else ''
                    self.zgroup[gr].attrs['md5sum'] = hexdigest
        comm.barrier()

        self._calc_rho_Z()



def create_group(*args, **kwargs):
    return DualEnergyCTGroup(*args, **kwargs)

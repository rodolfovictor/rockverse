"""
The seismic data module provides tools for handling seismic data.

It enables efficient storage and manipulation of seismic data in Zarr
format, while also offering methods for accessing inline (ilines),
crossline (xlines), and sample-related information. Visualization
utilities using libraries like Matplotlib are also supported.
"""

import zarr
import segyio
import numpy as np
import pandas as pd
from rockverse._utils import rvtqdm, datetimenow
from rockverse import _assert
from rockverse.errors import collective_raise
import matplotlib.pyplot as plt

#CONVERTER RAISE TO COLLECTIVE RAISE



class SeismicData():
    """
    High-level interface to handle seismic data stored in Zarr format.

    .. note::
       This class should not be instantiated directly. Instead, use the provided
       :ref:`creation functions <seismic data creation functions>`.
    """

    def __init__(self, zgroup):
        _assert.zarr_group('zgroup', zgroup)
        self.zgroup = zgroup

    def _get_index(self, ind, mode):
        """
        Finds the index of the specified inline or crossline.

        Parameters
        ----------
        ind : int
            The inline or crossline number to locate.
        mode : {'ilines', 'xlines'}
            The axis where to locate the number.

        Returns
        -------
        index : int
            The array index

        Raises
        ------
        Exception
            If the inline or crossline number is not found or has multiple matches.
        """
        _assert.condition.integer('ind', ind)
        _assert.in_group('mode', mode, ('ilines', 'xlines'))
        index = np.argwhere(self.zgroup[mode][...]==ind).flatten()
        error = "Iline" if mode=='ilines' else 'Xline'
        if len(index) == 0:
            raise Exception(f"{error} {ind} not found.")
        if len(index) > 1:
            raise Exception("What the heck?")
        return index[0]

    class Iline:
        """
        Provides inline-specific access to seismic data.
        """

        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, ind):
            """
            Returns the seismic data for the specified inline number.

            Parameters
            ----------
            ind : int
                The inline number.

            Returns
            -------
            numpy.ndarray
                The seismic data for the specified inline.
            """
            index = self.parent._get_index(ind, mode='ilines')
            return self.parent.zgroup['data'][index, ...]

    class Xline:
        """
        Provides crossline-specific access to seismic data.
        """

        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, ind):
            """
            Returns the seismic data for the specified crossline number.

            Parameters
            ----------
            ind : int
                The crossline number.

            Returns
            -------
            numpy.ndarray
                The seismic data for the specified crossline.
            """
            index = self.parent._get_index(ind, mode='xlines')
            return self.parent.zgroup['data'][:, index, ...]

    @property
    def iline(self):
        """
        Provides an accessor for inline data.

        Example
        -------

        Retrieve data from inline 256 using a SeismicData object named `seis_data`:

        .. code-block:: python

            iline_data = seis_data.iline[256]

        """
        return self.Iline(self)

    @property
    def xline(self):
        """
        Provides an accessor for xline data.

        Example
        -------

        Retrieve data from xline 1256 using a SeismicData object named `seis_data`:

        .. code-block:: python

            xline_data = seis_data.xline[1256]

        """
        return self.Xline(self)

    @property
    def ilines(self):
        """
        Returns the array of inline numbers.

        Returns
        -------
        numpy.ndarray
            Array of inline numbers.
        """
        return self.zgroup['ilines'][...]

    @property
    def xlines(self):
        """
        Returns the array of crossline numbers.

        Returns
        -------
        numpy.ndarray
            Array of crossline numbers.
        """
        return self.zgroup['xlines'][...]

    @property
    def samples(self):
        """
        Returns the array of sample depths.

        Returns
        -------
        numpy.ndarray
            Array of sample depths.
        """
        return self.zgroup['samples'][...]

    def plot_section(self, ind, mode='iline', **kwargs):
        """
        Plots a section of the seismic data (inline or crossline).

        Parameters
        ----------
        ind : int
            Inline or crossline index to plot.
        mode : {'iline', 'xline'}, optional
            Axis to plot. Default is 'iline'.
        **kwargs : dict
            Additional arguments for Matplotlib's `pcolormesh`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.

        matplotlib.collections.QuadMesh
            The section plot object.

        Raises
        ------
        KeyError
            If the specified index is not found in the data.

        Example
        -------
            Plot iline 5005 in gray colormap:

            .. code-block:: python

                clip = 50000 # Adjust to your data
                seis_data.plot_section(5005, cmap='gray', vmin=-clip, vmax=clip)
        """
        _assert.condition.integer('ind', ind)
        _assert.in_group('mode', mode, ('iline', 'xline'))
        y = self.samples
        if mode == 'iline':
            x = self.xlines
            if ind not in self.ilines:
                collective_raise(KeyError(f'Index {ind} not in data ilines'))
            z = self.iline[ind].T
        else:
            x = self.ilines
            if ind not in self.xlines:
                collective_raise(KeyError(f'Index {ind} not in data xlines'))
            z = self.xline[ind].T
        ax = plt.gca()
        quadmesh = ax.pcolormesh(x, y, z, **kwargs)
        ax.set_title(f'{mode.capitalize()} {ind}')
        ax.set_xlabel('Xline' if mode == 'iline' else 'Iline')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_ylim(ax.get_ylim()[::-1])

        return ax, quadmesh


#GATHERS?
def create(store, ilines, xlines, samples, data_type, data_chunks, overwrite=False):
    '''
    Create empty seismic data.

    #ILINE XLINE MUST BE INTEGER
    '''
    zgroup = zarr.create_group(store=store, overwrite=overwrite)
    zgroup.attrs['_ROCKVERSE_DATATYPE'] = 'SeismicData'
    seis_data = SeismicData(zgroup)

    for name, data in zip(('ilines', 'xlines', 'samples'), (ilines, xlines, samples)):
        seis_data.zgroup.create_array(name,
                                      shape=data.shape,
                                      chunks=data.shape,
                                      dtype=data.dtype,
                                      overwrite=overwrite)
        seis_data.zgroup[name][...] = data

    Nil = len(ilines)
    Nxl = len(xlines)
    Ns = len(samples)

    seis_data.zgroup.create_array('cdp_x_map', shape=(Nil, Nxl), chunks=(Nil, Nxl), dtype='f8', fill_value=np.nan, overwrite=True)
    seis_data.zgroup.create_array('cdp_y_map', shape=(Nil, Nxl), chunks=(Nil, Nxl), dtype='f8', fill_value=np.nan, overwrite=True)
    seis_data.zgroup.create_array('data',
                                  shape=(Nil, Nxl, Ns),
                                  chunks=data_chunks,
                                  dtype=data_type,
                                  overwrite=True)

    return seis_data



#PENSAR EM PARALELIZAR LEITURA DE VÁRIOS ARQUIVOS
#UM RANK LÊ CADA ARQUIVO
#SÓ UM RANK
#OFFSETS?
def import_segy(filename, store,
                iline_byte, xline_byte,
                coordinate_scaling_byte,
                mpi_index=0,
                verbose=True,
                overwrite=False):
    """
    Import segy data into a SeismicData class.

    Uses the `segyio library <https://segyio.readthedocs.io/en/stable/>`_.
    """

    #if mpi_index % mpi_nprocs != mpi_rank:
    #    return

    with segyio.open(filename, "r", iline=iline_byte, xline=xline_byte) as f:
        ilines = f.ilines
        xlines = f.xlines
        offsets = f.offsets
        samples = f.samples
        data_type = f.iline[ilines[0]].dtype
    Nil = len(ilines)
    Nxl = len(xlines)
    Ns = len(samples)

    seis_data = create(store=store,
                       ilines=ilines,
                       xlines=xlines,
                       samples=samples,
                       data_type=data_type,
                       data_chunks=(1, Nxl, Ns),
                       overwrite=overwrite)

    # Populate main data
    with segyio.open(filename, "r", iline=iline_byte, xline=xline_byte) as f:
        for i in rvtqdm(range(len(ilines)), desc='Importing ilines', unit='il'):
            il = ilines[i]
            seis_data.zgroup['data'][i, :, :] = f.iline[il]

    # Trace spatial coordinates
    cdp_x_map = np.zeros(shape=(Nil, Nxl), dtype='f8')
    cdp_y_map = np.zeros(shape=(Nil, Nxl), dtype='f8')
    with segyio.open(filename, "r", iline=iline_byte, xline=xline_byte) as f:
        header = f.header
        Ntraces = len(header)
        for k in rvtqdm(range(Ntraces), desc='Reading spatial coodinates', unit='trace'):
            h = header[k]
            scaling_factor = h[coordinate_scaling_byte]
            if scaling_factor < 0:
                scaling_factor = 1/np.abs(scaling_factor)
            if scaling_factor == 0:
                scaling_factor = 1
            cdp_x = h[segyio.TraceField.CDP_X]*scaling_factor
            cdp_y = h[segyio.TraceField.CDP_Y]*scaling_factor
            iline = h[iline_byte]
            xline = h[xline_byte]

            iline_ind = np.argwhere(ilines==iline).flatten()
            xline_ind = np.argwhere(xlines==xline).flatten()

            if len(iline_ind) == 0 or len(xline_ind) == 0:
                raise Exception('What the heck?')
            if len(iline_ind) > 1 or len(xline_ind) > 1:
                raise Exception('What the heck?')

            cdp_x_map[iline_ind, xline_ind] = cdp_x
            cdp_y_map[iline_ind, xline_ind] = cdp_y

    seis_data.zgroup['cdp_x_map'][...] = cdp_x_map
    seis_data.zgroup['cdp_y_map'][...] = cdp_y_map

    return seis_data

"""


#%% Upanema

#%%
filename = '/togp/GOB7/Pseudowell/Upanema/sismica/PSTM_0027_UPANEMA_MGRA_T_v3o2.sgy'
iline_byte = 193
xline_byte = 197
coordinate_scaling_byte = 71
mpi_index=0
verbose=True

store = '/togp/GOB7/Pseudowell/Upanema/sismica/imported.zarr'
seis_data =  SeismicData(zarr.open(store))
#segy_import(filename, store, iline_byte, xline_byte, coordinate_scaling_byte, overwrite=True)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
x = seis_data.xlines
y = seis_data.samples
z = seis_data.iline[50].T
clip = 3
ax.pcolormesh(x, y, z, vmin=-clip, vmax=clip, cmap='gray')
ax.set_xlim(min(x), max(x))
ax.set_ylim(2500, min(y))
ax.set_xlim(600, 800)


#%% Buzios
filename = '/togp/GOB7/Pseudowell/Buzios2023/sismica/PP-DW_PSDM_T_LSRTM_AVA_3_13_GAIN_FLT_FIN_ALIGNED_CD44029.sgy'
iline_byte = 193
xline_byte = 21
coordinate_scaling_byte = 71
store = '/togp/GOB7/Pseudowell/Buzios2025/sismica/test.zarr'
#seis_data = import_segy(filename, store, iline_byte, xline_byte, coordinate_scaling_byte, overwrite=True)

seis_data =  SeismicData(zarr.open(store))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
x = seis_data.ilines
y = seis_data.samples
z = seis_data.xline[3000].T
clip = 50000
ax.pcolormesh(x, y, z, vmin=-clip, vmax=clip, cmap='gray')
ax.set_xlim(min(x), max(x))
ax.set_ylim(max(y), min(y))
#ax.set_xlim(600, 800)
#%%
"""

#store = '/togp/GOB7/Pseudowell/Buzios2025/sismica/test.zarr'
#seis_data =  SeismicData(zarr.open(store))
#clip = 50000
#seis_data.plot_section(5005, cmap='gray', vmin=-clip, vmax=clip)
#%%

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
from rockverse._utils import rvtqdm, datetimenow
from rockverse import _assert
from rockverse.errors import collective_raise
import matplotlib.pyplot as plt

from rockverse.configure import config
comm = config.mpi_comm
mpi_rank = config.mpi_rank
mpi_nprocs = config.mpi_nprocs



#CONVERTER RAISE TO COLLECTIVE RAISE
#CREATION MPI RUN OTHERS WAIT
#Seismic 2D
#GATHERS
#SEISMIC 4d

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
        Return the array of samples with appropriate intervals.

        Returns
        -------
        numpy.ndarray
            Array of samples.
        """
        return self.zgroup['samples'][...]

    @property
    def values(self):
        """
        Return the Zarr array of data values.

        Returns
        -------
        Zarr array
            Array of samples.
        """
        return self.zgroup['data']


    def get_nearest_trace(self, x, y):
        """
        Return the (x, y) array indices for trace closest to location (x, y) in CDP coordinates.
        """
        ind = None
        if mpi_rank == 0:
            distance = np.sqrt(((self.zgroup['cdp_x_map'][...] - x)**2)
                               +((self.zgroup['cdp_y_map'][...] - y)**2))
            ind = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
        ind = comm.bcast(ind, root=0)
        return ind


    def plot_section(self, number, mode='iline', **kwargs):
        """
        Plots a section of the seismic data (inline or crossline).

        Parameters
        ----------
        number : int
            Inline or crossline number to plot.
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
            If the specified line number is not found in the data.

        Example
        -------
            Plot iline 5005 in gray colormap:

            .. code-block:: python

                number, clip = 5005, 50000 # Adjust to your data
                ax, img = seis_data.plot_section(number, cmap='gray', vmin=-clip, vmax=clip)
        """
        _assert.condition.integer('number', number)
        _assert.in_group('mode', mode, ('iline', 'xline'))
        y = self.samples
        if mode == 'iline':
            x = self.xlines
            if number not in self.ilines:
                collective_raise(KeyError(f'Iline {number} not in data ilines'))
            z = self.iline[number].T
        else:
            x = self.ilines
            if number not in self.xlines:
                collective_raise(KeyError(f'Crossline {number} not in data xlines'))
            z = self.xline[number].T
        ax = plt.gca()
        quadmesh = ax.pcolormesh(x, y, z, **kwargs)
        ax.set_title(f'{mode.capitalize()} {number}')
        ax.set_xlabel('Xline' if mode == 'iline' else 'Iline')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_ylim(ax.get_ylim()[::-1])

        return ax, quadmesh


#GATHERS?
def create(store, ilines, xlines, samples, data_type, data_chunks, overwrite=False, **kwargs):
    '''
    Create empty seismic data.

    #ILINE XLINE MUST BE INTEGER
    '''
    if 'path' not in kwargs:
        kwargs['path'] = None
    kwargs['store'] = store
    kwargs['overwrite'] = overwrite
    zgroup = zarr.create_group(**kwargs)
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
#GENERALIZE CHUNKS
def import_segy(filename,
                store,
                path=None,
                overwrite=False,
                iline=segyio.TraceField.INLINE_3D,
                xline=segyio.TraceField.CROSSLINE_3D,
                cdp_x=segyio.TraceField.CDP_X,
                cdp_y=segyio.TraceField.CDP_Y,
                coordinate_scaling=segyio.TraceField.SourceGroupScalar,
                **kwargs):
    """
    Imports seismic data from a SEG-Y file into a SeismicData object.

    This function reads seismic data stored in SEG-Y format
    using the `segyio library <https://segyio.readthedocs.io/en/stable/>`_.

    Parameters
    ----------
    filename : str
        Path to the SEG-Y file.
    store : str or Zarr store
        Path or Zarr store where the imported data will be saved.
    mpi_index : int, optional
        MPI index for parallel execution. Default is 0.
    overwrite : bool, optional
        Whether to overwrite existing data in the specified store. If `True`, the
        entire store will be deleted before new data is written. Default is `False`.
    kwargs :
        Keyword arguments to be passed to the underlying
        `segyio open function <https://segyio.readthedocs.io/en/stable/segyio.html#open-and-create>`_..

    Returns
    -------
    SeismicData
        A SeismicData object representing the imported seismic data.
    """

    ilines = None
    xlines = None
    offsets = None
    samples = None
    data_type = None
    kwargs['mode'] = 'r'
    if mpi_rank == 0:
        with segyio.open(filename, iline=iline, xline=xline, **kwargs) as f:
            ilines = f.ilines
            xlines = f.xlines
            offsets = f.offsets
            samples = f.samples
            data_type = str(f.iline[ilines[0]].dtype)
    ilines = comm.bcast(ilines, root=0)
    xlines = comm.bcast(xlines, root=0)
    offsets = comm.bcast(offsets, root=0)
    samples = comm.bcast(samples, root=0)
    data_type = np.dtype(comm.bcast(data_type, root=0))

    if len(ilines) < 2 or len(xlines) < 2 or len(samples) < 2 or len(offsets) > 1:
        collective_raise(Exception('So far RockVerse only accepts post-stack 3D seismic data.'))

    Nil = len(ilines)
    Nxl = len(xlines)
    Ns = len(samples)
    seis_data = create(store=store,
                       path=path,
                       ilines=ilines,
                       xlines=xlines,
                       samples=samples,
                       data_type=data_type,
                       data_chunks=(1, Nxl, Ns),
                       overwrite=overwrite)

    # Populate main data
    with segyio.open(filename, iline=iline, xline=xline, **kwargs) as f:
        for i in rvtqdm(range(len(ilines)), desc='Importing ilines', unit='il'):
            il = ilines[i]
            seis_data.zgroup['data'][i, :, :] = f.iline[il]

    # Trace spatial coordinates
    cdp_x_map = np.zeros(shape=(Nil, Nxl), dtype='f8')
    cdp_y_map = np.zeros(shape=(Nil, Nxl), dtype='f8')
    with segyio.open(filename, iline=iline, xline=xline, **kwargs) as f:
        header = f.header
        Ntraces = len(header)
        for k in rvtqdm(range(Ntraces), desc='Reading spatial coodinates', unit='trace'):
            h = header[k]
            scaling_factor = h[coordinate_scaling]
            if scaling_factor < 0:
                scaling_factor = 1/np.abs(scaling_factor)
            if scaling_factor == 0:
                scaling_factor = 1
            cdp_x_value = h[cdp_x]*scaling_factor
            cdp_y_value = h[cdp_y]*scaling_factor
            iline_value = h[iline]
            xline_value = h[xline]

            iline_ind = np.argwhere(ilines==iline_value).flatten()[0]
            xline_ind = np.argwhere(xlines==xline_value).flatten()[0]

            cdp_x_map[iline_ind, xline_ind] = cdp_x_value
            cdp_y_map[iline_ind, xline_ind] = cdp_y_value

    seis_data.zgroup['cdp_x_map'][...] = cdp_x_map
    seis_data.zgroup['cdp_y_map'][...] = cdp_y_map

    return seis_data
#%%
#%% Upanema
#if False:
#%%
if False:
    filename = '/togp/GOB7/Pseudowell/Upanema2025/sismica/PSTM_0027_UPANEMA_MGRA_T_v3o2.sgy'
    iline, xline = 193, 197
    store = '/togp/GOB7/Pseudowell/Upanema2025/sismica/PSTM_0027_UPANEMA_MGRA_T_v3o2.zarr'
    overwrite=True
    seis_data = import_segy(filename=filename,
                            store=store,
                            overwrite=True,
                            iline=iline,
                            xline=xline)
    #seis_data = SeismicData(zarr.open(store))

    #clip=4
    #ax, img = seis_data.plot_section(414, cmap='gray_r', vmin=-clip, vmax=clip)

    #x = seis_data.zgroup['data'][...].flatten()
    #print(np.min(x), np.max(x), np.mean(x), np.std(x))

    #plt.figure()
    #plt.hist(seis_data.zgroup['data'][...].flatten(), 100)




#%% Buzios
if False:
    filename = '/togp/GOB7/Pseudowell/Buzios2023/sismica/PP-DW_PSDM_T_LSRTM_AVA_3_13_GAIN_FLT_FIN_ALIGNED_CD44029.sgy'
    iline_byte = 193
    xline_byte = 21
    coordinate_scaling_byte = 71
    store = '/togp/GOB7/Pseudowell/Buzios2025/sismica/test.zarr'
    #seis_data =  SeismicData(zarr.open(store))
    #seis_data = import_segy(filename, store, iline_byte, xline_byte, coordinate_scaling_byte, overwrite=True)
    #clip = 50000
    #ax, img = seis_data.plot_section(5005, cmap='gray', vmin=-clip, vmax=clip)

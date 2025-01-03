#%%
'''
The orthogonal slices module provides the OrthogonalViewer class for visualizing
orthogonal slices of an image along with its histogram.

.. todo::
    * Get templates from rcParams.
    * scalebar
    * calibrationbar
'''

import numpy as np
import matplotlib.pyplot as plt
import rockverse._assert as _assert
from rockverse.voxel_image.histogram import Histogram
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.pyplot as plt
from rockverse import rcparams

from numba import njit
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()


@njit()
def _region_mask_xy_slice(mask, func, ox, oy, oz, hx, hy, hz, z):
    '''
    mask: numpy array, slice from Image
    func: region_njitted function
    '''
    nx, ny = mask.shape
    x = ox
    for i in range(nx):
        x += hx
        y = float(oy)
        for j in range(ny):
            y += hy
            if not func(x, y, z):
                mask[i, j] = True


@njit()
def _region_mask_xz_slice(mask, func, ox, oy, oz, hx, hy, hz, y):
    '''
    mask: numpy array, slice from Image
    func: region_njitted function
    '''
    nx, nz = mask.shape
    x = ox
    for i in range(nx):
        x += hx
        z = oz
        for k in range(nz):
            z += hz
            if not func(x, y, z):
                mask[i, k] = True


@njit()
def _region_mask_zy_slice(mask, func, ox, oy, oz, hx, hy, hz, x):
    '''
    mask: numpy array, slice from Image
    func: region_njitted function
    '''
    ny, nz = mask.shape
    y = oy
    for j in range(ny):
        y += hy
        z = oz
        for k in range(nz):
            z += hz
            if not func(x, y, z):
                mask[j, k] = True


def _build_segmentation_colormap(self, v):
    if v in ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
             'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
             'Pastel1_r', 'Pastel2_r', 'Paired_r', 'Accent_r', 'Dark2_r', 'Set1_r',
             'Set2_r', 'Set3_r', 'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r']:
        map_colors = plt.get_cmap(v).colors
    else:
        try:
            aux = ListedColormap(v)
            map_colors = aux.colors
        except Exception:
            _assert.collective_raise(ValueError('Invalid value for segmentation colors.'))
    seg_phases = np.array(self.histogram.phases).astype(int)
    color_phases = {}
    ind = 0
    for k in seg_phases:
        color_phases[str(k)] = map_colors[ind]
        ind = (ind+1) % len(map_colors)
    cmap = np.zeros((max(seg_phases)+1, 3))
    for k in range(max(seg_phases)+1):
        cmap[k] = color_phases[str(seg_phases[np.argmin(np.abs(seg_phases-k))])]
    cmapl = [list(cmap[k, :]) for k in range(cmap.shape[0])]
    return ListedColormap(cmapl)


class OrthogonalViewer():

    """
    Visualize orthogonal slices (XY, XZ, ZY planes) of a voxel image
    and its histogram. Supports overlays for masks, segmentations,
    and region-based filtering.

    Parameters
    ----------

        image : VoxelImage
            The image object to be visualized.

        region : Region, optional
            Region object to mask specific voxels on slices and histogram.

        mask : VoxelImage, optional
            Boolean voxel image for masking specific voxels on slices and histogram.

        segmentation : VoxelImage, optional
            Segmentation array overlay to display labeled regions on slices and histogram.

        bins : int or sequence of scalars, optional
            Binning definition for histogram calculation.

        ref_voxel : tuple of int, optional
            Reference voxel (i, j, k) coordinates for slice positioning.
            Default is the center voxel.

        ref_point : tuple of float, optional
            Physical coordinates (x, y, z) in voxel length units, for slice positioning.
            Overrides `ref_voxel`.

        show_xy_plane : bool, optional
            Display the XY plane slice. Default is True.

        show_xz_plane : bool, optional
            Display the XZ plane slice. Default is True.

        show_zy_plane : bool, optional
            Display the ZY plane slice. Default is True.

        show_histogram : bool, optional
            Display histogram alongside slices. Default is True.

        show_guide_lines : bool, optional
            Show guide lines marking slice intersections. Default is True.

        hide_axis : bool, optional
            Hide axis labels and ticks in the slice plots. Default is False.

        image_dict : dict, optional
            Matplotlib's ``AxisImage`` custom options for image rendering.

        segmentation_alpha : float, optional
            Tranparency level for segmentation overlay.

        segmentation_colors : str or list, optional
            A string representing a predefined colormap or a list of colors.
            If a string, it should be the name of a Matplotlib colormap (e.g., 'Set1', 'Pastel1', 'Pastel1_r').
            If a list, it should contain colors in a format acceptable by Matplotlib (e.g., RGB tuples, hex codes).
            The colors will be cycled through and assigned to the segmentation phases.

        mask_color : Any, optional
            Color for mask overlay.  Any color format accepted by Matplotlib.

        mask_alpha : float, optional
            Tranparency level for mask overlay.

        guide_line_dict : dict, optional
            Matplotlib's ``Line2D`` custom options for guide lines (e.g., linestyle, color, linewidth).

        histogram_line_dict : dict, optional
            Custom options for Matplotlib's ``Line2D`` in the histogram plot.

            This dictionary must include the following structure:

            ```python
            {
                'full': <**kwargs>,          # Custom options for the full histogram line
                'phases': <**kwargs>,        # Custom options for segmentation phases histogram lines
                'clim': <**kwargs>           # Custom options for the vertical lines marking image CLIMs
            }
            ```

            Example:
            ```python
            histogram_line_dict = {
                'full': {'color': 'blue', 'linewidth': 2},
                'phases': {'linestyle': '--', 'alpha': 0.7},
                'clim': {'color': 'red', 'linewidth': 1}
            }
            ```

        figure_dict : dict, optional
           Dictionary of keyword arguments to be passed to the
           underlying Matplotlib figure creation.

        gridspec_dict : dict, optional
            Dictionary of keyword arguments for customizing the grid
            layout of the figure, generated using Matplotlib gridspec.
            Width and height ratios are automatically calculated from image
            dimensions.

        template : {'X-ray CT', 'Scalar field'}, optional
            The template to use for visualizing the orthogonal slices.
            This parameter determines the default settings for the viewer,
            including colormaps, alpha values, and other rendering options.
            Note that values from templates have lower precedence than other
            customization parameters.

            Available templates:
            - 'X-ray CT': Optimized for X-ray computed tomography images,
            with settings suitable for visualizing attenuation values.
            - 'Scalar field': Optimized for scalar fields such as electric
            potentials or velocity components.

            The default template is 'X-ray CT'.

        statusbar_mode : {'coordinate', 'index'}
            The desired status bar information mode when hovering the mouse
            over the figure in interactive mode:
            - 'coordinate' for physical coordinates.
            - 'index' for voxel indices.

        mpi_proc : int, optional
            MPI process rank responsible for rendering. Default is 0.

        Returns
        -------
        OrthogonalViewer
            The `OrthogonalViewer` instance.
    """

    def __init__(self,
                 image,
                 *,
                 region=None,
                 mask=None,
                 segmentation=None,
                 bins=None,
                 ref_voxel=None,
                 ref_point=None,
                 show_xy_plane=True,
                 show_xz_plane=True,
                 show_zy_plane=True,
                 show_histogram=True,
                 show_guide_lines=True,
                 hide_axis=False,
                 image_dict=None,
                 segmentation_alpha=None,
                 segmentation_colors=None,
                 mask_color=None,
                 mask_alpha=None,
                 guide_line_dict=None,
                 histogram_line_dict=None,
                 figure_dict=None,
                 gridspec_dict=None,
                 template='X-ray CT',
                 statusbar_mode = 'coordinate',
                 mpi_proc=0):

        _assert.in_group('template', template, ('X-ray CT',))

        _assert.instance('mpi_proc', mpi_proc, 'string', (int,))
        self._mpi_proc = mpi_proc

        _assert.in_group('statusbar_mode', statusbar_mode, ('coordinate', 'voxel'))
        self._statusbar_mode = statusbar_mode

        #Assign arrays and region -------------------------
        _assert.rockverse_instance(image, 'image', ('VoxelImage',))
        self._image = image

        self._image.check_mask_and_segmentation(segmentation=segmentation, mask=mask)
        self._segmentation = segmentation
        self._mask = mask

        if region is not None:
            _assert.rockverse_instance(region, 'region', ('Region',))
        self._region = region

        #Calc histogram -----------------------------------
        self._histogram = Histogram(image,
                                    bins=bins,
                                    mask=mask,
                                    segmentation=segmentation,
                                    region=region)

        self._segmentation_colors = _build_segmentation_colormap(
            self, rcparams['orthogonal_viewer'][template]['segmentation']['colors'])
        if segmentation_colors is not None:
            self._segmentation_colors = _build_segmentation_colormap(self, segmentation_colors)

        _assert.boolean('show_xy_plane', show_xy_plane)
        self._show_xy_plane = show_xy_plane

        _assert.boolean('show_xz_plane', show_xz_plane)
        self._show_xz_plane = show_xz_plane

        _assert.boolean('show_zy_plane', show_zy_plane)
        self._show_zy_plane = show_zy_plane

        _assert.boolean('show_zy_plane', show_histogram)
        self._show_histogram = show_histogram

        _assert.boolean('show_guide_lines', show_guide_lines)
        self._show_guide_lines = show_guide_lines

        _assert.boolean('hide_axis', hide_axis)
        self._hide_axis = hide_axis

        self._image_dict = {**rcparams['orthogonal_viewer'][template]['image']}
        self._image_dict['clim'] = self.histogram.percentile([0.05, 99.95])
        if image_dict is not None:
            _assert.dictionary('image_dict', image_dict)
            self._image_dict.update(**image_dict)

        self._segmentation_alpha = rcparams['orthogonal_viewer'][template]['segmentation']['alpha']
        if segmentation_alpha is not None:
            _assert.condition.integer_or_float('segmentation_alpha', segmentation_alpha)
            self._segmentation_alpha = segmentation_alpha

        self._mask_alpha = rcparams['orthogonal_viewer'][template]['mask']['alpha']
        if mask_alpha is not None:
            _assert.condition.integer_or_float('mask_alpha', mask_alpha)
            self._mask_alpha = mask_alpha

        self._mask_color = rcparams['orthogonal_viewer'][template]['mask']['color']
        if mask_color is not None:
            self._mask_color = mask_color

        self._guide_line_dict = {**rcparams['orthogonal_viewer'][template]['guide_lines']}
        if guide_line_dict is not None:
            _assert.dictionary('guide_line_dict', guide_line_dict)
            self._guide_line_dict.update(**guide_line_dict)

        self._histogram_line_dict = {**rcparams['orthogonal_viewer'][template]['histogram_lines']}
        if histogram_line_dict is not None:
            _assert.dictionary('histogram_line_dict', histogram_line_dict)
            self._histogram_line_dict.update(**histogram_line_dict)

        self._figure_dict = {**rcparams['orthogonal_viewer'][template]['figure']}
        if figure_dict is not None:
            _assert.dictionary('figure_dict', figure_dict)
            self._figure_dict.update(**figure_dict)

        self._gridspec_dict = {}
        if gridspec_dict is not None:
            _assert.dictionary('gridspec_dict', gridspec_dict)
            self._gridspec_dict.update(**gridspec_dict)

        #Set default parameters and build initial plot ----
        self._delay_update = True
        ((x0, y0, z0), (x1, y1, z1)) = self._image.bounding_box
        self.ref_point = ((x1-x0)/2, (y1-y0)/2, (z1-z0)/2)
        if ref_voxel is not None:
            self.ref_voxel = ref_voxel
        if ref_point is not None:
            self.ref_point = ref_point

        temp_dict = {'plot': True, 'ax': None,
                     'image': None, 'segmentation': None, 'mask': None,
                     'mpl_im': None, 'mpl_seg': None, 'mpl_mask': None,
                     'vline': None, 'hline': None,}
        self._slices = {
            'xy': {**temp_dict},
            'xz': {**temp_dict},
            'zy': {**temp_dict},
            'histogram': {'plot': True, 'ax': None, 'lines': {'cmin': None, 'cmax': None}},
            }

        #Figure ideal proportions
        nx, ny, nz = self._image.shape
        hx, hy, hz = self._image.voxel_length
        self._ideal_width = 0.
        if self._slices['xy']['plot'] or self._slices['xz']['plot']:
            self._ideal_width += nx*hx
        if self._slices['zy']['plot'] or self._slices['histogram']['plot']:
            self._ideal_width += nz*hz
        self._ideal_height = 0.
        if self._slices['xy']['plot'] or self._slices['zy']['plot']:
            self._ideal_height += ny*hy
        if self._slices['xz']['plot'] or self._slices['histogram']['plot']:
            self._ideal_height += nz*hz

        self._delay_update = False

        if mpi_rank == self._mpi_proc:
            self._fig = plt.figure(**self._figure_dict)
        else:
            self._fig = None

        self._build_from_scratch()

        if mpi_rank == self._mpi_proc:
            self._fig.canvas.mpl_connect('button_release_event', self._on_button_release)
            self._fig.canvas.mpl_connect('scroll_event', self._on_scroll)
            self.shrink_to_fit()

    # Builders and updaters ---------------------------------------------------
    def _get_slices(self):
        """
        Get image slices for visualization.
        """
        if self._delay_update:
            return
        ref_voxel = self._ref_voxel
        nx, ny, nz = self._image.shape
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin

        #Image
        self._slices['xy']['image'] = self._image.collective_getitem((slice(None), slice(None), ref_voxel[2])).copy()
        self._slices['xz']['image'] = self._image.collective_getitem((slice(None), ref_voxel[1], slice(None))).copy()
        self._slices['zy']['image'] = self._image.collective_getitem((ref_voxel[0], slice(None), slice(None))).copy()

        #Mask
        self._slices['xy']['mask'] = np.zeros(self._slices['xy']['image'].shape).astype('bool')
        self._slices['xz']['mask'] = np.zeros(self._slices['xz']['image'].shape).astype('bool')
        self._slices['zy']['mask'] = np.zeros(self._slices['zy']['image'].shape).astype('bool')
        if self._mask is not None:
            self._slices['xy']['mask'] = np.logical_or(
                self._slices['xy']['mask'],
                self._mask.collective_getitem((slice(None), slice(None), ref_voxel[2])))
            self._slices['xz']['mask'] = np.logical_or(
                self._slices['xz']['mask'],
                self._mask.collective_getitem((slice(None), ref_voxel[1], slice(None))))
            self._slices['zy']['mask'] = np.logical_or(
                self._slices['zy']['mask'],
                self._mask.collective_getitem((ref_voxel[0], slice(None), slice(None))))

        #Region
        if self._region is not None:
            func = self._region.contains_point_njit
            _region_mask_xy_slice(self._slices['xy']['mask'], func, ox, oy, oz, hx, hy, hz, self.ref_point[2])
            _region_mask_xz_slice(self._slices['xz']['mask'], func, ox, oy, oz, hx, hy, hz, self.ref_point[1])
            _region_mask_zy_slice(self._slices['zy']['mask'], func, ox, oy, oz, hx, hy, hz, self.ref_point[0])

        for plane in ('xy', 'xz', 'zy'):
            self._slices[plane]['mask'] = np.ma.array(
                self._slices[plane]['mask'].astype('int'),
                mask=~self._slices[plane]['mask'])

        #Segmentation
        self._slices['xy']['segmentation'] = None
        self._slices['xz']['segmentation'] = None
        self._slices['zy']['segmentation'] = None
        if self._segmentation is not None:
            self._slices['xy']['segmentation'] = self._segmentation.collective_getitem((slice(None), slice(None), ref_voxel[2])).copy()
            self._slices['xz']['segmentation'] = self._segmentation.collective_getitem((slice(None), ref_voxel[1], slice(None))).copy()
            self._slices['zy']['segmentation'] = self._segmentation.collective_getitem((ref_voxel[0], slice(None), slice(None))).copy()

    def _set_grid(self):
        """
        Set up the grid layout for the figure.

        This method determines the layout of the figure based on which planes
        (xy, xz, zy) and histogram are to be displayed. It calculates the
        appropriate width and height ratios for the grid based on the image
        dimensions and voxel sizes.

        The method performs the following operations:
        1. Calculates the total width and height of the figure.
        2. Determines the width and height ratios for the grid.
        3. Clears the current figure.
        4. Creates a new gridspec for the figure layout.

        The gridspec is created using parameters from self._gridspec_dict,
        which can be customized by the user.

        This method is typically called internally when the viewer layout
        needs to be rebuilt, such as when toggling the visibility of planes
        or the histogram. It does not return anything but updates the internal
        state of the OrthogonalViewer object, specifically self._gs (gridspec).
        """
        self._delay_update = True
        nx, ny, nz = self._image.shape
        hx, hy, hz = self._image.voxel_length
        width, height = 0, 0
        width_ratios, height_ratios = [], []
        if self._slices['xy']['plot'] or self._slices['xz']['plot']:
            width += nx*hx
            width_ratios.append(nx*hx)
        if self._slices['zy']['plot'] or self._slices['histogram']['plot']:
            width += nz*hz
            width_ratios.append(nz*hz)
        if self._slices['xy']['plot'] or self._slices['zy']['plot']:
            height += ny*hy
            height_ratios.append(ny*hy)
        if self._slices['xz']['plot'] or self._slices['histogram']['plot']:
            height += nz*hz
            height_ratios.append(nz*hz)

        self._fig.clf()
        gridspec_dict_ = {'nrows': len(height_ratios), 'ncols': len(width_ratios)}
        if len(width_ratios) > 1:
            gridspec_dict_['width_ratios'] = width_ratios
        if len(height_ratios) > 1:
            gridspec_dict_['height_ratios'] = height_ratios
        gridspec_dict_.update(**self._gridspec_dict)
        self._gs = self._fig.add_gridspec(**gridspec_dict_)
        self._delay_update = False

    def _build_xyplane(self):
        """
        Build and configure the XY plane subplot.

        This method creates and sets up the XY plane subplot in the figure.
        It performs the following operations:

        1. Determines the subplot position based on the current grid layout.
        2. Creates the subplot if XY plane is to be displayed, otherwise sets it to None.
        3. Sets the aspect ratio of the subplot to 1 (equal scaling).
        4. Calculates the extent of the image in physical units.
        5. Plots the XY slice of the main image using imshow.
        6. If present, overlays the segmentation on the XY slice.
        7. If present, overlays the mask on the XY slice.
        8. Adds vertical and horizontal guide lines at the current reference point.
        9. Sets the x and y axis labels with appropriate units.
        10. Configures the plot limits to match the image extent.


        This method is typically called internally when initializing the viewer
        or when the layout needs to be rebuilt. It does not return anything but
        updates the internal state of the OrthogonalViewer object, specifically
        the XY plane subplot and related attributes.
        """
        plane = 'xy'
        l, c = self._gs.get_geometry()
        if not self._slices[plane]['plot']:
            self._slices[plane]['ax'] = None
            return
        if l == c == 2:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[0, 0])
        else:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[0])
        axi = self._slices[plane]['ax']
        axi.format_coord = self._format_coord_xy

        axi.set_aspect(1)
        nx, ny, nz = self._image.shape
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin
        voxel_unit = self._image.voxel_unit
        extent=(ox, (nx-1)*hx+ox, (ny-1)*hy+oy, oy)
        self._slices[plane]['mpl_im'] = axi.imshow(self._slices[plane]['image'].T,
                                                   origin='upper', extent=extent,
                                                   **self._image_dict)
        if self._slices[plane]['segmentation'] is not None:
            self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'].T,
                                                        origin='upper', extent=extent,
                                                        alpha=self._segmentation_alpha,
                                                        vmin=min(self.histogram.phases),
                                                        vmax=max(self.histogram.phases),
                                                        cmap=self._segmentation_colors)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'].T,
                                                         origin='upper', extent=extent,
                                                         clim=(0, 1),
                                                         cmap=ListedColormap(['k', self.mask_color]),
                                                         alpha=self._mask_alpha)
        ref_point = self.ref_point
        self._slices[plane]['vline'] = axi.axvline(ref_point[0], **self._guide_line_dict)
        self._slices[plane]['hline'] = axi.axhline(ref_point[1], **self._guide_line_dict)
        axi.set_xlim(extent[0], extent[1])
        axi.set_ylim(extent[2], extent[3])
        axi.set_xlabel(f'{plane[0]} ({voxel_unit})' if voxel_unit else f'{plane[0]}')
        axi.set_ylabel(f'{plane[1]} ({voxel_unit})' if voxel_unit else f'{plane[1]}')
        self._set_visibility()


    def _build_zyplane(self):
        """
        See documentation for _build_xyplane.
        """
        plane = 'zy'
        l, c = self._gs.get_geometry()
        if not self._slices[plane]['plot']:
            self._slices[plane]['ax'] = None
            return
        if l == c == 2:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[0, 1])
        elif c == 2:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[1])
        else:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[0])
        axi = self._slices[plane]['ax']
        axi.format_coord = self._format_coord_zy
        axi.set_aspect(1)
        nx, ny, nz = self._image.shape
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin
        voxel_unit = self._image.voxel_unit
        extent = (oz, (nz-1)*hz+oz, (ny-1)*hy+oy, oy)
        self._slices[plane]['mpl_im'] = axi.imshow(self._slices[plane]['image'],
                                                   origin='upper', extent=extent,
                                                   **self._image_dict)
        if self._slices[plane]['segmentation'] is not None:
            self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'],
                                                       origin='upper', extent=extent,
                                                       alpha=self._segmentation_alpha,
                                                       vmin=min(self.histogram.phases),
                                                       vmax=max(self.histogram.phases),
                                                       cmap=self._segmentation_colors)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'],
                                                         origin='upper', extent=extent,
                                                         cmap=ListedColormap(['k', self._mask_color]),
                                                         alpha=self._mask_alpha)
        ref_point = self.ref_point
        self._slices[plane]['vline'] = axi.axvline(ref_point[2], **self._guide_line_dict)
        self._slices[plane]['hline'] = axi.axhline(ref_point[1], **self._guide_line_dict)
        axi.set_xlim(extent[0], extent[1])
        axi.set_ylim(extent[2], extent[3])
        axi.set_xlabel(f'{plane[0]} ({voxel_unit})' if voxel_unit else f'{plane[0]}')
        axi.set_ylabel(f'{plane[1]} ({voxel_unit})' if voxel_unit else f'{plane[1]}')
        self._set_visibility()


    def _build_xzplane(self):
        """
        See documentation for _build_xyplane
        """
        plane = 'xz'
        l, c = self._gs.get_geometry()
        if not self._slices[plane]['plot']:
            self._slices[plane]['ax'] = None
            return
        if l == c == 2:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[1, 0])
        elif l == 2:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[1])
        else:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[0])
        axi = self._slices[plane]['ax']
        axi.format_coord = self._format_coord_xz
        axi.set_aspect(1)
        nx, ny, nz = self._image.shape
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin
        voxel_unit = self._image.voxel_unit
        extent = (ox, (nx-1)*hx+ox, (nz-1)*hz+oz, oz)
        self._slices[plane]['mpl_im'] = axi.imshow(self._slices[plane]['image'].T,
                                                   origin='upper', extent=extent,
                                                   **self._image_dict)
        if self._slices[plane]['segmentation'] is not None:
            self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'].T,
                                                        origin='upper', extent=extent,
                                                        alpha=self._segmentation_alpha,
                                                        vmin=min(self.histogram.phases),
                                                        vmax=max(self.histogram.phases),
                                                        cmap=self._segmentation_colors)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'].T,
                                                         origin='upper', extent=extent,
                                                         cmap=ListedColormap(['k', self._mask_color]),
                                                         alpha=self._mask_alpha)
        ref_point = self.ref_point
        self._slices[plane]['vline'] = axi.axvline(ref_point[0], **self._guide_line_dict)
        self._slices[plane]['hline'] = axi.axhline(ref_point[2], **self._guide_line_dict)
        axi.set_xlim(extent[0], extent[1])
        axi.set_ylim(extent[2], extent[3])
        axi.set_xlabel(f'{plane[0]} ({voxel_unit})' if voxel_unit else f'{plane[0]}')
        axi.set_ylabel(f'{plane[1]} ({voxel_unit})' if voxel_unit else f'{plane[1]}')
        self._set_visibility()

    def _set_visibility(self):

        """
        Set the visibility of various plot elements.
        """
        for plane in ('xy', 'xz', 'zy'):
            axi = self._slices[plane]['ax']
            if axi is None:
                continue

            if self._hide_axis:
                axi.get_xaxis().set_visible(False)
                axi.get_yaxis().set_visible(False)
            else:
                axi.get_xaxis().set_visible(True)
                axi.get_yaxis().set_visible(True)

            if self._show_guide_lines:
                self._slices[plane]['vline'].set_visible(True)
                self._slices[plane]['hline'].set_visible(True)
            else:
                self._slices[plane]['vline'].set_visible(False)
                self._slices[plane]['hline'].set_visible(False)

    def _build_histogram_plane(self):
        """
        Builds the histogram plane.
        """
        plane = 'histogram'
        l, c = self._gs.get_geometry()
        if not self._slices[plane]['plot']:
            self._slices[plane]['ax'] = None
            return
        if l == c == 2:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[1, 1])
        elif c == 2 or l == 2:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[1])
        else:
            self._slices[plane]['ax'] = self._fig.add_subplot(self._gs[0])
        ax = self.ax_histogram
        ax.grid(True, alpha=0.5)
        x = self.histogram.bin_centers
        self._slices['histogram']['lines']['full'] = ax.plot(
            x, self.histogram.count['full'], label='full', **self._histogram_line_dict['full'])
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_ylabel('Count')
        xlabel = self.image.field_name.strip()
        if not xlabel:
            xlabel = 'Value'
        unit = self.image.field_unit.strip()
        if unit:
            xlabel = f'{xlabel} ({unit})'
        ax.set_xlabel(xlabel)
        self._slices['histogram']['lines']['cmin'] = ax.axvline(
            self._image_dict['clim'][0], **self._histogram_line_dict['clim'])
        self._slices['histogram']['lines']['cmax'] = ax.axvline(
            self._image_dict['clim'][1], **self._histogram_line_dict['clim'])
        self._plot_histogram_seg_phases()

    def _plot_histogram_seg_phases(self):
        phases = self.histogram.phases
        cmap = self._slices['xy']['mpl_seg'].get_cmap()
        lines = self._slices['histogram']['lines']
        ax = self.ax_histogram
        if len(phases)>0:
            for k in sorted(phases):
                if k in lines:
                   _ = [obj.remove() for obj in lines[k]]
                lines[k] = ax.plot(self.histogram.bin_centers,
                                   self.histogram.count[k],
                                   color=cmap(Normalize(vmin=min(self.histogram.phases),
                                                        vmax=max(self.histogram.phases))
                                                        (k)),
                                    label=f'{k}')
        ax.legend()


    def _build_from_scratch(self):
        """
        Call the various building methods to build the image from scratch.
        """
        if self._delay_update:
            return
        if not any(self._slices[plane]['plot'] for plane in ('xy', 'xz', 'zy', 'histogram')):
            self._slices['xy']['plot'] = True
        self._get_slices()
        if mpi_rank == self._mpi_proc:
            self._set_grid()
            self._build_xyplane()
            self._build_zyplane()
            self._build_xzplane()
            self._build_histogram_plane()
            self._set_visibility()
            self._update_plots()


    def _update_plots(self):
        """
        Redraw every plane in the figure.
        """
        if self._delay_update:
            return
        self._get_slices()
        ref_point = self.ref_point

        self._slices['xy']['mpl_im'].set_data(self._slices['xy']['image'].T)
        self._slices['zy']['mpl_im'].set_data(self._slices['zy']['image'])
        self._slices['xz']['mpl_im'].set_data(self._slices['xz']['image'].T)
        for plane in ('xy', 'zy', 'xz'):
            self._slices[plane]['mpl_im'].set(**self._image_dict)

        self._slices['xy']['mpl_mask'].set_data(self._slices['xy']['mask'].T)
        self._slices['zy']['mpl_mask'].set_data(self._slices['zy']['mask'])
        self._slices['xz']['mpl_mask'].set_data(self._slices['xz']['mask'].T)
        for plane in ('xy', 'zy', 'xz'):
            if self._slices[plane]['mpl_mask'] is not None:
                self._slices[plane]['mpl_mask'].set(clim=(0, 1),
                                                    cmap=ListedColormap(['k', self._mask_color]),
                                                    alpha=self._mask_alpha)

        if self._segmentation is not None:
            self._slices['xy']['mpl_seg'].set_data(self._slices['xy']['segmentation'].T)
            self._slices['zy']['mpl_seg'].set_data(self._slices['zy']['segmentation'])
            self._slices['xz']['mpl_seg'].set_data(self._slices['xz']['segmentation'].T)
            for plane in ('xy', 'zy', 'xz'):
                self._slices[plane]['mpl_seg'].set(alpha=self._segmentation_alpha,
                                                clim=(min(self.histogram.phases),
                                                        max(self.histogram.phases)),
                                                cmap=self._segmentation_colors)
        else:
            for plane in ('xy', 'zy', 'xz'):
                if self._slices[plane]['mpl_seg'] is not None:
                    self._slices[plane]['mpl_seg'].remove()
                    self._slices[plane]['mpl_seg'] = None


        self._slices['xy']['vline'].set_xdata([ref_point[0], ref_point[0]])
        self._slices['xy']['vline'].set_visible(True)
        if not self._slices['zy']['plot']:
            self._slices['xy']['vline'].set_visible(False)

        self._slices['xy']['hline'].set_ydata([ref_point[1], ref_point[1]])
        self._slices['xy']['hline'].set_visible(True)
        if not self._slices['xz']['plot']:
            self._slices['xy']['hline'].set_visible(False)

        self._slices['zy']['vline'].set_xdata([ref_point[2], ref_point[2]])
        self._slices['zy']['vline'].set_visible(True)
        if not self._slices['xy']['plot']:
            self._slices['zy']['vline'].set_visible(False)

        self._slices['zy']['hline'].set_ydata([ref_point[1], ref_point[1]])
        self._slices['zy']['hline'].set_visible(True)
        if not self._slices['xz']['plot']:
            self._slices['zy']['hline'].set_visible(False)

        self._slices['xz']['vline'].set_xdata([ref_point[0], ref_point[0]])
        self._slices['xz']['vline'].set_visible(True)
        if not self._slices['zy']['plot']:
            self._slices['xz']['vline'].set_visible(False)

        self._slices['xz']['hline'].set_ydata([ref_point[2], ref_point[2]])
        self._slices['xz']['hline'].set_visible(True)
        if not self._slices['xy']['plot']:
            self._slices['xz']['hline'].set_visible(False)

        for plane in ('xy', 'zy', 'xz'):
            if self._segmentation is not None:
                self._slices[plane]['mpl_seg'].set_clim(vmin=min(self._histogram._phases),
                                                        vmax=max(self._histogram._phases))


        self._slices['histogram']['lines']['cmin'].set_xdata((self._image_dict['clim'][0], self._image_dict['clim'][0]))
        self._slices['histogram']['lines']['cmax'].set_xdata((self._image_dict['clim'][1], self._image_dict['clim'][1]))
        self._plot_histogram_seg_phases()
        if mpi_rank == self._mpi_proc:
            self.figure.canvas.draw()


    #Methods ------------------------------------------------------------------

    def default_view(self):
        """
        Reset the viewer to its default view settings:

        * Enables display of all three orthogonal planes (XY, XZ, ZY).
        * Enables display of the histogram.
        * Enables drawing of guide lines.
        """
        self._slices['xy']['plot'] = True
        self._slices['xz']['plot'] = True
        self._slices['zy']['plot'] = True
        self._slices['histogram']['plot'] = True
        self.show_guide_lines = True #<-- This calls the update method


    def shrink_to_fit(self):
        """
        Adjust the figure size by reducing the largest side to match the ideal
        proportions of the content. It ensures that all planes and the histogram
        (if shown) fit within the figure without unnecessary white space.
        This method is useful for optimizing the figure layout, especially
        after changing which planes or components are displayed.
        """
        fwidth, fheight = self._fig.get_size_inches()
        if fwidth > fheight:
            fwidth = fheight/self._ideal_height*self._ideal_width
        else:
            fheight = fwidth/self._ideal_width*self._ideal_height
        self._fig.set_size_inches(fwidth, fheight)


    def expand_to_fit(self):
        """
        Adjust the figure size by increasing the smallest side to match the ideal
        proportions of the content. It ensures that all planes and the histogram
        (if shown) fit within the figure without unnecessary white space.
        This method is useful for optimizing the figure layout, especially
        after changing which planes or components are displayed.
        """
        fwidth, fheight = self._fig.get_size_inches()
        if fwidth < fheight:
            fwidth = fheight/self._ideal_height*self._ideal_width
        else:
            fheight = fwidth/self._ideal_width*self._ideal_height
        self._fig.set_size_inches(fwidth, fheight)


    # Properties --------------------------------------------------------------

    @property
    def figure(self):
        '''
        The Matplotlib Figure object.
        '''
        return self._fig

    @property
    def ax_xy(self):
        '''
        The Matplotlib Axes object for the xy slice.
        '''
        return self._slices['xy']['ax']

    @property
    def ax_zy(self):
        '''
        The Matplotlib Axes object for the zy slice.
        '''
        return self._slices['zy']['ax']

    @property
    def ax_xz(self):
        '''
        The Matplotlib Axes object for the xz slice.
        '''
        return self._slices['xz']['ax']

    @property
    def ax_histogram(self):
        '''
        The Matplotlib Axes object for the histogram plot.
        '''
        return self._slices['histogram']['ax']

    #Image must not have a setter
    @property
    def image(self):
        '''
        Get the input image data. Cannot be changed once the object is created.
        '''
        return self._image

    @property
    def region(self):
        '''
        Get or set the region of interest.

        Examples
        --------
            >>> from rockverse.regions import Cylinder
            >>> viewer = rockverse.plot.OrthogonalViewer(<your parameters here...>)
            >>> viewer.region = Cylinder(<your parameters here...>)  # Set a new region
            >>> region = viewer.region                               # Get the current region
            >>> viewer.region = None                                 # Remove the region
        '''
        return self._region

    @region.setter
    def region(self, v):
        if v is not None:
            _assert.rockverse_instance(v, 'region', ('Region',))
        self._region = v
        self._histogram = Histogram(self._image,
                                    bins=self._histogram.bins,
                                    mask=self._mask,
                                    segmentation=self._segmentation,
                                    region=self._region)
        self._update_plots()


    @property
    def mask(self):
        """
        Get or set the mask image.

        Examples
        --------
            >>> viewer = rockverse.plot.OrthogonalViewer(<your parameters here...>)
            >>> mask = viewer.mask      # Get the current mask
            >>> viewer.mask = new_mask  # Set a new mask
            >>> viewer.mask = None      # Remove the mask
        """
        return self._mask


    @mask.setter
    def mask(self, v):
        self._image.check_mask_and_segmentation(mask=v)
        self._mask = v
        self._histogram.mask = v
        self._update_plots()


    @property
    def segmentation(self):
        """
        Get or set the segmentation image.

        Examples
        --------
            >>> viewer = rockverse.plot.OrthogonalViewer(<your parameters here...>)
            >>> segmentation = viewer.segmentation     # Get the current segmentation
            >>> viewer.segmentation = new_segmentation # Set a new array for segmentation
            >>> viewer.segmentation = None             # Remove the segmentation
        """
        return self._segmentation


    @segmentation.setter
    def segmentation(self, v):
        self._image.check_mask_and_segmentation(segmentation=v)
        self._segmentation = v
        self._histogram = Histogram(self._image,
                                    bins=self._histogram.bins,
                                    mask=self._mask,
                                    segmentation=self._segmentation,
                                    region=self._region)
        self._update_plots()

    @property
    def segmentation_colors(self):
        '''
        Get or set the colormap for the segmentation phases.

        - If a string, it should be the name of a predefined Matplotlib colormap
          (e.g., 'Set1', 'Pastel1', 'Pastel1_r').
        - If a list, it should contain colors in a format acceptable by Matplotlib,
          such as RGB tuples (e.g., (1, 0, 0) for red) or hex codes (e.g., '#FF0000' for red).

        The colors will be cycled through and assigned to the segmentation phases.
        '''
        return self._segmentation_colors

    @segmentation_colors.setter
    def segmentation_colors(self, v):
        self._segmentation_colors = _build_segmentation_colormap(self, v)
        self._update_plots()

    @property
    def histogram(self):
        '''
        The :class:`rockverse.Histogram` object.
        '''
        return self._histogram

    @property
    def ref_voxel(self):
        '''
        Get or set the plot reference point in voxel position. Voxel position
        must be an iterable (i, j, k), where 0 <= i < nx, 0 <= j < ny, and
        0 <= k < nz, with nx, ny, nz = image.shape.

        Examples
        --------
           >>> viewer = rockverse.plot.OrthogonalViewer(<your parameters here...>)
           >>> ref_voxel = viewer.ref_voxel #get the current reference voxel
           >>> viewer.ref_voxel = (8, 33, 9) # Set new reference voxel and update
        '''
        return tuple(self._ref_voxel)

    @ref_voxel.setter
    def ref_voxel(self, v):
        if v is None:
            self._ref_voxel = np.floor(np.array(self._image.shape)/2).astype(int)
        else:
            _assert.iterable.ordered_numbers('ref_voxel', v)
            _assert.iterable.length('ref_voxel', v, 3)
            nx, ny, nz = self._image.shape
            rx = int(round(v[0]))
            rx = 0 if rx < 0 else rx
            rx = nx-1 if rx >= nx else rx
            ry = int(round(v[1]))
            ry = 0 if ry < 0 else ry
            ry = ny-1 if ry >= ny else ry
            rz = int(round(v[2]))
            rz = 0 if rz < 0 else rz
            rz = nz-1 if rz >= nz else rz
            self._ref_voxel = np.array((rx, ry, rz)).astype(int)
        self._update_plots()



    @property
    def ref_point(self):
        """
        Get or set the plot reference point in voxel units.

        Get or set the plot reference point in image position. Image position
        must be an ordered iterable (x, y, z), where ox <= x < ox+hx*nx,
        oy <= y < oy+hy*ny, and oz <= z < oz+hz*nz, with
        ox, oy, oz = image.voxel_origin,
        hx, hy, hz = image.voxel_length, and
        nx, ny, nz = image.shape.
        If the point (x, y, z) is not a grid point, the closest grid point will
        be used.

        Examples
        --------
           >>> viewer = rockverse.OrthogonalViewer(<your parameters here...>)
           >>> ref_point = viewer.ref_voxel #get the current reference voxel
           >>> viewer.ref_point = (3.33, 14.72, 10) # Set new reference point and update
        """
        point = np.array(self._ref_voxel).astype(float)
        point *= np.array(self._image.voxel_length).astype(float)
        point += np.array(self._image.voxel_origin).astype(float)
        return tuple(point)


    @ref_point.setter
    def ref_point(self, v):
        _assert.iterable.ordered_numbers('ref_point', v)
        _assert.iterable.length('ref_point', v, 3)
        nx, ny, nz = self._image.shape
        voxel = np.array(v).astype(float)
        voxel -= np.array(self._image.voxel_origin).astype(float)
        voxel /= np.array(self._image.voxel_length).astype(float)
        self.ref_voxel = voxel


    @property
    def show_xy_plane(self):
        '''
        Boolean to enable/disable the xy slice visibility.
        '''
        return self._slices['xy']['plot']

    @show_xy_plane.setter
    def show_xy_plane(self, v):
        _assert.instance('show_xy_plane', v, 'boolean', (bool,))
        if self._slices['xy']['plot'] != v:
            self._slices['xy']['plot'] = v
            self._build_from_scratch()

    @property
    def show_xz_plane(self):
        '''
        Boolean to enable/disable the xz slice visibility.
        '''
        return self._slices['xz']['plot']

    @show_xz_plane.setter
    def show_xz_plane(self, v):
        _assert.instance('show_xz_plane', v, 'boolean', (bool,))
        if self._slices['xz']['plot'] != v:
            self._slices['xz']['plot'] = v
            self._build_from_scratch()

    @property
    def show_zy_plane(self):
        '''
        Boolean to enable/disable the zy slice visibility.
        '''
        return self._slices['zy']['plot']

    @show_zy_plane.setter
    def show_zy_plane(self, v):
        _assert.instance('show_zy_plane', v, 'boolean', (bool,))
        if self._slices['zy']['plot'] != v:
            self._slices['zy']['plot'] = v
            self._build_from_scratch()

    @property
    def show_histogram(self):
        '''
        Boolean to enable/disable the histogram plot visibility.
        '''
        return self._slices['histogram']['plot']

    @show_histogram.setter
    def show_histogram(self, v):
        _assert.instance('show_histogram', v, 'boolean', (bool,))
        if self._slices['histogram']['plot'] != v:
            self._slices['histogram']['plot'] = v
            self._build_from_scratch()

    @property
    def show_guide_lines(self):
        '''
        Boolean to enable/disable the guide lines visibility.
        '''
        return self._show_guide_lines

    @show_guide_lines.setter
    def show_guide_lines(self, v):
        _assert.instance('show_guide_lines', v, 'boolean', (bool,))
        if self._show_guide_lines != v:
            self._show_guide_lines = v
            self._update_plots()

    @property
    def hide_axis(self):
        '''
        Boolean. If True, do not draw the image axes and labels.
        Default is False.
        '''
        return self._hide_axis

    @hide_axis.setter
    def hide_axis(self, v):
        _assert.instance('hide_axis', v, 'boolean', (bool,))
        if self._hide_axis != v:
            self._hide_axis = v
            self._set_visibility()

    @property
    def image_dict(self):
        """
        Dictionary of keyword arguments for customizing the image display.

        This dictionary is passed to Matplotlib's imshow function when
        displaying the image slices. It can be used to control aspects
        such as colormap, alpha (transparency), etc. See the documentation for
        Matplotlib imshow function. Changing elements directly in this dictionary
        requires calling the update function for the changes to take effect.

        See Also
        --------
        update_image_dict : Method to update this dictionary.
        reset_image_dict : Method to reset this dictionary to default values.
        """
        return self._image_dict

    @property
    def statusbar_mode(self):
        """
        Get or set the status bar display mode.

        Parameters
        ----------
        mode : {'coordinate', 'index'}
            The desired status bar mode:
            - 'coordinate' for physical coordinates.
            - 'index' for voxel indices.
        """
        return self._statusbar_mode

    @statusbar_mode.setter
    def statusbar_mode(self, v):
        _assert.in_group('statusbar_mode', v, ('coordinate', 'voxel'))
        self._statusbar_mode = v

    def update_image_dict(self, **kwargs):
        """
        Update the image display settings dictionary and refreshes the display.
        See the documentation for the image_dict method.
        """
        self._image_dict.update(**kwargs)
        self._update_plots()

    @property
    def segmentation_alpha(self):
        """
        Get or set the transparency level for segmentation overlay.
        """
        return self._segmentation_alpha

    @segmentation_alpha.setter
    def segmentation_alpha(self, v):
        self._segmentation_alpha = v
        self._update_plots()

    @property
    def mask_color(self):
        """
        Get or set the color for mask overlay.  It can be any color format accepted by Matplotlib.
        """
        return self._mask_color


    @mask_color.setter
    def mask_color(self, v):
        self._mask_color = v
        self._update_plots()

    @property
    def mask_alpha(self):
        """
        Get or set the transparency level for mask overlay.
        """
        return self._mask_alpha

    @mask_alpha.setter
    def mask_alpha(self, v):
        self._mask_alpha = v
        self._update_plots()

    @property
    def guide_line_dict(self):
        """
        Dictionary of keyword arguments for customizing the guide lines.

        This dictionary is passed to Matplotlib's axvline and axhline functions
        when drawing the guide lines. It can be used to control aspects such as
        color, line style, line width, etc. See the documentation for
        Matplotlib axvline and axhline functions. Changing elements directly in
        this dictionary requires calling the update function for the changes to
        take effect.

        See Also
        --------
        update_guide_line_dict : Method to update this dictionary.
        reset_guide_line_dict : Method to reset this dictionary to default values.
        """
        return self._guide_line_dict

    def update_guide_line_dict(self, **kwargs):
        """
        Update the guide line display settings dictionary and refreshes the display.
        See the documentation for the guide_line_dict method.
        """
        self._guide_line_dict.update(**kwargs)
        self._update_plots()

    @property
    def gridspec_dict(self):
        """
        Dictionary of keyword arguments for customizing the figure's grid layout.

        This dictionary is passed to Matplotlib's GridSpec when creating the
        figure layout. It can be used to control aspects such as spacing
        between subplots. Width and height ratios are automatically calculated
        from image dimensions. See the documentation for Matplotlib GridSpec.
        Changing elements directly in this dictionary requires calling the
        update function for the changes to take effect.

        See Also
        --------
        update_gridspec_dict : Method to update this dictionary.
        reset_gridspec_dict : Method to reset this dictionary to default values.
        """
        return self._gridspec_dict

    def update_gridspec_dict(self, **kwargs):
        """
        Update the gridspec settings dictionary and refreshes the display.
        See the documentation for the gridspec_dict method.
        """
        self._gridspec_dict.update(**kwargs)
        self._update_plots()

    def reset_gridspec_dict(self):
        """
        Reset the guide gridspec settings to default values and refreshes the display.
        """
        self._gridspec_dict = {}
        self._update_plots()

    def _get_ijk_xyz(self, xo, yo, axis):
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin
        if axis == 'xy':
            i = int(round((xo-ox)/hx))
            j = int(round((yo-oy)/hy))
            k = self.ref_voxel[2]
        elif axis == 'xz':
            i = int(round((xo-ox)/hx))
            j = self.ref_voxel[1]
            k = int(round((yo-oz)/hz))
        elif axis == 'zy':
            i = self.ref_voxel[0]
            j = int(round((yo-oy)/hy))
            k = int(round((xo-oz)/hz))
        x = float(ox+i*hx)
        y = float(oy+j*hy)
        z = float(oz+k*hz)
        if self._statusbar_mode == 'coordinate':
            msg = f"(x, y, z): ({x:.2f}, {y:.2f}, {z:.2f})"
        else:
            msg = f"(i, j, k): {i, j, k}"
        return i, j, k, msg

    def _format_coord_xy(self, xo, yo):
        nx, ny, nz = self._image.shape
        plane = 'xy'
        i, j, k, msg = self._get_ijk_xyz(xo, yo, plane)
        label = 'v'
        if 0 <= i < nx and 0 < j < ny:
            value = f"{self._slices[plane]['image'][i, j]:1.2f}"
        else:
            value = "-"
        if self.segmentation is not None:
            label = label + ', s'
            if 0 <= i < nx and 0 < j < ny:
                value = f"{value}, {self._slices[plane]['segmentation'][i, j]:d}"
            else:
                value = "-"
        if self.mask is not None or self.region is not None:
            label = label + ', m'
            if 0 <= i < nx and 0 < j < ny:
                value = f"{value}, {not self._slices[plane]['mask'].mask[i, j]}"
            else:
                value = "-"

        left = "(" if len(label)>1 else ""
        right = ")" if len(label)>1 else ""
        return f"{msg}, {left}{label}{right}: {left}{value}{right}"

    def _format_coord_xz(self, xo, yo):
        nx, ny, nz = self._image.shape
        plane = 'xz'
        i, j, k, msg = self._get_ijk_xyz(xo, yo, plane)
        label = 'v'
        value = '-'
        if 0 <= i < nx and 0 <= k < nz:
            value = f"{self._slices[plane]['image'][i, k]:1.2f}"
        else:
            value = '-'
        if self.segmentation is not None:
            label = label + ', s'
            if 0 <= i < nx and 0 <= k < nz:
                value = f"{value}, {self._slices[plane]['segmentation'][i, k]:d}"
            else:
                value = '-'
        if self.mask is not None or self.region is not None:
            label = label + ', m'
            if 0 <= i < nx and 0 <= k < nz:
                value = f"{value}, {not self._slices[plane]['mask'].mask[i, k]}"
            else:
                value = '-'
        left = "(" if len(label)>1 else ""
        right = ")" if len(label)>1 else ""
        return f"{msg}, {left}{label}{right}: {left}{value}{right}"

    def _format_coord_zy(self, xo, yo):
        nx, ny, nz = self._image.shape
        plane = 'zy'
        i, j, k, msg = self._get_ijk_xyz(xo, yo, plane)
        label = 'v'
        if 0 <= j < ny and 0 <= k < nz:
            value = f"{self._slices[plane]['image'][j, k]:1.2f}"
        else:
            value = '-'
        if self.segmentation is not None:
            label = label + ', s'
            if 0 <= j < ny and 0 <= k < nz:
                value = f"{value}, {self._slices[plane]['segmentation'][j, k]:d}"
            else:
                value = '-'
        if self.mask is not None or self.region is not None:
            label = label + ', m'
            if 0 <= j < ny and 0 <= k < nz:
                value = f"{value}, {not self._slices[plane]['mask'].mask[j, k]}"
            else:
                value = '-'
        left = "(" if len(label)>1 else ""
        right = ")" if len(label)>1 else ""
        return f"{msg}, {left}{label}{right}: {left}{value}{right}"

    # Events ------------------------------------------------------------------
    def _on_button_release(self, event):
        """
        Handle mouse button release events.

        This method updates the viewer based on where the user clicked:
        - On the histogram: adjusts image contrast
        - On image planes: updates reference point or prints voxel data
        """

        def print_data(self, ref_point):
            hx, hy, hz = self._image.voxel_length
            ox, oy, oz = self._image.voxel_origin
            i = int(round((ref_point[0]-ox)/hx))
            j = int(round((ref_point[1]-oy)/hy))
            k = int(round((ref_point[2]-oz)/hz))
            pr = f"\nvoxel: {i, j, k}"

            x = float(ox+i*hx)
            y = float(oy+j*hy)
            z = float(oz+k*hz)
            pr = pr + f"\npoint: {x, y, z}"
            if self.image.voxel_unit:
                pr = pr + f' {self.image.voxel_unit}'

            pr = pr + f"\nimage value: {self.image.collective_getitem((i, j, k))}"
            if self.image.field_unit:
                pr = pr + f' {self.image.field_unit}'
            if self.segmentation is not None:
                pr = pr + f"\nsegmentation phase: {self.segmentation.collective_getitem((i, j, k))}"
            if self.mask is not None:
                pr = pr + f"\nmasked: {self.mask.collective_getitem((i, j, k))}"
            print(pr)

        x, y = event.xdata, event.ydata
        ref_point = list(self.ref_point)
        vmin, vmax = self.image_dict['clim']

        if event.inaxes == self._slices['histogram']['ax']:
            if event.button == MouseButton.LEFT:
                vmin = max(x, self.histogram.min)
                if vmin < vmax:
                    self.image_dict.update(clim=(vmin, vmax))
                    self._update_plots()
            if event.button == MouseButton.RIGHT:
                vmax = min(x, self.histogram.max)
                if vmax > vmin:
                    self.image_dict.update(clim=(vmin, vmax))
                    self._update_plots()
            if event.button == MouseButton.MIDDLE:
                self.image_dict.update(clim=self.histogram.percentile([0.05, 99.95]))
                self._update_plots()

        if event.inaxes == self._slices['xy']['ax']:
            ref_point[0] = x
            ref_point[1] = y
            if event.button == MouseButton.RIGHT:
                self.ref_point = ref_point
            if event.button == MouseButton.LEFT:
                print_data(self, ref_point)
            if event.button == MouseButton.MIDDLE:
                bb = self._image.bounding_box
                ref_point[0] = 0.5*(bb[0][0]+bb[1][0])
                ref_point[1] = 0.5*(bb[0][1]+bb[1][1])
                self.ref_point = ref_point

        if event.inaxes == self._slices['xz']['ax']:
            ref_point[0] = x
            ref_point[2] = y
            if event.button == MouseButton.RIGHT:
                self.ref_point = ref_point
            if event.button == MouseButton.LEFT:
                print_data(self, ref_point)
            if event.button == MouseButton.MIDDLE:
                bb = self._image.bounding_box
                ref_point[0] = 0.5*(bb[0][0]+bb[1][0])
                ref_point[2] = 0.5*(bb[0][2]+bb[1][2])
                self.ref_point = ref_point

        if event.inaxes == self._slices['zy']['ax']:
            ref_point[2] = x
            ref_point[1] = y
            if event.button == MouseButton.RIGHT:
                self.ref_point = ref_point
            if event.button == MouseButton.LEFT:
                print_data(self, ref_point)
            if event.button == MouseButton.MIDDLE:
                bb = self._image.bounding_box
                ref_point[2] = 0.5*(bb[0][2]+bb[1][2])
                ref_point[1] = 0.5*(bb[0][1]+bb[1][1])
                self.ref_point = ref_point


    def _on_scroll(self, event):
        """
        Handle scroll events to navigate through image slices.
        """
        nx, ny, nz = self._image.shape
        ref_voxel = np.array(self.ref_voxel)
        if event.inaxes == self._slices['xy']['ax']:
            ref_voxel[2] -= event.step
            if ref_voxel[2] < 0:
                ref_voxel[2] = 0
            if ref_voxel[2] > (nz-1):
                ref_voxel[2] = nz-1
            self.ref_voxel = ref_voxel
        elif event.inaxes == self._slices['xz']['ax']:
            ref_voxel[1] -= event.step
            if ref_voxel[1] < 0:
                ref_voxel[1] = 0
            if ref_voxel[1] > (ny-1):
                ref_voxel[1] = ny-1
            self.ref_voxel = ref_voxel
        elif event.inaxes == self._slices['zy']['ax']:
            ref_voxel[0] -= event.step
            if ref_voxel[0] < 0:
                ref_voxel[0] = 0
            if ref_voxel[0] > (nx-1):
                ref_voxel[0] = nx-1
            self.ref_voxel = ref_voxel



import rockverse as rv

image = rv.voxel_image.import_raw(
    rawfile='/path/to/rawdata/Bentheimer/Cropped_Oxyz_001_001_001_Nxyz_500_500_500.raw',
    store='/path/to/imported/Bentheimer/original',  #<- path to the imported the voxel image
    shape=(500, 500, 500),         #<- From metadata, image size
    dtype='<u2',                     #<- From metadata, big-endian 16-bit unsigned integer
    offset=0,                        #<- From metadata
    voxel_length=(5, 5, 5),          #<- From metadata
    voxel_unit='um',                 #<- From metadata, micrometer
    raw_file_order='F',              #<- Fortran file order
    chunks=(250, 250, 250),          #<- Our choice of chunk size will give a 2x2x2 chunk grid
    field_name='Attenuation',        #<- Our choice for field name (X-ray attenuation)
    field_unit='a.u.',               #<- Our choice for field units (arbitrary units)
    description='Bentheimer sandstone original X-ray CT',
    overwrite=True                   #<- Overwrite if file exists in disk
    )

segmentation = rv.voxel_image.import_raw(
    rawfile='/path/to/rawdata/Bentheimer/Seg_Oxyz_0001_0001_0001.raw',
    store='/path/to/imported/Bentheimer/segmented',  #<- path to the imported the voxel image
    shape=(500, 500, 500),           #<- From metadata, image size
    dtype='|u1',                     #<- From metadata, big-endian 16-bit unsigned integer
    offset=0,                        #<- From metadata
    voxel_length=(5, 5, 5),          #<- From metadata
    voxel_unit='um',                 #<- From metadata, micrometer
    raw_file_order='F',              #<- Fortran file order
    chunks=(250, 250, 250),          #<- Our choice of chunk size will give a 2x2x2 chunk grid
    field_name='',                   #<- Our choice for field name (X-ray attenuation)
    field_unit='',                   #<- Our choice for field units (arbitrary units)
    description='Bentheimer sandstone segmentation',
    overwrite=True                   #<- Overwrite if file exists in disk
    )


import matplotlib.pyplot as plt
plt.close('all')
self=OrthogonalViewer(image=image, segmentation=segmentation,
                      region=rv.region.Cylinder(p=(1000,1001,1000), r=1000, v=(0,1,1)))

self.mask_alpha=1
self.mask_color='w'
self.segmentation_alpha=0.75
self.segmentation_colors='Set1'
self.mask_color=[0.25, 0.30, 0.25]

#plt.colorbar(mappable=self._slices['xy']['mpl_seg'], ax=self._slices['xy']['ax'])

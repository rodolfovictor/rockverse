'''
The orthogonal slices module provides the OrthogonalViewer class for visualizing
orthogonal slices of an image along with its histogram.

.. todo::
    * Add scalebar
    * Add calibrationbar
    * Add contour lines to scalar fields
    * Limit to 3D images
'''

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import Normalize, ListedColormap, to_rgba
from numba import njit
from mpi4py import MPI

from rockverse import _assert
from rockverse import rcparams
from rockverse.voxel_image.histogram import Histogram

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
            Unsigned int voxel image with segmentation phases to overlay labeled
            regions on slices and histogram.

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
            If a string, it should be the name of a Matplotlib qualitative colormap
            (e.g., 'Set1', 'Pastel1', 'Pastel1_r').
            If a list, it should contain colors in a format acceptable by Matplotlib
            (e.g., RGB tuples, hex codes).
            The colors will be cycled through and assigned to the segmentation phases.

        mask_color : Any, optional
            Color for mask overlay.  Any color format accepted by Matplotlib.

        mask_alpha : float, optional
            Tranparency level for mask overlay. It must be a float between 0.0 and 1.0.

        guide_line_dict : dict, optional
            Matplotlib's ``Line2D`` custom options for guide lines
            (e.g., linestyle, color, linewidth).

        histogram_line_dict : dict, optional
            Custom options for Matplotlib's ``Line2D`` in the histogram plot.

            This dictionary must include the following structure:

            .. code-block:: python

                {
                    'full': <**kwargs>,   # For the full histogram
                    'phases': <**kwargs>, # For segmentation phases
                    'clim': <**kwargs>    # For the vertical CLIM lines
                }

            Example:

            .. code-block:: python

                histogram_line_dict = {
                    'full': {'color': 'blue', 'linewidth': 2},
                    'phases': {'linestyle': '--', 'alpha': 0.7},
                    'clim': {'color': 'red', 'linewidth': 1}
                }

        figure_dict : dict, optional
           A Dictionary of keyword arguments to be passed to the
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

            - 'X-ray CT': the default value, optimized for X-ray computed tomography
              images, with settings suitable for visualizing attenuation values.
            - 'Scalar field': optimized for scalar fields such as electric
              potentials or velocity components.

        statusbar_mode : {'coordinate', 'index'}
            The desired status bar information mode when hovering the mouse
            over the figure in interactive mode:

            - 'coordinate' for physical coordinates.
            - 'index' for voxel indices.

        layout : {'2x2', 'vertical', 'horizontal'}, optional
            The layout configuration for displaying the orthogonal slices and histogram.

            This parameter determines how the various components (XY, XZ, ZY slices,
            and histogram) are arranged within the figure. The following layout options
            are available:

            - '2x2': Arranges the XY and ZY slices in the top row and the XZ slice
              and histogram in the bottom row, creating a 2x2 grid layout.
            - 'vertical': Stacks the XY, ZY, XZ, and XZ slices vertically in a single column,
              and places the histogram below them.
            - 'horizontal': Places the XY slice on the left, followed by the ZY slice,
              the XZ slice, and the histogram plot.

            The default layout is '2x2'. This parameter allows for flexible visualization
            of the slices and histogram based on user preferences or specific analysis needs.

        mpi_proc : int, optional
            MPI process rank responsible for figure rendering. Default is 0.

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
                 layout='2x2',
                 mpi_proc=0):

        _assert.in_group('template', template, ('X-ray CT', 'scalar field'))

        _assert.in_group('layout', layout, ('2x2', 'vertical', 'horizontal'))
        self._layout = layout

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

        self._histogram = Histogram(image,
                                    bins=bins,
                                    mask=mask,
                                    segmentation=segmentation,
                                    region=region)


        self._segmentation_colors = None
        self._segmentation_colormap = None
        if segmentation_colors is not None:
            self._build_segmentation_colormap(segmentation_colors)
        else:
            self._build_segmentation_colormap(
                rcparams['orthogonal_viewer'][template]['segmentation']['colors'])


        _assert.boolean('hide_axis', hide_axis)
        self._hide_axis = hide_axis

        self._image_dict = copy.deepcopy(rcparams['orthogonal_viewer'][template]['image'])
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

        self._guide_line_dict = copy.deepcopy(rcparams['orthogonal_viewer'][template]['guide_lines'])
        if guide_line_dict is not None:
            _assert.dictionary('guide_line_dict', guide_line_dict)
            self._guide_line_dict.update(**guide_line_dict)

        self._histogram_line_dict = copy.deepcopy(rcparams['orthogonal_viewer'][template]['histogram_lines'])
        if histogram_line_dict is not None:
            _assert.dictionary('histogram_line_dict', histogram_line_dict)
            for key in ('full', 'phases', 'clim'):
                if key in histogram_line_dict:
                    self._histogram_line_dict[key].update(**histogram_line_dict[key])

        self._figure_dict = copy.deepcopy(rcparams['orthogonal_viewer'][template]['figure'])
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

        _assert.boolean('show_guide_lines', show_guide_lines)
        self._show_guide_lines = show_guide_lines

        self._delay_update = False

        self._fig = plt.figure(**self._figure_dict)





        self._build_planes()

        if mpi_rank == self._mpi_proc or True:
            self._fig.canvas.mpl_connect('button_press_event', self._on_button_click)
            self._fig.canvas.mpl_connect('button_release_event', self._on_button_release)
            self._fig.canvas.mpl_connect('scroll_event', self._on_scroll)

        _assert.boolean('show_xy_plane', show_xy_plane)
        self.show_xy_plane = show_xy_plane

        _assert.boolean('show_xz_plane', show_xz_plane)
        self.show_xz_plane = show_xz_plane

        _assert.boolean('show_zy_plane', show_zy_plane)
        self.show_zy_plane = show_zy_plane

        _assert.boolean('show_histogram', show_histogram)
        self.show_histogram = show_histogram

        self.shrink_to_fit()

    # Builders and updaters ---------------------------------------------------
    def _get_slices(self):
        """
        Get image slices for visualization.
        """
        if self._delay_update:
            return
        ref_voxel = self._ref_voxel
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin

        #Image
        self._slices['xy']['image'] = self._image.collective_getitem(
            (slice(None), slice(None), ref_voxel[2])).copy()
        self._slices['xz']['image'] = self._image.collective_getitem(
            (slice(None), ref_voxel[1], slice(None))).copy()
        self._slices['zy']['image'] = self._image.collective_getitem(
            (ref_voxel[0], slice(None), slice(None))).copy()

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
        self._slices['xy']['segmentation'] = np.zeros(self._slices['xy']['image'].shape).astype('int')
        self._slices['xz']['segmentation'] = np.zeros(self._slices['xz']['image'].shape).astype('int')
        self._slices['zy']['segmentation'] = np.zeros(self._slices['zy']['image'].shape).astype('int')
        if self._segmentation is not None:
            self._slices['xy']['segmentation'] = self._segmentation.collective_getitem((slice(None), slice(None), ref_voxel[2])).copy()
            self._slices['xz']['segmentation'] = self._segmentation.collective_getitem((slice(None), ref_voxel[1], slice(None))).copy()
            self._slices['zy']['segmentation'] = self._segmentation.collective_getitem((ref_voxel[0], slice(None), slice(None))).copy()

    def _set_grid(self):
        """
        Set up the grid layout for the figure.
        """
        self._delay_update = True
        nx, ny, nz = self._image.shape
        hx, hy, hz = self._image.voxel_length
        width_ratios, height_ratios = [], []
        if self._layout == '2x2':
            width_ratios.append(nx*hx)
            width_ratios.append(nz*hz)
            height_ratios.append(ny*hy)
            height_ratios.append(nz*hz)
        elif self._layout == 'horizontal':
            height_ratios.append(0)
            if self._slices['xy']['plot']:
                width_ratios.append(nx*hx)
                height_ratios[0] = max(height_ratios[0], ny*hy)
            if self._slices['zy']['plot']:
                width_ratios.append(nz*hz)
                height_ratios[0] = max(height_ratios[0], ny*hy)
            if self._slices['xz']['plot']:
                width_ratios.append(nx*hx)
                height_ratios[0] = max(height_ratios[0], nz*hz)
            if self._slices['histogram']['plot']:
                width_ratios.append(nz*hz)
                height_ratios[0] = max(height_ratios[0], nz*hz)
        elif self._layout == 'vertical':
            width_ratios.append(0)
            if self._slices['xy']['plot']:
                width_ratios[0] = max(width_ratios[0], nx*hx)
                height_ratios.append(ny*hy)
            if self._slices['zy']['plot']:
                width_ratios[0] = max(width_ratios[0], nz*hz)
                height_ratios.append(ny*hy)
            if self._slices['xz']['plot']:
                width_ratios[0] = max(width_ratios[0], nx*hx)
                height_ratios.append(nz*hz)
            if self._slices['histogram']['plot']:
                width_ratios[0] = max(width_ratios[0], nz*hz)
                height_ratios.append(nz*hz)
        else:
            _assert.collective_raise(Exception('What is happening?!'))

        self._ideal_width = np.sum(width_ratios)
        self._ideal_height = np.sum(height_ratios)

        gridspec_dict_ = {'nrows': len(height_ratios), 'ncols': len(width_ratios)}
        if len(width_ratios) > 1:
            gridspec_dict_['width_ratios'] = width_ratios
        if len(height_ratios) > 1:
            gridspec_dict_['height_ratios'] = height_ratios
        self._fig.clf()
        gridspec_dict_.update(**self._gridspec_dict)
        gs = self._fig.add_gridspec(**gridspec_dict_)

        if self._layout == '2x2':
            self._slices['xy']['gridspec'] = gs[0, 0]
            self._slices['zy']['gridspec'] = gs[0, 1]
            self._slices['xz']['gridspec'] = gs[1, 0]
            self._slices['histogram']['gridspec'] = gs[1, 1]
        else:
            count = 0
            if self._slices['xy']['plot']:
                self._slices['xy']['gridspec'] = gs[count]
                count += 1
            if self._slices['zy']['plot']:
                self._slices['zy']['gridspec'] = gs[count]
                count += 1
            if self._slices['xz']['plot']:
                self._slices['xz']['gridspec'] = gs[count]
                count += 1
            if self._slices['histogram']['plot']:
                self._slices['histogram']['gridspec'] = gs[count]
                count += 1
        self._delay_update = False

    def _build_xyplane(self):
        """
        Build and configure the XY plane subplot.
        """
        plane = 'xy'
        if not self._slices[plane]['plot']:
            self._slices[plane]['ax'] = None
            return
        self._slices[plane]['ax'] = self._fig.add_subplot(self._slices[plane]['gridspec'])
        axi = self._slices[plane]['ax']
        axi.format_coord = self._format_coord_xy

        axi.set_aspect(1)
        nx, ny, _ = self._image.shape
        hx, hy, _ = self._image.voxel_length
        ox, oy, _ = self._image.voxel_origin
        voxel_unit = self._image.voxel_unit
        extent=(ox, (nx-1)*hx+ox, (ny-1)*hy+oy, oy)
        self._slices[plane]['mpl_im'] = axi.imshow(self._slices[plane]['image'].T,
                                                   origin='upper', extent=extent,
                                                   **self._image_dict)
        if self._segmentation is None:
            vmin, vmax = 0, 1
        else:
            vmin = min(self.histogram.phases)
            vmax = max(self.histogram.phases)
        self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'].T,
                                                    origin='upper', extent=extent,
                                                    alpha=self._segmentation_alpha,
                                                    vmin=vmin, vmax=vmax,
                                                    cmap=self._segmentation_colormap)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'].T,
                                                         origin='upper', extent=extent,
                                                         clim=(0, 1),
                                                         cmap=ListedColormap(
                                                             ['k', self._mask_color]),
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
        Build and configure the ZY plane subplot.
        """
        plane = 'zy'
        if not self._slices[plane]['plot']:
            self._slices[plane]['ax'] = None
            return
        self._slices[plane]['ax'] = self._fig.add_subplot(self._slices[plane]['gridspec'])
        axi = self._slices[plane]['ax']
        axi.format_coord = self._format_coord_zy
        axi.set_aspect(1)
        _, ny, nz = self._image.shape
        _, hy, hz = self._image.voxel_length
        _, oy, oz = self._image.voxel_origin
        voxel_unit = self._image.voxel_unit
        extent = (oz, (nz-1)*hz+oz, (ny-1)*hy+oy, oy)
        self._slices[plane]['mpl_im'] = axi.imshow(self._slices[plane]['image'],
                                                   origin='upper', extent=extent,
                                                   **self._image_dict)
        if self._segmentation is None:
            vmin, vmax = 0, 1
        else:
            vmin = min(self.histogram.phases)
            vmax = max(self.histogram.phases)
        self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'],
                                                    origin='upper', extent=extent,
                                                    alpha=self._segmentation_alpha,
                                                    vmin=vmin, vmax=vmax,
                                                    cmap=self._segmentation_colormap)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'],
                                                         origin='upper', extent=extent,
                                                         cmap=ListedColormap(
                                                             ['k', self._mask_color]),
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
        Build and configure the XZ plane subplot.
        """
        plane = 'xz'
        if not self._slices[plane]['plot']:
            self._slices[plane]['ax'] = None
            return
        self._slices[plane]['ax'] = self._fig.add_subplot(self._slices[plane]['gridspec'])
        axi = self._slices[plane]['ax']
        axi.format_coord = self._format_coord_xz
        axi.set_aspect(1)
        nx, _, nz = self._image.shape
        hx, _, hz = self._image.voxel_length
        ox, _, oz = self._image.voxel_origin
        voxel_unit = self._image.voxel_unit
        extent = (ox, (nx-1)*hx+ox, (nz-1)*hz+oz, oz)
        self._slices[plane]['mpl_im'] = axi.imshow(self._slices[plane]['image'].T,
                                                   origin='upper', extent=extent,
                                                   **self._image_dict)
        if self._segmentation is None:
            vmin, vmax = 0, 1
        else:
            vmin = min(self.histogram.phases)
            vmax = max(self.histogram.phases)
        self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'].T,
                                                    origin='upper', extent=extent,
                                                    alpha=self._segmentation_alpha,
                                                    vmin=vmin, vmax=vmax,
                                                    cmap=self._segmentation_colormap)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'].T,
                                                         origin='upper', extent=extent,
                                                         cmap=ListedColormap(
                                                             ['k', self._mask_color]),
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
        if not self._slices[plane]['plot']:
            self._slices[plane]['ax'] = None
            return
        self._slices[plane]['ax'] = self._fig.add_subplot(self._slices[plane]['gridspec'])
        ax = self.ax_histogram
        self._plot_histogram_full()
        self._plot_histogram_seg_phases()

    def _plot_histogram_full(self):
        """
        Clear the histogram axes and plot the full line.
        """
        if not self.show_histogram:
            return
        ax = self.ax_histogram
        ax.cla()
        ax.grid(True, alpha=0.5)
        x = self.histogram.bin_centers
        self._slices['histogram']['lines']['full'], = ax.plot(
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

    def _plot_histogram_seg_phases(self):
        """
        Plot the histogram for the segmentation phases.
        """
        if not self.show_histogram:
            return
        phases = self.histogram.phases
        lines = self._slices['histogram']['lines']
        to_pop = []
        for k, v in lines.items():
            if k not in ['cmin', 'cmax', 'full']:
                v.remove()
                to_pop.append(k)
        for k in to_pop:
            _ = lines.pop(k)
        ax = self.ax_histogram
        if len(phases)>0:
            cmap = self._slices['xy']['mpl_seg'].get_cmap()
            for k in sorted(phases):
                lines[str(k)], = ax.plot(self.histogram.bin_centers,
                                        self.histogram.count[k],
                                        color=cmap(Normalize(vmin=min(self.histogram.phases),
                                                             vmax=max(self.histogram.phases))
                                                             (k)),
                                        label=f'{k}',
                                        **self._histogram_line_dict['phases'])
        ax.legend()

    def _build_planes(self):
        """
        Call the various building methods to build the image from scratch.
        """
        if self._delay_update:
            return
        if not any(self._slices[plane]['plot'] for plane in ('xy', 'xz', 'zy', 'histogram')):
            self._slices['xy']['plot'] = True
        self._get_slices()
        if mpi_rank == self._mpi_proc or True:
            self._set_grid() #Set up the grid layout.
            self._build_xyplane()
            self._build_zyplane()
            self._build_xzplane()
            self._build_histogram_plane()
            self._set_visibility()
            self._update_plots()

    def _update_plots(self):
        """
        Update every plane in the figure.
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
                                                   cmap=self._segmentation_colormap)
        else:
            for plane in ('xy', 'zy', 'xz'):
                self._slices[plane]['mpl_seg'].set(alpha=0)

        self._slices['xy']['vline'].set_xdata([ref_point[0], ref_point[0]])
        self._slices['xy']['vline'].set(**self.guide_line_dict)
        self._slices['xy']['vline'].set_visible(True)
        if not self._slices['zy']['plot']:
            self._slices['xy']['vline'].set_visible(False)

        self._slices['xy']['hline'].set_ydata([ref_point[1], ref_point[1]])
        self._slices['xy']['hline'].set(**self.guide_line_dict)
        self._slices['xy']['hline'].set_visible(True)
        if not self._slices['xz']['plot']:
            self._slices['xy']['hline'].set_visible(False)

        self._slices['zy']['vline'].set_xdata([ref_point[2], ref_point[2]])
        self._slices['zy']['vline'].set(**self.guide_line_dict)
        self._slices['zy']['vline'].set_visible(True)
        if not self._slices['xy']['plot']:
            self._slices['zy']['vline'].set_visible(False)

        self._slices['zy']['hline'].set_ydata([ref_point[1], ref_point[1]])
        self._slices['zy']['hline'].set(**self.guide_line_dict)
        self._slices['zy']['hline'].set_visible(True)
        if not self._slices['xz']['plot']:
            self._slices['zy']['hline'].set_visible(False)

        self._slices['xz']['vline'].set_xdata([ref_point[0], ref_point[0]])
        self._slices['xz']['vline'].set(**self.guide_line_dict)
        self._slices['xz']['vline'].set_visible(True)
        if not self._slices['zy']['plot']:
            self._slices['xz']['vline'].set_visible(False)

        self._slices['xz']['hline'].set_ydata([ref_point[2], ref_point[2]])
        self._slices['xz']['hline'].set(**self.guide_line_dict)
        self._slices['xz']['hline'].set_visible(True)
        if not self._slices['xy']['plot']:
            self._slices['xz']['hline'].set_visible(False)

        for plane in ('xy', 'zy', 'xz'):
            if self._segmentation is not None:
                self._slices[plane]['mpl_seg'].set_clim(vmin=min(self._histogram.phases),
                                                        vmax=max(self._histogram.phases))

        self._slices['histogram']['lines']['cmin'].set_xdata(
            (self._image_dict['clim'][0], self._image_dict['clim'][0]))
        self._slices['histogram']['lines']['cmin'].set(**self._histogram_line_dict['clim'])

        self._slices['histogram']['lines']['cmax'].set_xdata(
            (self._image_dict['clim'][1], self._image_dict['clim'][1]))
        self._slices['histogram']['lines']['cmax'].set(**self._histogram_line_dict['clim'])

        self._slices['histogram']['lines']['full'].set(**self._histogram_line_dict['full'])

        self._plot_histogram_seg_phases()
        if mpi_rank == self._mpi_proc or True:
            self.figure.canvas.draw()


    #Methods ------------------------------------------------------------------

    def shrink_to_fit(self):
        """
        Adjust the figure size to fit the ideal proportions of the content.

        This method reduces the size of the figure by scaling down the largest
        side to ensure that all axes and components fit neatly within the figure
        without unnecessary white space. It helps optimize the layout of the
        visualization, particularly after modifications to the displayed planes
        or components.

        This method is especially useful when toggling the visibility of
        different slices or adjusting the figure's contents, ensuring that
        the presentation remains visually appealing and focused on the data.
        """
        if mpi_rank == self._mpi_proc or True:
            fwidth, fheight = self._fig.get_size_inches()
            if fwidth/self._ideal_width > fheight/self._ideal_height:
                fwidth = fheight/self._ideal_height*self._ideal_width
            else:
                fheight = fwidth/self._ideal_width*self._ideal_height
            self._fig.set_size_inches(fwidth, fheight)


    def expand_to_fit(self):
        """
        Adjust the figure size to ensure that all content fits within the figure.

        This method increases the size of the figure by scaling up the smallest
        side to match the ideal proportions of the displayed content. It ensures
        that all axes, components, and visual elements are adequately visible
        without clipping or unnecessary overlap.

        This method is particularly useful after changing which planes or
        components are displayed, helping to maintain an optimized layout and
        providing a more comprehensive view of the data.
        """
        if mpi_rank == self._mpi_proc or True:
            fwidth, fheight = self._fig.get_size_inches()
            if fwidth/self._ideal_width < fheight/self._ideal_height:
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
        The input image data.
        '''
        return self._image


    @property
    def histogram_bins(self):
        """
        Get or set the histogram bins.
        """
        return self._histogram.bins

    @histogram_bins.setter
    def histogram_bins(self, v):
        self._histogram = Histogram(self._image,
                                    bins=v,
                                    mask=self._mask,
                                    segmentation=self._segmentation,
                                    region=self._region)
        self._plot_histogram_full()
        self._update_plots()

    @property
    def region(self):
        '''
        Get or set the region of interest.

        Examples
        --------
            >>> from rockverse.regions import Cylinder
            >>> viewer = rockverse.OrthogonalViewer(<your parameters here...>)
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
        self._plot_histogram_full()
        self._update_plots()

    @property
    def mask(self):
        """
        Get or set the mask image.

        Examples
        --------
            >>> viewer = rockverse.OrthogonalViewer(<your parameters here...>)
            >>> mask = viewer.mask            # Get the current mask
            >>> viewer.mask = new_mask_image  # Set a new voxel image as mask
            >>> viewer.mask = None            # Remove the mask
        """
        return self._mask


    @mask.setter
    def mask(self, v):
        self._image.check_mask_and_segmentation(mask=v)
        self._mask = v
        self._histogram = Histogram(self._image,
                                    bins=self._histogram.bins,
                                    mask=self._mask,
                                    segmentation=self._segmentation,
                                    region=self._region)
        self._plot_histogram_full()
        self._update_plots()


    @property
    def segmentation(self):
        """
        Get or set the segmentation image.

        Examples
        --------
            >>> viewer = rockverse.OrthogonalViewer(<your parameters here...>)
            >>> segmentation = viewer.segmentation     # Get the current segmentation image
            >>> viewer.segmentation = new_segmentation # Set a new image for segmentation
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
        self._build_segmentation_colormap(self._segmentation_colors)
        self._update_plots()
        self._plot_histogram_seg_phases()

    @property
    def segmentation_colors(self):
        '''
        Get or set the color list for the segmentation phases.

        - If a string, it should be the name of a predefined Matplotlib qualitative colormap
          (e.g., 'Set1', 'Pastel1', 'Pastel1_r').
        - If a list, it should contain colors in any format acceptable by Matplotlib,
          such as RGB tuples (e.g., (1, 0, 0) for red), hex codes (e.g., '#FF0000' for red),
          or named colors ('gold', 'forestgreen', etc.).

        The specified colors will be cycled through and assigned to the segmentation phases.

        Examples:
            >>> viewer.segmentation_colors = 'tab10'  # Using a predefined Matplotlib colormap
            >>> viewer.segmentation_colors = [(1, 0, 0), '#339966', 'gold']  # Using a list with any valid format
        '''
        return self._segmentation_colors

    @segmentation_colors.setter
    def segmentation_colors(self, v):
        self._build_segmentation_colormap(v)
        self._update_plots()

    @property
    def segmentation_colormap(self):
        '''
        Get the segmentation colorlist as a Matplotlib colormap.
        '''
        return self._segmentation_colormap


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
        must be an iterable (i, j, k), where ``0 <= i < nx``, ``0 <= j < ny``, and
        ``0 <= k < nz``, with ``nx, ny, nz = image.shape``.

        Examples
        --------
           >>> viewer = rockverse.OrthogonalViewer(<your parameters here...>)
           >>> ref_voxel = viewer.ref_voxel #get the current reference voxel
           >>> viewer.ref_voxel = (8, 33, 9) # Set new reference voxel and update
        '''
        return tuple(int(k) for k in self._ref_voxel)

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

        Get or set the plot reference point in image position. It
        must be an ordered iterable (x, y, z), where ``ox <= x < ox+hx*nx``,
        ``oy <= y < oy+hy*ny``, and ``oz <= z < oz+hz*nz``, with
        ``ox, oy, oz = image.voxel_origin``,
        ``hx, hy, hz = image.voxel_length``, and
        ``nx, ny, nz = image.shape``.
        If (x, y, z) is not a grid point, the closest grid point will
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
        return tuple(float(k) for k in point)


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
        Boolean flag to enable or disable the visibility of the XY slice.

        When set to True, the XY slice will be displayed in the viewer.
        When set to False, the XY slice will be hidden. Changing this
        setting will rebuild the visualization to reflect
        the current state.

        Examples
        --------
            >>> viewer.show_xy_plane          # Get the current state
            >>> viewer.show_xy_plane = True   # Show the XY slice
            >>> viewer.show_xy_plane = False  # Hide the XY slice
        '''
        return self._slices['xy']['plot']

    @show_xy_plane.setter
    def show_xy_plane(self, v):
        _assert.instance('show_xy_plane', v, 'boolean', (bool,))
        if self._slices['xy']['plot'] != v:
            self._slices['xy']['plot'] = v
            self._build_planes()

    @property
    def show_xz_plane(self):
        '''
        Boolean flag to enable or disable the visibility of the XZ slice.

        When set to True, the XZ slice will be displayed in the viewer.
        When set to False, the XZ slice will be hidden. Changing this
        setting will rebuild the visualization to reflect
        the current state.

        Examples
        --------
            >>> viewer.show_xz_plane          # Get the current state
            >>> viewer.show_xz_plane = True   # Show the XZ slice
            >>> viewer.show_xz_plane = False  # Hide the XZ slice
        '''
        return self._slices['xz']['plot']

    @show_xz_plane.setter
    def show_xz_plane(self, v):
        _assert.instance('show_xz_plane', v, 'boolean', (bool,))
        if self._slices['xz']['plot'] != v:
            self._slices['xz']['plot'] = v
            self._build_planes()

    @property
    def show_zy_plane(self):
        '''
        Boolean flag to enable or disable the visibility of the ZY slice.

        When set to True, the ZY slice will be displayed in the viewer.
        When set to False, the ZY slice will be hidden. Changing this
        setting will rebuild the visualization to reflect
        the current state.

        Examples
        --------
            >>> viewer.show_zy_plane          # Get the current state
            >>> viewer.show_zy_plane = True   # Show the ZY slice
            >>> viewer.show_zy_plane = False  # Hide the ZY slice
        '''
        return self._slices['zy']['plot']

    @show_zy_plane.setter
    def show_zy_plane(self, v):
        _assert.instance('show_zy_plane', v, 'boolean', (bool,))
        if self._slices['zy']['plot'] != v:
            self._slices['zy']['plot'] = v
            self._build_planes()

    @property
    def show_histogram(self):
        '''
        Boolean flag to enable or disable the visibility of the histogram plot.

        Examples
        --------
            >>> viewer.show_histogram         # Get the current visibility state of the histogram
            >>> viewer.show_histogram = True  # Show the histogram
            >>> viewer.show_histogram = False # Hide the histogram
        '''
        return self._slices['histogram']['plot']

    @show_histogram.setter
    def show_histogram(self, v):
        _assert.instance('show_histogram', v, 'boolean', (bool,))
        if self._slices['histogram']['plot'] != v:
            self._slices['histogram']['plot'] = v
            self._build_planes()

    @property
    def show_guide_lines(self):
        '''
        Boolean flag to enable or disable the visibility of the guide lines for
        slice intersection.

        Examples
        --------
            >>> viewer.show_guide_lines         # Get the current visibility state
            >>> viewer.show_guide_lines = True  # Show the guide lines
            >>> viewer.show_guide_lines = False # Hide the guide lines
        '''
        return self._show_guide_lines

    @show_guide_lines.setter
    def show_guide_lines(self, v):
        _assert.instance('show_guide_lines', v, 'boolean', (bool,))
        if self._show_guide_lines != v:
            self._show_guide_lines = v
            self._update_plots()
            self._set_visibility()

    @property
    def hide_axis(self):
        '''
        Boolean flag to control the visibility of image axes and labels.

        When set to True, the axes and labels for the image slices will not be drawn,
        providing a cleaner visual presentation that focuses solely on the image data.
        This can be useful in cases where the axes are not needed for interpretation
        or when a more aesthetic visualization is desired. The default value is False.

        Examples
        --------
            >>> viewer.hide_axis         # Get the current visibility state of the axes
            >>> viewer.hide_axis = True  # Hide the axes and labels
            >>> viewer.hide_axis = False # Show the axes and labels
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
        Get the dictionary of keyword arguments for customizing the image display.

        This dictionary is passed to Matplotlib's imshow function when
        displaying the image slices. It can be used to control aspects
        such as colormap, alpha (transparency), etc. See the documentation for
        Matplotlib imshow function.

        **Important:** Do not change elements directly in this dictionary.
        Call the ``update_image_dict`` method instead to apply any changes.
        """
        return self._image_dict


    def update_image_dict(self, **kwargs):
        """
        Update the image display settings dictionary and refreshes the display.
        See the documentation for the image_dict property.

        Example
        -------
            >>> # Update colormap and transparency
            >>> viewer.update_image_dict(cmap='gray', alpha=0.8)
        """
        self._image_dict.update(**kwargs)
        self._update_plots()


    @property
    def histogram_lines(self):
        """
        A dictionary with the Matplotlib ``Line2D`` for each line in the histogram plot.
        """
        return self._slices['histogram']['lines']

    @property
    def histogram_line_dict(self):
        """
        Get the dictionary of keyword arguments for customizing the histogram lines.

        **Important:** Do not change elements directly in this dictionary.
        Call the ``update_image_dict`` method instead to apply any changes.
        """
        return self._histogram_line_dict


    def update_histogram_line_dict(self, v):
        """
        Update the histogram lines display settings dictionary and refreshes the display.
        Value must be a dictionary with the following structure:

        .. code-block:: python

            {
                'full': <**kwargs>,   # For the full histogram
                'phases': <**kwargs>, # For segmentation phases
                'clim': <**kwargs>    # For the vertical CLIM lines
            }

        Examples:

        .. code-block:: python

            viewer.update_histogram_line_dict({
                'full': {'color': 'blue', 'linewidth': 2},
                'phases': {'linestyle': '--', 'alpha': 0.7},
                'clim': {'color': 'red', 'linewidth': 1}
            })

        You can use only a subset of the keywords:

        .. code-block:: python

            viewer.update_histogram_line_dict({
                'phases': {'linestyle': '--'}})

        """
        _assert.dictionary('histogram_line_dict', v)
        for key in ('full', 'phases', 'clim'):
            if key in v:
                self._histogram_line_dict[key].update(**v[key])
        self._update_plots()


    @property
    def statusbar_mode(self):
        """
        Get or set the status bar display mode.

        This property determines the information displayed in the status bar
        when hovering the mouse over the figure in insteractive mode. It can show
        either the physical coordinates of the cursor position or the voxel indices
        of the corresponding data point.

        Available modes:

        - 'coordinate': displays the physical coordinates (x, y, z) in the current
          units of the image.
        - 'index': displays the voxel indices (i, j, k) corresponding to the cursor
          position in the voxel grid.

        Examples
        --------
            >>> viewer.statusbar_mode                 # Get the current mode
            >>> viewer.statusbar_mode = 'coordinate'  # Set status bar to show physical coordinates
            >>> viewer.statusbar_mode = 'index'       # Set status bar to show voxel indices

        """
        return self._statusbar_mode


    @statusbar_mode.setter
    def statusbar_mode(self, v):
        _assert.in_group('statusbar_mode', v, ('coordinate', 'voxel'))
        self._statusbar_mode = v

    @property
    def segmentation_alpha(self):
        """
        Get or set the transparency level for segmentation overlay.
        The value must be a float between 0.0 and 1.0, where 0.0 is fully
        transparent (invisible) and 1.0 is fully opaque (completely visible).

        Examples
        --------
            >>> viewer.segmentation_alpha = 0.5            # Set the transparency to 50%
            >>> current_alpha = viewer.segmentation_alpha  # Get the current transparency level
        """
        return self._segmentation_alpha

    @segmentation_alpha.setter
    def segmentation_alpha(self, v):
        self._segmentation_alpha = v
        self._update_plots()

    @property
    def mask_color(self):
        """
        Get or set the color for the mask overlay.

        This property defines the color used for the mask overlay displayed on the
        image slices. The color can be specified in any format accepted by Matplotlib,
        including named colors (e.g., 'red', 'blue', 'royalblue'), RGB tuples (e.g., (1, 0, 0) for red),
        or hex codes (e.g., '#00FF00' for green).

        The chosen color will be used to visually indicate masked areas on the image.

        Examples
        --------
            >>> current_color = viewer.mask_color           # Get the current mask color
            >>> viewer.mask_color = 'white'                 # Set the mask color to white
            >>> viewer.mask_color = (0.25, 0.30, 0.25)      # Set the mask color using an RGB tuple
            >>> viewer.mask_color = '#008000'               # Set the mask color using a hex code
        """
        return self._mask_color


    @mask_color.setter
    def mask_color(self, v):
        self._mask_color = v
        self._update_plots()

    @property
    def mask_alpha(self):
        """
        Get or set the transparency level for the mask overlay.

        This property controls the alpha (transparency) value of the mask overlay
        displayed on the image slices. The value must be a float between 0.0 and 1.0,
        where 0.0 is fully transparent (invisible) and 1.0 is fully opaque (completely
        visible).

        Examples
        --------
            >>> viewer.mask_alpha = 0.5   # Set the transparency to 50%
            >>> current_alpha = viewer.mask_alpha  # Get the current transparency level
        """
        return self._mask_alpha

    @mask_alpha.setter
    def mask_alpha(self, v):
        self._mask_alpha = v
        self._update_plots()

    @property
    def guide_line_dict(self):
        """
        Get the dictionary of Matplotlib's ``Line2D`` keyword arguments used
        for customizing the guide lines.

        **Important:** Do not change elements directly in this dictionary.
        Call the ``update_guide_line_dict`` method instead to apply any changes.
        """
        return self._guide_line_dict


    def update_guide_line_dict(self, **kwargs):
        """
        Update the guide line display settings dictionary and refresh the display.

        This method allows you to modify the settings in the `guide_line_dict`
        that control the appearance of the guide lines drawn on the image slices.
        You can update various parameters by passing the desired keyword arguments.

        Examples
        --------
            >>> # Update guide lines to be red, dashed, and with a width of 2
            >>> viewer.update_guide_line_dict(color='red', linestyle='--', linewidth=2)
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

        **Important:** Do not change elements directly in this dictionary.
        Call the ``update_gridspec_dict`` method instead to apply any changes.
        """
        return self._gridspec_dict


    def update_gridspec_dict(self, **kwargs):
        """
        Update the gridspec settings dictionary and rebuilds the display.
        See the documentation for the gridspec_dict method.

        Examples
        --------
            >>> viewer.update_gridspec_dict(width_ratios=[1, 4])  # Update width ratios
        """
        self._gridspec_dict.update(**kwargs)
        self._build_planes()

    @property
    def layout(self):
        """
        Get or set the figure grid layout. Allowed values are:

        - '2x2': Arranges the XY and ZY slices in the top row and the XZ slice
          and histogram in the bottom row, creating a 2x2 grid layout.
        - 'vertical': Stacks the XY, XZ, and ZY slices vertically in a single column,
          and places the histogram below them.
        - 'horizontal': Places the XY slice on the left, the ZY slice in the middle,
          and the XZ slice on the right, with the histogram below.

        The default layout is '2x2'. This parameter allows for flexible visualization
        of the slices and histogram based on user preferences or specific analysis needs.

        Examples
        --------
            >>> layout = viewer.layout      #Get the current layout state
            >>> viewer.layout = 'vertical'  #Set the grid layout to vertical mode

        """
        return self._layout


    @layout.setter
    def layout(self, v):
        _assert.in_group('layout', v, ('2x2', 'vertical', 'horizontal'))
        self._layout = v
        self._build_planes()


    def _get_ijk_xyz(self, xo, yo, axis):
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin
        i, j, k = self.ref_voxel
        if axis == 'xy':
            i = int(round((xo-ox)/hx))
            j = int(round((yo-oy)/hy))
        elif axis == 'xz':
            i = int(round((xo-ox)/hx))
            k = int(round((yo-oz)/hz))
        elif axis == 'zy':
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
    def _on_button_click(self, event):
        """
        Just stores the click location.
        """
        self._event = [event.inaxes, event.x, event.y]

    def _on_button_release(self, event):
        """
        Handle mouse button release events.

        This method updates the viewer based on where the user clicked:
        - On the histogram: adjusts image contrast
        - On image planes: updates reference point or prints voxel data
        """

        if (event.inaxes != self._event[0]
            or abs(event.x-self._event[1])>5
            or abs(event.y-self._event[2])>5):
            return

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


    def _build_segmentation_colormap(self, v):
        if v in ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
                'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
                'Pastel1_r', 'Pastel2_r', 'Paired_r', 'Accent_r', 'Dark2_r', 'Set1_r',
                'Set2_r', 'Set3_r', 'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r']:
            self._segmentation_colors = [to_rgba(k) for k in plt.get_cmap(v).colors]
        else:
            try:
                converted = [to_rgba(k) for k in v]
                aux = ListedColormap(converted)
                self._segmentation_colors = aux.colors
            except Exception as e:
                _assert.collective_raise(ValueError(f'Invalid value for segmentation colors: {e}'))
        seg_phases = np.array(self.histogram.phases).astype(int)
        if len(seg_phases) == 0:
            self._segmentation_colormap = ListedColormap(self._segmentation_colors)
            return
        color_phases = {}
        ind = 0
        for k in seg_phases:
            color_phases[str(k)] = self._segmentation_colors[ind]
            ind = (ind+1) % len(self._segmentation_colors)
        cmap = np.zeros((max(seg_phases)+1, 4))
        for k in range(max(seg_phases)+1):
            cmap[k] = color_phases[str(seg_phases[np.argmin(np.abs(seg_phases-k))])]
        cmapl = [list(cmap[k, :]) for k in range(cmap.shape[0])]
        self._segmentation_colormap = ListedColormap(cmapl)
        return

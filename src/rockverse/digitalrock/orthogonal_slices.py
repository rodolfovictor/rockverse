#%%
'''
The orthogonal slices module provides the OrthogonalViewer class for visualizing
orthogonal slices of an image along with its histogram.

.. todo::
    * Get templates from rcParams.
    * Make a colormap for segmentation
    * scalebar
    * calibrationbar
'''

import numpy as np
import matplotlib.pyplot as plt
import rockverse._assert as _assert
from rockverse.digitalrock.voxel_image.histogram import Histogram
from rockverse.digitalrock.region.region import Region
from matplotlib.backend_bases import MouseButton
from numba import njit
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_nprocs = comm.Get_size()

FIGURE_DICT = {'layout': 'compressed'}
IMAGE_DICT = dict(cmap='gray', origin='upper', interpolation='none')
SEGMENTATION_DICT = dict(cmap='tab10', alpha=0.5, origin='upper', interpolation='none')
MASK_DICT = dict(cmap='gray_r', alpha=0.75, origin='upper', interpolation='none')
GUIDE_LINE_DICT = dict(linestyle='-', color='y', alpha=0.75, linewidth=1)

@njit()
def _region_mask_slice_cpu(mask, func, X, Y, Z):
    '''
    mask: numpy array, slice from Image
    func: region_njitted function
    X, Y, Z: meshgrid for voxel coordinates
    '''
    nxm, nym = mask.shape
    nxx, nyx = X.shape
    nxy, nyy = Y.shape
    nxz, nyz = Z.shape
    if (nxx!=nxm or nxy!=nxm or nxz!=nxm or nyx!=nym or nyy!=nym or nyz!=nym):
        raise Exception('Invalid shapes.')
    for i in range(nxm):
        for j in range(nym):
            if not func(X[i, j], Y[i, j], Z[i, j]):
                mask[i, j] = True


class OrthogonalViewer():

    """
    Visualize orthogonal slices (XY, XZ, ZY planes) of a voxel image
    and its histogram. Supports overlays like masks, segmentations,
    and region-based filtering.

    Parameters
    ----------

        image : drp.core.Array
            The image object to be visualized.

        region : drp.regions.Region, optional
            Region object to mask specific voxels on slices and histogram.

        mask : drp.core.Array, optional
            Boolean voxel image for masking specific voxels.

        segmentation : drp.core.Array, optional
            Segmentation array overlay to display labeled regions.

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

        hide_masked : bool, optional
            Hide masked voxels in the slices. Default is False.

        hide_axis : bool, optional
            Hide axis labels and ticks in the slices. Default is False.

        image_dict : dict, optional
            Matplotlib's ``AxisImage`` custom options for image rendering.

        segmentation_dict : dict, optional
            Matplotlib's ``AxisImage`` custom options for rendering segmentation overlays.

        mask_dict : dict, optional
            Matplotlib's ``AxisImage`` custom options for rendering masks.

        guide_line_dict : dict, optional
            Matplotlib's ``Line2D`` custom options for guide lines (e.g., linestyle, color, linewidth).

        figure_dict : dict, optional
           Dictionary of keyword arguments to be passed to the
           underlying Matplotlib figure creation.

        gridspec_dict : dict, optional
            Optional dictionary of keyword arguments for customizing the grid
            layout of the figure, generated using Matplotlib gridspec.
            Width and height ratios are automatically calculated from image
            dimensions.

        background_color : str, optional
            The background color of the plots. Usually visible when
            `hide_masked` is True. It can be any color format accepted by
            Matplotlib. Default is 'k' (black).

        statusbar_mode : {'coordinate', 'index'}
            The desired status bar information mode when hovering the mouse
            over the figure:
            - 'coordinate' for physical coordinates.
            - 'index' for voxel indices.

        mpi_proc : int, optional
            MPI process rank responsible for rendering. Default is 0.

        Returns
        -------
        OrthogonalViewer
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
                 hide_masked=False,
                 hide_axis=False,
                 image_dict=None,
                 segmentation_dict=None,
                 mask_dict=None,
                 guide_line_dict=None,
                 figure_dict=None,
                 gridspec_dict=None,
                 background_color='k',
                 statusbar_mode = 'coordinate',
                 mpi_proc=0):

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
            _assert.instance('region', region, 'Region', (Region,))
        self._region = region

        #Calc histogram -----------------------------------
        self._histogram = Histogram(image,
                                    bins=bins,
                                    mask=mask,
                                    segmentation=segmentation,
                                    region=region)

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

        _assert.boolean('hide_masked', hide_masked)
        self._hide_masked = hide_masked

        _assert.boolean('hide_axis', hide_axis)
        self._hide_axis = hide_axis

        self._image_dict = {**IMAGE_DICT}
        if image_dict is not None:
            _assert.dict('image_dict', image_dict)
            self._image_dict.update(**image_dict)

        self._segmentation_dict = {**SEGMENTATION_DICT}
        if segmentation_dict is not None:
            _assert.dict('segmentation_dict', segmentation_dict)
            self._segmentation_dict.update(**segmentation_dict)

        self._mask_dict = {**MASK_DICT}
        if mask_dict is not None:
            _assert.dict('mask_dict', mask_dict)
            self._mask_dict.update(**mask_dict)

        self._guide_line_dict = {**GUIDE_LINE_DICT}
        if guide_line_dict is not None:
            _assert.dict('guide_line_dict', guide_line_dict)
            self._guide_line_dict.update(**guide_line_dict)

        self._figure_dict = {**FIGURE_DICT}
        if figure_dict is not None:
            _assert.dict('figure_dict', figure_dict)
            self._figure_dict.update(**figure_dict)

        self._gridspec_dict = {}
        if gridspec_dict is not None:
            _assert.dict('gridspec_dict', gridspec_dict)
            self._gridspec_dict.update(**gridspec_dict)

        self._background_color = background_color

        #Set default parameters and build initial plot ----
        self._delay_update = True
        self.ref_voxel = ref_voxel
        self.ref_point = ref_point

        temp_dict = {'plot': True, 'ax': None,
                     'image': None, 'segmentation': None, 'mask': None,
                     'mpl_im': None, 'mpl_seg': None, 'mpl_mask': None,
                     'vline': None, 'hline': None,}
        self._slices = {
            'xy': {**temp_dict},
            'xz': {**temp_dict},
            'zy': {**temp_dict},
            'histogram': {'plot': True, 'ax': None, 'leftline': None, 'rightline': None},
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
        Prepare image slices for visualization.

        This method extracts 2D slices from the 3D image data based on the
        current reference voxel. It handles the main image, segmentation
        (if present), and mask (if present). It also applies any region-based
        masking.

        The method updates the following for each plane (xy, xz, zy):
        - Image slice
        - Segmentation slice (if segmentation is present)
        - Mask slice (combining input mask and region-based mask if applicable)

        It also handles the 'hide_masked' option and updates the image display
        range (vmin, vmax) if not explicitly set.

        This method is typically called internally when the viewer needs to be
        updated, such as when the reference point changes. It does not return
        anything but updates the internal state of the OrthogonalViewer object.
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

        #Mask and region
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

        if self._region is not None:
            x = ox + hx*np.arange(nx)
            y = oy + hy*np.arange(ny)
            z = oz + hz*np.arange(nz)
            func = self._region._contains_point
            X, Y = np.meshgrid(x, y)
            Z = 0*X + self.ref_point[2]
            _region_mask_slice_cpu(self._slices['xy']['mask'].T, func, X, Y, Z)
            #xz
            X, Z = np.meshgrid(x, z)
            Y = 0*X + self.ref_point[1]
            _region_mask_slice_cpu(self._slices['xz']['mask'].T, func, X, Y, Z)
            #zy
            Y, Z = np.meshgrid(y, z)
            X = 0*Z + self.ref_point[0]
            _region_mask_slice_cpu(self._slices['zy']['mask'].T, func, X, Y, Z)

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

        if self._hide_masked and not (self._mask is None and self._region is None):
            for plane in ('xy', 'xz', 'zy'):
                self._slices[plane]['image'] = np.ma.array(
                    self._slices[plane]['image'], mask=self._slices[plane]['mask'])
                if self._segmentation is not None:
                    self._slices[plane]['segmentation'] = np.ma.array(
                        self._slices[plane]['segmentation'], mask=self._slices[plane]['mask'])
                if self._mask is not None:
                    self._slices[plane]['mask'] = np.ma.array(
                        self._slices[plane]['mask'], mask=self._slices[plane]['mask'])

        if ('vmin' not in self._image_dict or self._image_dict['vmin'] is None
            or 'vmax' not in self._image_dict or self._image_dict['vmax'] is None):
            temp = np.concatenate((self._slices['xy']['image'].flatten(),
                                   self._slices['xz']['image'].flatten(),
                                   self._slices['zy']['image'].flatten()))
            if 'vmin' not in self._image_dict or self._image_dict['vmin'] is None:
                self._image_dict['vmin'] = temp.min()
            if 'vmax' not in self._image_dict or self._image_dict['vmax'] is None:
                self._image_dict['vmax'] = temp.max()


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

        The method uses various dictionaries (self._image_dict,
        self._segmentation_dict, etc.)  to customize the appearance of
        different plot elements.

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
                                                   extent=extent,
                                                   **self._image_dict)
        if self._slices[plane]['segmentation'] is not None:
            self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'].T,
                                                       extent=extent,
                                                       vmin=min(self._histogram._phases),
                                                       vmax=max(self._histogram._phases),
                                                       **self._segmentation_dict)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'].T,
                                                         extent=extent,
                                                         vmin=0,
                                                         vmax=1,
                                                         **self._mask_dict)
        ref_point = self.ref_point
        self._slices[plane]['vline'] = axi.axvline(ref_point[0], **self._guide_line_dict)
        self._slices[plane]['hline'] = axi.axhline(ref_point[1], **self._guide_line_dict)
        axi.set_xlim(extent[0], extent[1])
        axi.set_ylim(extent[2], extent[3])
        axi.set_xlabel(f'{plane[0]} ({voxel_unit})')
        axi.set_ylabel(f'{plane[1]} ({voxel_unit})')
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
                                                   extent=extent,
                                                   **self._image_dict)
        if self._slices[plane]['segmentation'] is not None:
            self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'],
                                                       extent=extent,
                                                       vmin=min(self._histogram._phases),
                                                       vmax=max(self._histogram._phases),
                                                       **self._segmentation_dict)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'],
                                                         extent=extent,
                                                         vmin=0,
                                                         vmax=1,
                                                         **self._mask_dict)
        ref_point = self.ref_point
        self._slices[plane]['vline'] = axi.axvline(ref_point[2], **self._guide_line_dict)
        self._slices[plane]['hline'] = axi.axhline(ref_point[1], **self._guide_line_dict)
        axi.set_xlim(extent[0], extent[1])
        axi.set_ylim(extent[2], extent[3])
        axi.set_xlabel(f'{plane[0]} ({voxel_unit})')
        axi.set_ylabel(f'{plane[1]} ({voxel_unit})')
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
                                                   extent=extent,
                                                   **self._image_dict)
        if self._slices[plane]['segmentation'] is not None:
            self._slices[plane]['mpl_seg'] = axi.imshow(self._slices[plane]['segmentation'].T,
                                                       extent=extent,
                                                       vmin=min(self._histogram._phases),
                                                       vmax=max(self._histogram._phases),
                                                       **self._segmentation_dict)
        if self._slices[plane]['mask'] is not None:
            self._slices[plane]['mpl_mask'] = axi.imshow(self._slices[plane]['mask'].T,
                                                         extent=extent,
                                                         vmin=0,
                                                         vmax=1,
                                                         **self._mask_dict)
        ref_point = self.ref_point
        self._slices[plane]['vline'] = axi.axvline(ref_point[0], **self._guide_line_dict)
        self._slices[plane]['hline'] = axi.axhline(ref_point[2], **self._guide_line_dict)
        axi.set_xlim(extent[0], extent[1])
        axi.set_ylim(extent[2], extent[3])
        axi.set_xlabel(f'{plane[0]} ({voxel_unit})')
        axi.set_ylabel(f'{plane[1]} ({voxel_unit})')
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

            axi.set_facecolor(self._background_color)

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
        ax.cla()
        x = self.histogram.bin_centers
        phases = set(self.histogram.count.columns) - set(['full',])
        if len(phases)>0:
            for k in sorted(phases):
                ax.plot(x, self.histogram.count[k])
        ax.plot(x, self.histogram.count['full'], 'k--')
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_ylabel('Count')
        xlabel = self.image.field_name.strip()
        if not xlabel:
            xlabel = '???'
        unit = self.image.field_unit.strip()
        if unit:
            xlabel = f'{xlabel} ({unit})'
        ax.set_xlabel(xlabel)
        self._slices['histogram']['leftline'] = ax.axvline(self._image_dict['vmin'])
        self._slices['histogram']['rightline'] = ax.axvline(self._image_dict['vmax'])
        ax.grid(True, alpha=0.5)


    def _build_from_scratch(self):
        """
        Call the various building methods to build the image from  scratch.
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
            self._fig.show()


    def _update_plots(self, new=False):
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

        if self._segmentation is not None:
            self._slices['xy']['mpl_seg'].set_data(self._slices['xy']['segmentation'].T)
            self._slices['zy']['mpl_seg'].set_data(self._slices['zy']['segmentation'])
            self._slices['xz']['mpl_seg'].set_data(self._slices['xz']['segmentation'].T)

        if self._mask is not None:
            self._slices['xy']['mpl_mask'].set_data(self._slices['xy']['mask'].T)
            self._slices['zy']['mpl_mask'].set_data(self._slices['zy']['mask'])
            self._slices['xz']['mpl_mask'].set_data(self._slices['xz']['mask'].T)

        self._slices['xy']['vline'].set_xdata([ref_point[0], ref_point[0]])
        self._slices['xy']['hline'].set_ydata([ref_point[1], ref_point[1]])

        self._slices['zy']['vline'].set_xdata([ref_point[2], ref_point[2]])
        self._slices['zy']['hline'].set_ydata([ref_point[1], ref_point[1]])

        self._slices['xz']['vline'].set_xdata([ref_point[0], ref_point[0]])
        self._slices['xz']['hline'].set_ydata([ref_point[2], ref_point[2]])

        for plane in ('xy', 'zy', 'xz'):
            self._slices[plane]['mpl_im'].set_clim(vmin=self._image_dict['vmin'],
                                                   vmax=self._image_dict['vmax'])
            if self._segmentation is not None:
                self._slices[plane]['mpl_seg'].set_clim(vmin=min(self._histogram._phases),
                                                        vmax=max(self._histogram._phases))
            if self._mask is not None:
                self._slices[plane]['mpl_mask'].set_clim(vmin=0, vmax=1)

        self._slices['histogram']['leftline'].set_xdata((self._image_dict['vmin'], self._image_dict['vmin']))
        self._slices['histogram']['rightline'].set_xdata((self._image_dict['vmax'], self._image_dict['vmax']))

        self._fig.canvas.draw()
        self._fig.show()

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


    def refresh(self):
        """
        Refresh the OrthogonalViewer display.

        This method forces the display refresh of to reflect any changes in the
        image data, reference point, or display settings. It can be called
        after changing arrays or dictionaries that do not automatically trigger
        the display update. It ensures that all visual elements are in sync
        with the current state of the viewer.
        """
        if mpi_rank == self._mpi_proc:
            self._build_from_scratch()

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
            >>> from drp.regions.cylinder import Cylinder
            >>> viewer = drp.plot.OrthogonalViewer(<your parameters here...>)
            >>> region = viewer.region                               # Get the current region
            >>> viewer.region = Cylinder(<your parameters here...>)  # Set a new region
            >>> viewer.region = None                                 # Remove the region
        '''
        return self._region

    @region.setter
    def region(self, v):
        if v is not None:
            _assert.instance('region', v, 'Region', (Region,))
        self._region = v
        self._histogram.region = v
        self._build_from_scratch()


    @property
    def mask(self):
        """
        Get or set the mask image.

        Examples
        --------
            >>> viewer = drp.plot.OrthogonalViewer(<your parameters here...>)
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
        self._build_from_scratch()


    @property
    def segmentation(self):
        """
        Get or set the segmentation image.

        Examples
        --------
            >>> viewer = drp.plot.OrthogonalViewer(<your parameters here...>)
            >>> segmentation = viewer.segmentation     # Get the current segmentation
            >>> viewer.segmentation = new_segmentation # Set a new array for segmentation
            >>> viewer.segmentation = None             # Remove the segmentation
        """
        return self._segmentation


    @segmentation.setter
    def segmentation(self, v):
        self._image.check_mask_and_segmentation(segmentation=v)
        self._segmentation = v
        self._histogram.segmentation = v
        self._build_from_scratch()

    @property
    def histogram(self):
        '''
        The :class:`drp.Histogram` object.
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
           >>> viewer = drp.plot.OrthogonalViewer(<your parameters here...>)
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
        must be an iterable (x, y, z), where ox <= x < ox+hx*nx,
        oy <= y < oy+hy*ny, and oz <= z < oz+hz*nz, with
        ox, oy, oz = image.voxel_origin,
        hx, hy, hz = image.voxel_length, and
        nx, ny, nz = image.shape.
        If the point (x, y, z) is not a grid point, the closest grid point will
        be used.

        Examples
        --------
           >>> viewer = drp.plot.OrthogonalViewer(<your parameters here...>)
           >>> ref_point = viewer.ref_voxel #get the current reference voxel
           >>> viewer.ref_point = (3.33, 14.72, 10) # Set new reference point and update
        """
        point = np.array(self._ref_voxel).astype(float)
        point *= np.array(self._image.voxel_length).astype(float)
        point += np.array(self._image.voxel_origin).astype(float)
        return tuple(point)


    @ref_point.setter
    def ref_point(self, v):
        if v is None:
            self._ref_voxel = np.floor(np.array(self._image.shape)/2).astype(int)
        else:
            _assert.iterable.ordered_numbers('ref_point', v)
            _assert.iterable.length('ref_point', v, 3)
            nx, ny, nz = self._image.shape
            point = np.array(v).astype(float)
            point -= np.array(self._image.voxel_origin).astype(float)
            point /= np.array(self._image.voxel_length).astype(float)
            rx = int(round(point[0]))
            rx = 0 if rx < 0 else rx
            rx = nx-1 if rx >= nx else rx
            ry = int(round(point[1]))
            ry = 0 if ry < 0 else ry
            ry = ny-1 if ry >= ny else ry
            rz = int(round(point[2]))
            rz = 0 if rz < 0 else rz
            rz = nz-1 if rz >= nz else rz
            self._ref_voxel = np.array((rx, ry, rz)).astype(int)
        self._update_plots()

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
            self.refresh()

    @property
    def hide_masked(self):
        '''
        Boolean. If True, hide the masked or outside region voxels.
        Default is False.
        '''
        return self._hide_masked

    @hide_masked.setter
    def hide_masked(self, v):
        _assert.instance('hide_masked', v, 'boolean', (bool,))
        if self._hide_masked != v:
            self._hide_masked = v
            self.refresh()

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
            self.refresh()

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
        self.refresh()

    def reset_image_dict(self):
        """
        Reset the image display settings to default values and refreshes the display.
        """
        self._image_dict = {**IMAGE_DICT}
        self.refresh()


    @property
    def segmentation_dict(self):
        """
        Dictionary of keyword arguments for customizing the segmentation display.

        This dictionary is passed to Matplotlib's imshow function when
        displaying the segmentation overlay. It can be used to control aspects
        such as colormap, alpha (transparency), etc. See the documentation for
        Matplotlib imshow function. Changing elements directly in this dictionary
        requires calling the update function for the changes to take effect.

        See Also
        --------
        update_segmentation_dict : Method to update this dictionary.
        reset_segmentation_dict : Method to reset this dictionary to default values.
        """
        return self._segmentation_dict

    def update_segmentation_dict(self, **kwargs):
        """
        Update the segmentation display settings dictionary and refreshes the display.
        See the documentation for the segmentation_dict method.
        """
        self._segmentation_dict.update(**kwargs)
        self.refresh()

    def reset_segmentation_dict(self):
        """
        Reset the segmentation display settings to default values and refreshes the display.
        """
        self._segmentation_dict = {**SEGMENTATION_DICT}
        self.refresh()


    @property
    def mask_dict(self):
        """
        Dictionary of keyword arguments for customizing the mask display.

        This dictionary is passed to Matplotlib's imshow function when
        displaying the mask+region overlay. It can be used to control aspects
        such as colormap, alpha (transparency), etc. See the documentation for
        Matplotlib imshow function. Changing elements directly in this dictionary
        requires calling the update function for the changes to take effect.

        See Also
        --------
        update_mask_dict : Method to update this dictionary.
        reset_mask_dict : Method to reset this dictionary to default values.
        """
        return self._mask_dict

    def update_mask_dict(self, **kwargs):
        """
        Update the mask display settings dictionary and refreshes the display.
        See the documentation for the mask_dict method.
        """
        self._mask_dict.update(**kwargs)
        self.refresh()

    def reset_mask_dict(self):
        """
        Reset the mask display settings to default values and refreshes the display.
        """
        self._mask_dict = {**MASK_DICT}
        self.refresh()


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
        self.refresh()

    def reset_guide_line_dict(self):
        """
        Reset the guide line display settings to default values and refreshes the display.
        """
        self._guide_line_dict = {**GUIDE_LINE_DICT}
        self.refresh()

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
        self._build_from_scratch()

    def reset_gridspec_dict(self):
        """
        Reset the guide gridspec settings to default values and refreshes the display.
        """
        self._gridspec_dict = {}
        self._build_from_scratch()


    @property
    def background_color(self):
        """
        Get or set the background color of the plot areas.

        This color is particularly visible when hide_masked is True. It can be
        any color format accepted by Matplotlib (e.g., 'w' for white, 'k' for
        black, '#RRGGBB' for RGB hex code).

        See Also
        --------
        hide_masked : Property to control visibility of masked areas.
        """
        return self._background_color

    @background_color.setter
    def background_color(self, v):
        _assert.instance('background_color', v, 'string', (str,))
        self._background_color = v
        self.refresh()

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
        plane = 'xy'
        i, j, k, msg = self._get_ijk_xyz(xo, yo, plane)
        label = 'v'
        value = f"{self._slices[plane]['image'][i, j]:1.2f}"
        if self.segmentation is not None:
            label = label + ', s'
            value = f"{value}, {self._slices[plane]['segmentation'][i, j]:d}"
        if self.mask is not None:
            label = label + ', m'
            value = f"{value}, {self._slices[plane]['mask'][i, j]}"
        left = "(" if len(label)>1 else ""
        right = ")" if len(label)>1 else ""
        return f"{msg}, {left}{label}{right}: {left}{value}{right}"

    def _format_coord_xz(self, xo, yo):
        plane = 'xz'
        i, j, k, msg = self._get_ijk_xyz(xo, yo, plane)
        label = 'v'
        value = f"{self._slices[plane]['image'][i, k]:1.2f}"
        if self.segmentation is not None:
            label = label + ', s'
            value = f"{value}, {self._slices[plane]['segmentation'][i, k]:d}"
        if self.mask is not None:
            label = label + ', m'
            value = f"{value}, {self._slices[plane]['mask'][i, k]}"
        left = "(" if len(label)>1 else ""
        right = ")" if len(label)>1 else ""
        return f"{msg}, {left}{label}{right}: {left}{value}{right}"

    def _format_coord_zy(self, xo, yo):
        plane = 'zy'
        i, j, k, msg = self._get_ijk_xyz(xo, yo, plane)
        label = 'v'
        value = f"{self._slices[plane]['image'][j, k]:1.2f}"
        if self.segmentation is not None:
            label = label + ', s'
            value = f"{value}, {self._slices[plane]['segmentation'][j, k]:d}"
        if self.mask is not None:
            label = label + ', m'
            value = f"{value}, {self._slices[plane]['mask'][j, k]}"
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
        vmin = self.image_dict['vmin']
        vmax = self.image_dict['vmax']

        if event.inaxes == self._slices['histogram']['ax']:
            if event.button == MouseButton.LEFT:
                vmin = max(x, self.histogram.percentile(0))
                if vmin < self.image_dict['vmax']:
                    self.image_dict.update(vmin=vmin)
                    self._update_plots()
            if event.button == MouseButton.RIGHT:
                vmax = min(x, self.histogram.percentile(100))
                if vmax > self.image_dict['vmin']:
                    self.image_dict.update(vmax=vmax)
                    self._update_plots()

        if event.inaxes == self._slices['xy']['ax']:
            ref_point[0] = x
            ref_point[1] = y
            if event.button == MouseButton.RIGHT:
                self.ref_point = ref_point
            if event.button == MouseButton.LEFT:
                print_data(self, ref_point)

        if event.inaxes == self._slices['xz']['ax']:
            ref_point[0] = x
            ref_point[2] = y
            if event.button == MouseButton.RIGHT:
                self.ref_point = ref_point
            if event.button == MouseButton.LEFT:
                print_data(self, ref_point)

        if event.inaxes == self._slices['zy']['ax']:
            ref_point[2] = x
            ref_point[1] = y
            if event.button == MouseButton.RIGHT:
                self.ref_point = ref_point
            if event.button == MouseButton.LEFT:
                print_data(self, ref_point)


    def _on_scroll(self, event):
        """
        Handle scroll events to navigate through image slices.
        """
        nx, ny, nz = self._image.shape
        hx, hy, hz = self._image.voxel_length
        ox, oy, oz = self._image.voxel_origin
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


#Test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')
    import rockverse as rv
    #%matplotlib qt
    try:
        belgian_fieldstone_data.info
    except Exception:
        belgian_fieldstone_data = rv.digitalrock.voxel_image.import_raw(
            #rawfile='/MyDownloads/Fieldstone_1000x1000x861_16b.raw', #<- Original file path
            rawfile=r'C:\Users\GOB7\Downloads\Rocha digital\Belgian Fieldstone\Fieldstone_1000x1000x861_16b.raw',
            store=None,#'/estgf_dados/P_D/GOB7/BelgianFieldstone/src.zarr', #<- path where to put the voxel image
            shape=(1000, 1000, 861),         #<- From metadata
            dtype='>u2',                     #<- From metadata, big-endian 16-bit unsigned integer
            offset=0,                        #<- From metadata
            voxel_length=(4.98, 4.98, 4.98), #<- From metadata
            voxel_unit='um',                 #<- From metadata
            raw_file_order='F',              #<- Fortran file order
            chunks=(500, 500, 431),          #<- Our choice of chunk size
            overwrite=True,                  #<- Overwrite if file exists in disk
            field_name='Attenuation',
            )
    self=OrthogonalViewer(belgian_fieldstone_data)
    #self.statusbar_mode='voxel'

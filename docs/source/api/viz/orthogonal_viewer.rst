.. _OrthogonalViewer class:

rockverse.viz.OrthogonalViewer
==============================

.. currentmodule:: rockverse.viz

.. autoclass:: OrthogonalViewer

Matplotlib-related attributes
-----------------------------

.. autosummary::
    :toctree: _autogen

    ~OrthogonalViewer.figure
    ~OrthogonalViewer.ax_xy
    ~OrthogonalViewer.ax_zy
    ~OrthogonalViewer.ax_xz
    ~OrthogonalViewer.ax_histogram

RockVerse objects
-----------------

.. autosummary::
    :toctree: _autogen

    ~OrthogonalViewer.image
    ~OrthogonalViewer.region
    ~OrthogonalViewer.mask
    ~OrthogonalViewer.segmentation
    ~OrthogonalViewer.histogram

Axes building attributes
------------------------

.. autosummary::
    :toctree: _autogen

    ~OrthogonalViewer.show_xy_plane
    ~OrthogonalViewer.show_xz_plane
    ~OrthogonalViewer.show_zy_plane
    ~OrthogonalViewer.show_histogram
    ~OrthogonalViewer.gridspec_dict
    ~OrthogonalViewer.update_gridspec_dict
    ~OrthogonalViewer.layout

View customization attributes and methods
-----------------------------------------

.. autosummary::
    :toctree: _autogen

    ~OrthogonalViewer.ref_voxel
    ~OrthogonalViewer.ref_point
    ~OrthogonalViewer.image_dict
    ~OrthogonalViewer.update_image_dict
    ~OrthogonalViewer.show_guide_lines
    ~OrthogonalViewer.guide_line_dict
    ~OrthogonalViewer.update_guide_line_dict
    ~OrthogonalViewer.segmentation_colors
    ~OrthogonalViewer.segmentation_colormap
    ~OrthogonalViewer.segmentation_alpha
    ~OrthogonalViewer.mask_color
    ~OrthogonalViewer.mask_alpha
    ~OrthogonalViewer.histogram_bins
    ~OrthogonalViewer.histogram_line_dict
    ~OrthogonalViewer.update_histogram_line_dict
    ~OrthogonalViewer.histogram_lines
    ~OrthogonalViewer.hide_axis
    ~OrthogonalViewer.shrink_to_fit
    ~OrthogonalViewer.expand_to_fit
    ~OrthogonalViewer.statusbar_mode


Interactive Features
====================

The ``OrthogonalViewer`` class offers interactive features that enhance the user
experience by allowing real-time updates and interactions with the visualized
data, including the displayed slices and histograms.

.. note::
    To utilize the interactive features, ensure that Matplotlib is configured
    for interactive mode. You can use backends such as `Qt5Agg` or `ipympl`
    for Jupyter Notebooks.

Users can interact with the visualization through mouse actions on the image
slices or the histogram. The following actions are supported:

+-------------------------------+---------------------------------------------+
| Left click on image slices    | Displays voxel data at the cursor position  |
|                               | in the console.                             |
+-------------------------------+---------------------------------------------+
| Right click on image slices   | Updates the reference point to the cursor   |
|                               | position, allowing users to change the      |
|                               | displayed slices.                           |
+-------------------------------+---------------------------------------------+
| Middle click on image slices  | Resets the reference point to the center of |
|                               | the image bounding box.                     |
+-------------------------------+---------------------------------------------+
| Left click on histogram       | Sets the image minimum color limit (clim)   |
|                               | value to the horizontal coordinate of the   |
|                               | click.                                      |
+-------------------------------+---------------------------------------------+
| Right click on histogram      | Sets the image maximum color limit (clim)   |
|                               | value to the horizontal coordinate of the   |
|                               | click.                                      |
+-------------------------------+---------------------------------------------+
| Middle click on histogram     | Resets the image color limit (clim) values  |
|                               | to the 99.9% confidence interval based on   |
|                               | the histogram data.                         |
+-------------------------------+---------------------------------------------+


Related tutorials
=================

.. nblinkgallery::

   ../../tutorials/digitalrock/orthogonal_viewer/orthogonal_viewer
   ../../tutorials/digitalrock/dual_energy/dual_energy_tutorial_prepare_data

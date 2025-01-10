.. _OrthogonalViewer class:

The OrthogonalViewer class
--------------------------

.. currentmodule:: rockverse.viz

.. autoclass:: OrthogonalViewer

   .. _OrthogonalViewer matplotlib attributes:

   .. rubric:: Matplotlib-related attributes

   .. autosummary::
      ~OrthogonalViewer.figure
      ~OrthogonalViewer.ax_xy
      ~OrthogonalViewer.ax_zy
      ~OrthogonalViewer.ax_xz
      ~OrthogonalViewer.ax_histogram

   .. rubric:: RockVerse objects

   .. autosummary::
      ~OrthogonalViewer.image
      ~OrthogonalViewer.region
      ~OrthogonalViewer.mask
      ~OrthogonalViewer.segmentation
      ~OrthogonalViewer.histogram

   .. rubric:: Axes building attributes

   .. autosummary::
      ~OrthogonalViewer.show_xy_plane
      ~OrthogonalViewer.show_xz_plane
      ~OrthogonalViewer.show_zy_plane
      ~OrthogonalViewer.show_histogram
      ~OrthogonalViewer.gridspec_dict
      ~OrthogonalViewer.update_gridspec_dict
      ~OrthogonalViewer.layout

   .. rubric:: View customization attributes and methods

   .. autosummary::
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

   .. autoattribute:: figure
   .. autoattribute:: ax_xy
   .. autoattribute:: ax_zy
   .. autoattribute:: ax_xz
   .. autoattribute:: ax_histogram

   .. autoattribute:: image
   .. autoattribute:: region
   .. autoattribute:: mask
   .. autoattribute:: segmentation
   .. autoattribute:: histogram

   .. autoattribute:: show_xy_plane
   .. autoattribute:: show_xz_plane
   .. autoattribute:: show_zy_plane
   .. autoattribute:: show_histogram
   .. autoattribute:: gridspec_dict
   .. automethod:: update_gridspec_dict
   .. autoattribute:: layout

   .. autoattribute:: ref_voxel
   .. autoattribute:: ref_point
   .. autoattribute:: image_dict
   .. automethod:: update_image_dict
   .. autoattribute:: show_guide_lines
   .. autoattribute:: guide_line_dict
   .. automethod:: update_guide_line_dict
   .. autoattribute:: segmentation_colors
   .. autoattribute:: segmentation_colormap
   .. autoattribute:: segmentation_alpha
   .. autoattribute:: mask_color
   .. autoattribute:: mask_alpha
   .. autoattribute:: histogram_bins
   .. autoattribute:: histogram_line_dict
   .. automethod:: update_histogram_line_dict
   .. autoattribute:: histogram_lines
   .. autoattribute:: hide_axis
   .. automethod:: shrink_to_fit
   .. automethod:: expand_to_fit
   .. autoattribute:: statusbar_mode

Interactive Features
^^^^^^^^^^^^^^^^^^^^

The ``OrthogonalViewer`` class offers interactive features that enhance the user
experience by allowing real-time updates and interactions with the visualized
data, including the displayed slices and histograms.

.. note::
    To utilize the interactive features, ensure that Matplotlib is configured
    for interactive mode. You can use backends such as `Qt5Agg` or `ipympl`
    for Jupyter Notebooks.

Users can interact with the visualization through mouse actions on the image
slices or the histogram. The following actions are supported:

- **Left click on image slices:** Displays voxel data at the cursor position in
  the console.
- **Right click on image slices:** Updates the reference point to the cursor
  position, allowing users to change the displayed slices.
- **Middle click on image slices:** Resets the reference point to the center of
  the image bounding box.
- **Left click on the histogram:** Sets the image minimum color limit (clim)
  value to the horizontal coordinate of the click.
- **Right click on the histogram:** Sets the image maximum color limit (clim)
  value to the horizontal coordinate of the click.
- **Middle click on the histogram:** Resets the image color limit (clim) values
  to the 99.9% confidence interval based on the histogram data.


Related tutorials
^^^^^^^^^^^^^^^^^

.. nbgallery::

   ../../tutorials/digitalrock/orthogonal_viewer

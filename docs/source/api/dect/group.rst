========================
rockverse.dect.DECTGroup
========================

.. currentmodule:: rockverse.dect

.. autoclass:: DECTGroup

Attributes
----------

Voxel images
~~~~~~~~~~~~

.. autosummary::
   :toctree: _autogen

   ~DECTGroup.lowECT
   ~DECTGroup.highECT
   ~DECTGroup.mask
   ~DECTGroup.segmentation
   ~DECTGroup.rho_min
   ~DECTGroup.rho_p25
   ~DECTGroup.rho_p50
   ~DECTGroup.rho_p75
   ~DECTGroup.rho_max
   ~DECTGroup.Z_min
   ~DECTGroup.Z_p25
   ~DECTGroup.Z_p50
   ~DECTGroup.Z_p75
   ~DECTGroup.Z_max
   ~DECTGroup.valid

Calibration materials
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autogen

   ~DECTGroup.calibration_material
   ~DECTGroup.periodic_table

Inversion parameters
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autogen

   ~DECTGroup.maxA
   ~DECTGroup.maxB
   ~DECTGroup.maxn
   ~DECTGroup.lowE_inversion_coefficients
   ~DECTGroup.highE_inversion_coefficients
   ~DECTGroup.tol
   ~DECTGroup.whis
   ~DECTGroup.required_iterations
   ~DECTGroup.maximum_iterations
   ~DECTGroup.threads_per_block


Methods
-------

.. _dect_array_creation:

Handling voxel images
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autogen

   ~DECTGroup.copy_image
   ~DECTGroup.create_mask
   ~DECTGroup.delete_mask
   ~DECTGroup.create_segmentation
   ~DECTGroup.delete_segmentation

Monte Carlo Inversion
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autogen

   ~DECTGroup.check
   ~DECTGroup.preprocess
   ~DECTGroup.run

Visualization
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autogen

   ~DECTGroup.view_pdfs
   ~DECTGroup.view_inversion_coefs
   ~DECTGroup.view_inversion_Zeff

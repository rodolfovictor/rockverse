========================================
rockverse.dualenergyct.DualEnergyCTGroup
========================================

.. currentmodule:: rockverse.dualenergyct

.. autoclass:: DualEnergyCTGroup

Attributes
----------

Voxel images
~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autogen

   ~DualEnergyCTGroup.lowECT
   ~DualEnergyCTGroup.highECT
   ~DualEnergyCTGroup.mask
   ~DualEnergyCTGroup.segmentation
   ~DualEnergyCTGroup.rho_min
   ~DualEnergyCTGroup.rho_p25
   ~DualEnergyCTGroup.rho_p50
   ~DualEnergyCTGroup.rho_p75
   ~DualEnergyCTGroup.rho_max
   ~DualEnergyCTGroup.Z_min
   ~DualEnergyCTGroup.Z_p25
   ~DualEnergyCTGroup.Z_p50
   ~DualEnergyCTGroup.Z_p75
   ~DualEnergyCTGroup.Z_max
   ~DualEnergyCTGroup.valid

Calibration materials
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autogen

   ~DualEnergyCTGroup.calibration_material
   ~DualEnergyCTGroup.periodic_table

Inversion parameters
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autogen

   ~DualEnergyCTGroup.maxA
   ~DualEnergyCTGroup.maxB
   ~DualEnergyCTGroup.maxn
   ~DualEnergyCTGroup.lowE_inversion_coefficients
   ~DualEnergyCTGroup.highE_inversion_coefficients
   ~DualEnergyCTGroup.tol
   ~DualEnergyCTGroup.whis
   ~DualEnergyCTGroup.required_iterations
   ~DualEnergyCTGroup.maximum_iterations
   ~DualEnergyCTGroup.threads_per_block


Methods
-------

.. _dect_array_creation:

Handling voxel images
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autogen

   ~DualEnergyCTGroup.copy_image
   ~DualEnergyCTGroup.create_mask
   ~DualEnergyCTGroup.delete_mask
   ~DualEnergyCTGroup.create_segmentation
   ~DualEnergyCTGroup.delete_segmentation

Monte Carlo Inversion
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../_autogen

   ~DualEnergyCTGroup.check
   ~DualEnergyCTGroup.preprocess
   ~DualEnergyCTGroup.run

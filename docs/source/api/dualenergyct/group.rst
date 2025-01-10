The Group class
---------------

.. currentmodule:: rockverse.dualenergyct

.. autoclass:: DualEnergyCTGroup

Attribute summary
^^^^^^^^^^^^^^^^^

.. rubric:: Voxel image

.. autosummary::
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

.. rubric:: Histograms

.. autosummary::
   ~DualEnergyCTGroup.histogram_bins
   ~DualEnergyCTGroup.lowEhistogram
   ~DualEnergyCTGroup.highEhistogram

.. rubric:: Calibration parameters

.. autosummary::
   ~DualEnergyCTGroup.calibration_material0
   ~DualEnergyCTGroup.calibration_material1
   ~DualEnergyCTGroup.calibration_material2
   ~DualEnergyCTGroup.calibration_material3
   ~DualEnergyCTGroup.calibration_gaussian_coefficients
   ~DualEnergyCTGroup.periodic_table
   ~DualEnergyCTGroup.maxA
   ~DualEnergyCTGroup.maxB
   ~DualEnergyCTGroup.maxn
   ~DualEnergyCTGroup.lowE_inversion_coefficients
   ~DualEnergyCTGroup.highE_inversion_coefficients

.. rubric:: Inversion parameters

.. autosummary::
   ~DualEnergyCTGroup.tol
   ~DualEnergyCTGroup.whis
   ~DualEnergyCTGroup.required_iterations
   ~DualEnergyCTGroup.maximum_iterations
   ~DualEnergyCTGroup.threads_per_block
   ~DualEnergyCTGroup.hash_buffer_size

Attributes
^^^^^^^^^^

.. autoattribute:: DualEnergyCTGroup.calibration_gaussian_coefficients
.. autoattribute:: DualEnergyCTGroup.calibration_material0
.. autoattribute:: DualEnergyCTGroup.calibration_material1
.. autoattribute:: DualEnergyCTGroup.calibration_material2
.. autoattribute:: DualEnergyCTGroup.calibration_material3
.. autoattribute:: DualEnergyCTGroup.hash_buffer_size
.. autoattribute:: DualEnergyCTGroup.highECT
.. autoattribute:: DualEnergyCTGroup.highE_inversion_coefficients
.. autoattribute:: DualEnergyCTGroup.highEhistogram
.. autoattribute:: DualEnergyCTGroup.histogram_bins
.. autoattribute:: DualEnergyCTGroup.lowECT
.. autoattribute:: DualEnergyCTGroup.lowEhistogram
.. autoattribute:: DualEnergyCTGroup.lowE_inversion_coefficients
.. autoattribute:: DualEnergyCTGroup.mask
.. autoattribute:: DualEnergyCTGroup.maxA
.. autoattribute:: DualEnergyCTGroup.maxB
.. autoattribute:: DualEnergyCTGroup.maximum_iterations
.. autoattribute:: DualEnergyCTGroup.maxn
.. autoattribute:: DualEnergyCTGroup.periodic_table
.. autoattribute:: DualEnergyCTGroup.required_iterations
.. autoattribute:: DualEnergyCTGroup.rho_max
.. autoattribute:: DualEnergyCTGroup.rho_min
.. autoattribute:: DualEnergyCTGroup.rho_p25
.. autoattribute:: DualEnergyCTGroup.rho_p50
.. autoattribute:: DualEnergyCTGroup.rho_p75
.. autoattribute:: DualEnergyCTGroup.segmentation
.. autoattribute:: DualEnergyCTGroup.threads_per_block
.. autoattribute:: DualEnergyCTGroup.tol
.. autoattribute:: DualEnergyCTGroup.valid
.. autoattribute:: DualEnergyCTGroup.whis
.. autoattribute:: DualEnergyCTGroup.Z_max
.. autoattribute:: DualEnergyCTGroup.Z_min
.. autoattribute:: DualEnergyCTGroup.Z_p25
.. autoattribute:: DualEnergyCTGroup.Z_p50
.. autoattribute:: DualEnergyCTGroup.Z_p75

Methods summary
^^^^^^^^^^^^^^^

.. _dect_array_creation:

.. rubric:: Handling arrays

.. autosummary::
   ~DualEnergyCTGroup.copy_image
   ~DualEnergyCTGroup.create_mask
   ~DualEnergyCTGroup.delete_mask
   ~DualEnergyCTGroup.create_segmentation

.. rubric:: Monte Carlo Inversion

.. autosummary::
   ~DualEnergyCTGroup.check
   ~DualEnergyCTGroup.preprocess
   ~DualEnergyCTGroup.run


Methods
^^^^^^^

.. automethod:: DualEnergyCTGroup.copy_image
.. automethod:: DualEnergyCTGroup.create_mask
.. automethod:: DualEnergyCTGroup.create_segmentation
.. automethod:: DualEnergyCTGroup.delete_mask
.. automethod:: DualEnergyCTGroup.run

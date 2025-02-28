Monte Carlo Dual Energy CT processing
-------------------------------------

This tutorial demonstrates how to utilize RockVerse to process dual-energy
X-ray computed tomography (CT) data, following the workflow developed by
Victor et al., which introduced a Monte Carlo-based inversion method for
estimating electron density and effective atomic number.

In this tutorial, we will cover the steps to prepare and process dual-energy
CT data using RockVerse, including data download, preparation, and processing.
By the end of this tutorial, you will be able to apply the Monte Carlo method
to your own dual-energy CT data.

Details about the method are publicly available in the following references:

- Victor, Rodolfo Araujo. Multiscale, image-based interpretation
  of well logs acquired in a complex, deepwater carbonate reservoir.
  Diss. 2017. Chapter 3.
  `doi:10.15781/T2XP6VK50 <https://doi.org/10.15781/T2XP6VK50>`_

- Victor, Rodolfo A., Maša Prodanović, and Carlos Torres-Verdín.
  "Monte Carlo approach for estimating density and atomic number from
  dual-energy computed tomography images of carbonate rocks."
  Journal of Geophysical Research: Solid Earth 122.12 (2017): 9804-9824.
  `doi:10.1002/2017JB014408 <https://doi.org/10.1002/2017JB014408>`_

.. toctree::
    :hidden:

    dual_energy/dual_energy_tutorial_download_data.ipynb
    dual_energy/dual_energy_tutorial_prepare_data.ipynb
    dual_energy/dual_energy_tutorial_process.ipynb
    dual_energy/dual_energy_tutorial_performance.ipynb

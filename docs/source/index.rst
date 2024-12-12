.. _rockverse_docs_mainpage:

:html_theme.sidebar_secondary.remove:

.. title:: RockVerse Documentation | Python Tools for Computational Petrophysics

.. image:: _static/RockVerse_logo_model1_for_white_background_facecolor_transparent_True.png
  :class: only-light,
  :align: center
  :width: 400px

.. image:: _static/RockVerse_logo_model1_for_black_background_facecolor_transparent_True.png
  :class: only-dark
  :align: center
  :width: 400px

.. rst-class:: h2 text-center font-weight-light my-4

  Python tools for Computational Petrophysics

.. rst-class:: h5 text-center font-weight-light my-4

  Version |version|

**RockVerse** is an open-source Python library designed to support high-performance
computational petrophysics workflows. It is tailored for researchers and professionals
in digital rock petrophysics, formation evaluation, well logging, and laboratory data
analysis. Built to handle complex, data-intensive tasks, RockVerse provides
high-performance capabilities for large-scale simulations and data analysis.

RockVerse (and this documentation site) is part of an ongoing post-doctoral research project
under active development. We welcome contributions, bug reports, and feature suggestions to
help improve and expand the library. Check back regularly for updates, tutorials, and new
features, or :ref:`join our mailing list <rockverse_docs_maillist>` to stay informed about the
latest developments.

.. rst-class:: h1 text-center font-weight-light my-4

  Key Features

.. dropdown:: High-Performance Parallel Computing
  :animate: fade-in

  Optimized for deployment in high-performance computing (HPC) environments, RockVerse
  supports distributed parallel processing using MPI (Message Passing Interface). It is fully
  compatible with both CPU and GPU architectures, enabling scalable performance across
  diverse systems.

.. dropdown:: Digital Rock Petrophysics (DRP)
  :animate: fade-in

  Efficiently analyze 3D digital rock images using memory-efficient Digital Rock workflows.
  Supports larger than memory datasets.

.. dropdown:: Well Logging and Laboratory Data
  :animate: fade-in

  Process and analyze well log and laboratory data, supporting a wide variety of formats
  and offering streamlined workflows for petrophysical analysis.

.. rst-class:: h1 text-center font-weight-light my-4

  Getting Started

.. grid:: 2
  :gutter: 4

  .. grid-item-card:: Install
    :text-align: center
    :link: rockverse_docs_install
    :link-type: ref

    .. image:: _static/Install_light_background.svg
      :class: only-light,
      :align: center
      :width: 80px

    .. image:: _static/Install_dark_background.svg
      :class: only-dark,
      :align: center
      :width: 80px

    ^^^^^^
    Installation instructions to set up your environment.

  .. grid-item-card::  Gallery
    :text-align: center
    :link: rockverse_docs_gallery
    :link-type: ref

    .. image:: _static/notebook_computer_light_background.svg
      :class: only-light,
      :align: center
      :width: 80px

    .. image:: _static/notebook_computer_dark_background.svg
      :class: only-dark,
      :align: center
      :width: 80px

    ^^^^^^
    Explore real-world examples and Jupyter notebooks showcasing RockVerse workflows.

  .. grid-item-card::  API
    :text-align: center
    :link: rockverse_docs_api
    :link-type: ref

    .. image:: _static/api_light_background.svg
      :class: only-light,
      :align: center
      :width: 80px

    .. image:: _static/api_dark_background.svg
      :class: only-dark,
      :align: center
      :width: 80px

    ^^^^^^
    API references for in-depth understanding of library capabilities.

  .. grid-item-card::  Get help
    :text-align: center
    :link: rockverse_docs_gethelp
    :link-type: ref

    .. image:: _static/gethelp_light_background.svg
      :class: only-light,
      :align: center
      :width: 80px

    .. image:: _static/gethelp_dark_background.svg
      :class: only-dark,
      :align: center
      :width: 80px

    ^^^^^^
    If you encounter issues or have questions, here are the ways to get support.

.. toctree::
    :maxdepth: 1
    :hidden:

    install.rst
    tutorials.rst
    gallery.rst
    api.rst
    gethelp

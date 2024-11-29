# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'rockverse'
copyright = '2024, Rodolfo A. Victor'
author = 'Rodolfo A. Victor'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    "sphinx.ext.viewcode",
    'nbsphinx',
    'sphinx_design',
]

nbsphinx_thumbnails = {
    'notebooks/dual_energy': 'notebooks/sample-data-dual-energy-thumbnail.jpg',
}

autoclass_content = 'class'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "navigation_with_keys": False,
    "logo": {
        "image_light": "_static/logo_mode3_whitebackground_transparencyTrue.png",
        "image_dark": "_static/logo_mode3_blackbackground_transparencyTrue.png",
        "alt_text": "RockVerse documentation - Home",
    },
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rodolfovictor/rockverse",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_context = {
    "github_user": "rodolfovictor",
    "github_repo": "rockverse",
    "github_version": "main",
    "doc_path": "docs",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
]

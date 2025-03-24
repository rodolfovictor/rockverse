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

import os
import rockverse

# -- Project information -----------------------------------------------------

project = 'rockverse'
copyright = '2024-%Y, Rodolfo A. Victor'
author = 'Rodolfo A. Victor'
version = release = rockverse.__version__


# Set the root rst to load. This is required to be named contents to allow
# readthedocs to host the docs using its default configuration.
master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_gallery.load_style',
    'sphinx_design',
    'myst_parser',
]

nbsphinx_thumbnails = {
    'tutorials/miscellaneous/logo': 'tutorials/miscellaneous/rockverse_logo_model1_white.png',
}

autoclass_content = 'class'

language = 'en'

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
html_last_updated_fmt = ""
html_show_sourcelink = False
html_theme_options = {
    "logo": {
        "image_light": "_static/RockVerse_logo_model3_for_white_background_facecolor_transparent_True.png",
        "image_dark": "_static/RockVerse_logo_model3_for_black_background_facecolor_transparent_True.png",
        "alt_text": "RockVerse documentation - Home",
    },
    "footer_center": ["last-updated"],
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/rodolfovictor/rockverse",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "StackOverflow",
            "url": "https://stackoverflow.com/tags/rockverse",
            "icon": "fab fa-stack-overflow",
            "type": "fontawesome",
        },
        {
            "name": "LinkedIn",
            "url": "https://br.linkedin.com/in/rodolfovictor",
            "icon": "fa-brands fa-linkedin",
        },
    ],
}


html_context = {
    "github_user": "rodolfovictor",
    "github_repo": "rockverse",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "light",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
templates_path = ['_templates']
html_css_files = [
    'custom.css',
]

#Hide sidebars from pages
html_sidebars = {
  "install": [],
  "gallery": [],
  "subscribe": [],
}


# Assemble the tutorials
# Each entry: [main rst file in tutorial_folder, thumbnail in thumbs_folder]
# Toctree in tutorials must contain only ipynb
# Use always forward slash as filesep here
tutorial_folder = "tutorials"
thumbs_folder = "_static/tutorial_thumbnails"
tutorials = [
    ['Digital Rock Petrophysics', [
        ['digitalrock/voxel_image.rst', 'voxel_image.png'],
        ['digitalrock/orthogonal_viewer.rst', 'exploring_orthogonal_viewer.png'],
        ['digitalrock/dual_energy.rst', 'Monte_Carlo_Dual_energy_CT_processing.png'],
    ]],
    ['Miscellaneous', [
        ['miscellaneous/logo.ipynb', 'using_logo.png'],
    ]],
    ['Runtime Configuration', [
        ['configuration/selecting_gpu_devices.ipynb', 'configuration-gears.png'],
    ]],
]

tutorials_page = '.. _rockverse_docs_tutorials:\n'
tutorials_page = f'{tutorials_page}\n\n=========\nTutorials\n=========\n'
main_toctree = []
for section in tutorials:
    section_name = section[0]
    tutorials_page = f'{tutorials_page}\n\n\n.. _tutorials_section_{section_name}:\n\n'
    tutorials_page = f'{tutorials_page}{section_name}\n{'='*len(section_name)}\n'
    section_tutorials = section[1]
    #Extract all tutorials in this section
    for tutorial_name, thumbnail in section_tutorials:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                tutorial_folder,
                                tutorial_name)
        main_toctree.append(f"{tutorial_folder}/{tutorial_name}")
        #extract tutorial toctree
        with open(filename, 'r') as file:
            lines = file.readlines()
        toctree = []
        passed = False
        for line in lines:
            if line.strip().startswith('.. toctree::'):
                passed = True
                continue
            if passed and line.strip().endswith('.ipynb'):
                toctree.append(line.strip())
        #Insert section
        tutorials_page = f"{tutorials_page}\n\n"
        tutorials_page = f"{tutorials_page}.. grid:: 2\n"
        tutorials_page = f"{tutorials_page}  :gutter: 0\n"
        tutorials_page = f"{tutorials_page}\n"
        tutorials_page = f"{tutorials_page}  .. grid-item-card::\n"
        tutorials_page = f"{tutorials_page}    :columns: 4\n"
        tutorials_page = f"{tutorials_page}    :shadow: none\n"
        tutorials_page = f"{tutorials_page}\n"
        tutorials_page = f"{tutorials_page}    .. image:: {thumbs_folder}/{thumbnail}\n"
        tutorials_page = f"{tutorials_page}      :align: center\n"
        tutorials_page = f"{tutorials_page}\n"
        tutorials_page = f"{tutorials_page}  .. grid-item-card::  :doc:`{tutorial_folder}/{tutorial_name.replace('.rst', '').replace('.ipynb', '')}`\n"
        tutorials_page = f"{tutorials_page}    :columns: 8\n"
        tutorials_page = f"{tutorials_page}    :shadow: none\n"
        tutorials_page = f"{tutorials_page}    :class-card: tutoriallist\n"
        tutorials_page = f"{tutorials_page}\n"
        for notebook in toctree:
            clean_path = f"{tutorial_folder}/{'/'.join(tutorial_name.split('/')[:-1])}/{notebook}"
            tutorials_page = f"{tutorials_page}    - :doc:`{clean_path.replace('.rst', '').replace('.ipynb', '')}`""\n"
if main_toctree:
    tutorials_page = f"{tutorials_page}\n"
    tutorials_page = f"{tutorials_page}\n"
    tutorials_page = f"{tutorials_page}.. toctree::\n"
    tutorials_page = f"{tutorials_page}  :maxdepth: 2\n"
    tutorials_page = f"{tutorials_page}  :hidden:\n\n"
    for doc in main_toctree:
        tutorials_page = f"{tutorials_page}  {doc.replace('.rst', '').replace('.ipynb', '')}\n"
tutorials_page = f"{tutorials_page}\n\nWell logging\n============\nComming soon!\n"
tutorials_page = f"{tutorials_page}\n\nPetrogeophysics\n===============\nComming soon!\n"
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tutorials_autogen.rst')
with open(filename, 'w') as file:
    print(tutorials_page, file=file)

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('RiVaPy/'))
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'rivapy'
#copyright = '2020, RIVACON GmbH'
#author = 'RIVACON GmbH'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    'nbsphinx',
    "sphinx_gallery.load_style",
    'sphinx.ext.viewcode'
]

intersphinx_mapping ={'pandas': ('https://pandas.pydata.org/docs/', None)}

# Add bibtex file
bibtex_bibfiles = ['refs.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': True
}
html_logo = "img/logo.svg"
html_favicon = 'img/favicon.svg'
github_url = "https://github.com/RIVACON/RiVaPy"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

autoclass_content = 'both'

# thumbnail image paths should be relative to the _static folder
nbsphinx_thumbnails = {
    'notebooks/instruments/repurchase_agreement': "../../source/figs/repo_schema.png"}
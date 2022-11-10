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
# sys.path.insert(0, os.path.abspath(".."))
# sys.path.append(os.path.abspath(
#     os.path.join(__file__, "../../src")
# ))
import re
from pfjax import __version__, __author__

# -- Project information -----------------------------------------------------

project = 'pfjax'
author = __author__
copyright = '2022, ' + author

# The full version, including alpha/beta/rc tags
version = __version__
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    # "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'notebooks/internal']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# A list of CSS files. The entry must be a filename string or a tuple
# containing the filename string and the attributes dictionary. The filename
# must be relative to the html_static_path, or a full URI with scheme like
# https://example.org/style.css. The attributes is used for attributes
# of <link> tag. It defaults to an empty list.
html_css_files = [
    'css/custom.css',
]

# --- Options for autoapi ------------------------------------------------------

autoapi_dirs = ["../src"]  # location to parse for API reference

autoapi_ignore = [
    "*/deprecated/*",
    "*/experimental/*",
    "*/test/*",
    "*/__metadata__.py"
]

autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members"
]

# -- Options for myst-nb -----------------------------------------------------


nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "md"}]
}

nb_execution_mode = "cache"

nb_execution_timeout = -1

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

myst_title_to_header = True

myst_heading_anchors = 3

# convert latexdefs.tex to mathjax format
mathjax3_config = {'tex': {'macros': {}}}
with open('notebooks/latexdefs.tex', 'r') as f:
    for line in f:
        # newcommand macros
        macros = re.findall(
            r'\\(newcommand){\\(.*?)}(\[(\d)\])?{(.+)}', line)
        for macro in macros:
            if len(macro[2]) == 0:
                mathjax3_config['tex']['macros'][macro[1]] = "{"+macro[4]+"}"
            else:
                mathjax3_config['tex']['macros'][macro[1]] = [
                    "{"+macro[4]+"}", int(macro[3])]
        # DeclarMathOperator macros
        macros = re.findall(r'\\(DeclareMathOperator\*?){\\(.*?)}{(.+)}', line)
        for macro in macros:
            mathjax3_config['tex']['macros'][macro[1]
                                             ] = "{\\operatorname{"+macro[2]+"}}"

# bibtex options
bibtex_bibfiles = ['notebooks/biblio.bib']
bibtex_default_style = 'plain'
bibtex_reference_style = 'label'

import os
import sys

# Add the project's root dir to the system path to make sure that toytrack is found
sys.path.insert(0, os.path.abspath('../../'))

master_doc = 'index'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ToyTrack'
copyright = '2023, Daniel Murnane'
author = 'Daniel Murnane'
release = '0.1.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'myst_parser',
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SLATE documentation build configuration file
#
# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'SLATE'
copyright = '2017-2025, Innovative Computing Laboratory, University of Tennessee'
author = 'Innovative Computing Laboratory'

# The version info for the project
version = '2023.11'
release = '2023.11.05'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'myst_parser',
    'breathe',
]

# Breathe configuration for Doxygen integration
breathe_projects = {
    "SLATE": "../doxygen/xml",
    "BLASPP": "../doxygen/xml",
    "LAPACKPP": "../doxygen/xml"
}
breathe_default_project = "SLATE"

# Breathe configuration options
breathe_default_members = ('members', 'undoc-members')
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True

# Support for both RST and Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory.
html_static_path = ['_static']
html_css_files = ['custom.css']

# The name of an image file to place at the top of the sidebar.
# html_logo = '_static/slate_logo.png'

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'SLATEdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
''',
}

latex_documents = [
    (master_doc, 'SLATE.tex', 'SLATE Documentation',
     'Innovative Computing Laboratory', 'manual'),
]


# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'slate', 'SLATE Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'SLATE', 'SLATE Documentation',
     author, 'SLATE', 'Software for Linear Algebra Targeting Exascale',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# MathJax configuration for LaTeX rendering
mathjax3_config = {
    'tex': {
        'macros': {
            'RR': r'\mathbb{R}',
            'CC': r'\mathbb{C}',
        }
    }
}

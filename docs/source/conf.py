# This file is used to configure the documentation of the package
# It is used by the Sphinx documentation generator to generate the documentation
# The documentation is generated in the docs folder of the package
# The documentation is generated in HTML and LaTeX format

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# 0. Import the necessary modules

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

# 1. Set the project informations

def read_version():
    version_file = os.path.join(os.path.dirname(__file__), '..', '..', 'pydistort', '__version__.py')
    with open(version_file, "r") as file:
        exec(file.read()) 
    return locals()["__version__"]

project = "pydistort"
copyright = "2025-%Y, Artezaru"
author = "Artezaru"
release = read_version()

# 2. General configuration

import pydata_sphinx_theme
html_theme = "pydata_sphinx_theme"

extensions = [
    "sphinx.ext.autodoc", # Automatically document the code
    "sphinx.ext.viewcode", # Add links to the code
    "sphinx.ext.napoleon", # Support for Google and Numpy docstring formats
    "sphinx.ext.githubpages", # Publish the documentation on GitHub
    "sphinx.ext.autosummary", # Generate summaries of the modules
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# 3. Options for Latex output

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": "",
}

latex_documents = [
    ("index", "pydistort.tex", "pydistort Documentation",
     "Artezaru", "manual"),
]

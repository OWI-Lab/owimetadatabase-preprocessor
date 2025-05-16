# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, module_path)

project = "Owimetadatabase preprocessor"
copyright = "2025, OWI-Lab"
author = "OWI-Lab"
version = "0.9.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]

# autodoc_mock_imports = ["owimetadatabase_preprocessor"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # "alabaster"
autodoc_default_options = {
    'private-members': False
}
autodoc_member_order = 'bysource'
# html_theme_options = {
#    'body_max_width': '100%',
# }
# Logo configuration
html_logo = "_static/LogoOWI.png"
html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': '#2980B9',  # Header background color
}

# To display version information
html_title = f"{project} {version}"
html_short_title = project

html_static_path = ["_static"]
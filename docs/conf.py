import os
import sys

# Add project root to sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath(".."))

project = "lsnms"
author = "Rémy Dubois"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Allow parsing Markdown files with MyST
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

# HTML theme
html_theme = "furo"

html_static_path = ["_static"]

# Mock Rust backend module for build environments without compiled extensions
autodoc_mock_imports = ["lsnms.lsnms_rs"]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

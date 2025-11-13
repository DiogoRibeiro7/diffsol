import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../bindings/diffsol-pytorch"))
sys.path.insert(0, os.path.abspath("./notebooks"))

project = "diffsol-pytorch"
author = "Diffsol Contributors"
copyright = f"{datetime.now():%Y}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "myst_nb",
]
templates_path = ["_templates"]
exclude_patterns: list[str] = []
html_static_path = ["_static"]
html_theme = "furo"
mathjax3_config = {
    "tex": {"macros": {"d": r"\\mathrm{d}"}},
}
nb_execution_mode = "auto"
nb_execution_timeout = 120
nb_execution_allow_errors = False
autodoc_mock_imports = ["diffsol_pytorch", "diffsol_pytorch.diffsol_pytorch"]

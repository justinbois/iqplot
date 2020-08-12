# -*- coding: utf-8 -*-

"""Top-level package for bokeh-catplot."""

# Force showing deprecation warnings.
import re
import warnings

warnings.filterwarnings(
    "always", category=DeprecationWarning, module="^{}\.".format(re.escape(__name__))
)

from .cat import *
from .dist import *


__author__ = """Justin Bois"""
__email__ = "bois@caltech.edu"
__version__ = "0.1.0"

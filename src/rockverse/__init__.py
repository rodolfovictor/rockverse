__version__ = "0.2.0d"

from rockverse._utils.logo import make_logo

import rockverse.digitalrock as digitalrock

# Expose RcParams as a library-wide instance
from rockverse.rc import rcparams

# Expose Config as a library-wide instance
from rockverse.config import config

# Define the public API
__all__ = [
    "__version__",
    "config",
    "rcParams",
    "make_logo",
    "digitalrock",
]
"""Package containing humanoid suite."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

ISAACLAB_METADATA = toml.load(os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_METADATA["package"]["version"]

##
# Register Gym environments.
##

# For old versions of IsaacLab, import_packages was located in omni.isaac.lab.utils
# If you have issues with this import, try uncomment the line below and comment the line after
# from omni.isaac.lab.utils import import_packages
from omni.isaac.lab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

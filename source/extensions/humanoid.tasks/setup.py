"""Installation script for the 'humanoid'' python package."""

import itertools
import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch>=2.2.2",
    "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    # 5.26.0 introduced a breaking change, so we restricted it for now.
    # See issue https://github.com/tensorflow/tensorboard/issues/6808 for details.
    "protobuf>=3.20.2, < 5.0.0",
    # data collection
    "h5py",
    # basic logger
    "tensorboard",
    # video recording
    "moviepy",
]

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {}

# cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))


# Installation operation
setup(
    name="humanoid-suite",
    author="Qinxi Yu",
    maintainer="Qinxi Yu",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=["humanoid.tasks"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.0.0",
        "Isaac Sim :: 2023.1.1",
    ],
    zip_safe=False,
)

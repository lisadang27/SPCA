import setuptools, sys, os
import subprocess
from SPCA import __version__, name

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["numpy", "scipy", "astropy", "matplotlib", "emcee>=3", 'mc3',
                    "batman-package", "corner", "photutils>=0.7", "pandas", "pyyaml", "threadpoolctl", "scikit-learn", "tqdm"]

setuptools.setup(
    name = name,
    version = __version__,
    author = "Lisa Dang & Taylor James Bell",
    author_email = "lisa.dang@physics.mcgill.ca & taylor.bell@mail.mcgill.ca",
    description = "An open-source, modular, and automated pipeline for Spitzer Phase Curve Analyses.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/lisadang27/SPCA",
    license="GPL3",
    package_data={"": ["LICENSE"]},
    packages = setuptools.find_packages(),
    classifiers = (
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GPL3 License",
        "Operating System :: OS Independent",
    ),
    include_package_data = True,
    zip_safe = True,
    install_requires = install_requires,
    extras_require = {'GP': ['pybind11', 'george']}
)


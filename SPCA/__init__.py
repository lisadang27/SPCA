from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

name = "SPCA"

try:
    __version__ = _version(name)
except PackageNotFoundError:
    # A source checkout can be imported before the package is installed.
    __version__ = "0+unknown"

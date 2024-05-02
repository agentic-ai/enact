"""Version information for Enact."""

from importlib import metadata
import logging


PKG_NAME = 'enact'


try:
  __version__ = metadata.version(PKG_NAME)
except metadata.PackageNotFoundError:
  __version__ = '0.0.0'
  logging.warning(
    'Could not find installed version for %s, using %s',
    PKG_NAME, __version__)

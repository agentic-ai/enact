# Copyright 2023 Agentic.AI Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tracks installation paths of python distributions."""

from importlib import metadata
import inspect
import os
from typing import Any, Dict, Iterable, List, Optional

from enact import interfaces


_singleton: Optional['DistributionRegistry'] = None


def registry() -> 'DistributionRegistry':
  """Get the global package registry."""
  global _singleton
  if _singleton is None:
    _singleton = DistributionRegistry()
  assert _singleton is not None
  return _singleton


def _is_editable_pointer_file(dist_name: str, file_path: str) -> bool:
  """Check if this is the pointer file in the editable package."""
  base = os.path.basename(file_path)
  return (
    base.startswith('__editable__') and base.endswith('.pth')
    and dist_name in base)


class DistributionRegistry:
  """Keeps track of installed packages and their associated files."""

  def __init__(self):
    """Initialize a new package registry."""
    self._file_map: Dict[str, interfaces.DistributionInfo] = {}
    self._dir_map: Dict[str, interfaces.DistributionInfo] = {}
    self._registered: Dict[str, interfaces.DistributionInfo] = {}

  def registered(self) -> Iterable[interfaces.DistributionInfo]:
    """Yield the registered packages."""
    return self._registered.values()

  def is_registered(self, dist_name: str) -> bool:
    """Return whether the distribution is registered."""
    return dist_name in self._registered

  def register_distribution(
      self,
      dist_name: str,
      dist_version: Optional[str]=None,
      path: Optional[str]=None):
    """Register a distribution by the name you would use to install via pip.

    Args:
      dist_name: The name of the distribution.
      dist_version: The version of the distribution. If None, the function will
        attempt to infer the version via the installed package metadata.
      path: The install directory of distribution. If None, the function will
        attempt to directly register all files associated with the distribution
        with the provided package. For editable installs, this may be brittle
        and providing an explicit path is recommended.

    Raises:
      importlib.metadata.PackageNotFoundError: If the distribution is not
        installed and version and path were not explicitly provided.
    """
    files: List[str] = []
    if not dist_version or not path:
      dist = metadata.distribution(dist_name)
      if dist_version is None:
        dist_version = dist.metadata['version']
      if path is None and dist.files:
        files = [os.path.abspath(str(f.locate())) for f in dist.files]
    info = interfaces.DistributionInfo(
      name=dist_name, version=dist_version)
    if path:
      self._dir_map[path] = info
    for file_path in files:
      if _is_editable_pointer_file(dist_name, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
          path = os.path.abspath(file.read().strip())
          assert os.path.isdir(path), (
            f'Could not find path "{path}" to editable package '
            f'dist_name')
          self._dir_map[path] = info
      self._file_map[file_path] = info
    self._registered[dist_name] = info

  def get_path_distribution_info(self, path: str) -> (
      Optional[interfaces.DistributionInfo]):
    """Get the distribution info for a given path if it exists."""
    path = os.path.abspath(path)
    dist_info = self._file_map.get(path)
    if dist_info:
      return dist_info
    for dir_path, info in self._dir_map.items():
      if os.path.commonpath([dir_path, path]) == dir_path:
        return info
    return None

  def get_distribution_info(self, python_obj: Any) -> (
      Optional[interfaces.DistributionInfo]):
    """Get the distribution info for a given python type if it exists."""
    return self.get_path_distribution_info(inspect.getfile(python_obj))



def register_distribution(
    dist_name: str,
    dist_version: Optional[str]=None,
    path: Optional[str]=None):
  """Register a distribution by the name you would use to install via pip.

  Args:
    dist_name: The name of the distribution.
    dist_version: The version of the distribution. If None, the function will
      attempt to infer the version via the installed package metadata.
    path: The install directory of distribution. If None, the function will
      attempt to directly register all files associated with the distribution
      with the provided package. For editable installs, this may be brittle
      and providing an explicit path is recommended.

  Raises:
    importlib.metadata.PackageNotFoundError: If the distribution is not
      installed and version and path were not explicitly provided.
  """
  return registry().register_distribution(dist_name, dist_version, path)

def get_path_distribution_info(path: str) -> (
    Optional[interfaces.DistributionInfo]):
  """Get the distribution info for a given path if it exists."""
  return registry().get_path_distribution_info(path)

def get_distribution_info(python_obj: Any) -> (
    Optional[interfaces.DistributionInfo]):
  """Get the distribution info for a given python type if it exists."""
  return registry().get_distribution_info(python_obj)

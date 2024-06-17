# Copyright 2024 Agentic.AI Corporation.
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

"""Tests for the distribution registry."""

from importlib import metadata
import unittest
import re

import wrapt

import enact
from enact import distribution_registry
from enact import version


def _has_distribution(dist_name: str) -> bool:
  """Check if a distribution has been installed."""
  try:
    metadata.distribution(dist_name)
    return True
  except metadata.PackageNotFoundError:
    return False


def _is_editable_distribution(dist_name: str) -> bool:
  """Check if a distribution has been installed as editable."""
  expected_file_pattern = r'__editable__.*\.pth'
  files = metadata.distribution(dist_name).files
  return any(re.match(expected_file_pattern, str(f)) for f in files)


class DistributionRegistryTest(unittest.TestCase):
  """Tests for the distribution registry."""

  def test_explicit_registration(self):
    """Tests that explicit registration works."""
    d = distribution_registry.DistributionRegistry()
    d.register_distribution('foo', '1.0', '/path/to/foo')
    self.assertEqual(
      d.get_path_distribution_key('/path/to/foo/my.py'),
      enact.DistributionKey('foo', '1.0'))

  def test_register_editable_install(self):
    """Tests that editable installs can be registered."""
    # Check if enact is installed in editable mode.
    if (not _has_distribution(version.DIST_NAME)
        or not _is_editable_distribution(version.DIST_NAME)):
      self.skipTest('enact is not installed in editable mode')

    d = distribution_registry.DistributionRegistry()
    d.register_distribution(version.DIST_NAME)
    self.assertEqual(
      d.get_distribution_key(enact.ResourceBase),
      enact.DistributionKey(version.DIST_NAME, version.__version__))

  def test_register_non_editable_install(self):
    """Tests that editable installs can be registered."""
    # Check if wrapt is installed in non-editable mode.
    if (not _has_distribution('wrapt')
        or _is_editable_distribution('wrapt')):
      self.skipTest('wrapt is not installed in non-editable mode')

    d = distribution_registry.DistributionRegistry()
    d.register_distribution('wrapt')
    self.assertEqual(
      d.get_distribution_key(wrapt.decorator),
      enact.DistributionKey('wrapt', wrapt.__version__))

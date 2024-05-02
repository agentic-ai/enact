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

"""Checks that the package version and python version match."""

import os
import unittest
import re

import enact

class TestVersion(unittest.TestCase):
  """Tests for the version."""

  def test_version(self):
    """Tests that the version matches."""
    try:
      pyproject_path = os.path.join(
          os.path.dirname(__file__), os.pardir, 'pyproject.toml')
      with open(pyproject_path, 'r', encoding='utf8') as f:
        pyproject_contents = f.read()
    except FileNotFoundError:
      self.skipTest('pyproject.toml not found')
    # Python 3.8 does not have tomllib, so we use a regex instead.
    pyproject_contents = re.search(r'version = "(.*)"', pyproject_contents)
    pyproject_version = pyproject_contents.group(1)
    self.assertEqual(enact.__version__, pyproject_version)

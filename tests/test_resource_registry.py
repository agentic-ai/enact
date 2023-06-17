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

"""Tests for the resource_registry module."""

import dataclasses
import unittest

import enact
from enact import resource_registry

@enact.register
@dataclasses.dataclass
class SimpleResource(enact.Resource):
  """A simple resource for testing."""


class RegistryTest(unittest.TestCase):
  """Tests the Registry module."""

  def test_registry(self):
    """Tests that the registry works."""
    registry = resource_registry.Registry()
    registry.register(SimpleResource)
    self.assertEqual(registry.lookup(SimpleResource.type_id()), SimpleResource)

  def test_registry_error(self):
    """Tests that the registry raises an error for non-resources."""
    registry = resource_registry.Registry()
    with self.assertRaises(resource_registry.RegistryError):
      registry.register(int)  # type: ignore

  def test_lookup_error(self):
    """Tests that lookup raises the correct error."""
    registry = resource_registry.Registry()
    with self.assertRaises(resource_registry.ResourceNotFound):
      registry.lookup('SimpleResource')

  def test_singleton_and_decorator(self):
    """Tests the Registry singleton."""
    self.assertEqual(
      resource_registry.Registry.get().lookup(SimpleResource.type_id()),
      SimpleResource)

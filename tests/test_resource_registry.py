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
from typing import Tuple, Type
import unittest

import enact

@enact.register
@dataclasses.dataclass
class SimpleResource(enact.Resource):
  """A simple resource for testing."""


@dataclasses.dataclass
class CustomType:
  """A custom non-resource type for testing."""
  tuple_value: Tuple


@enact.register_wrapper
@dataclasses.dataclass
class CustomWrapper(enact.ResourceWrapper[CustomType]):
  """A simple resource for testing."""
  list_value: list

  @classmethod
  def wrapped_type(cls) -> Type[CustomType]:
    return CustomType

  @classmethod
  def wrap(cls, wrapped: CustomType) -> 'CustomWrapper':
    return cls(list(wrapped.tuple_value))

  def unwrap(self) -> CustomType:
    return CustomType(tuple(self.list_value))


class RegistryTest(unittest.TestCase):
  """Tests the Registry module."""

  def test_registry(self):
    """Tests that the registry works."""
    registry = enact.Registry()
    registry.register(SimpleResource)
    self.assertEqual(registry.lookup(SimpleResource.type_id()), SimpleResource)

  def test_registry_error(self):
    """Tests that the registry raises an error for non-resources."""
    registry = enact.Registry()
    with self.assertRaises(enact.RegistryError):
      registry.register(int)  # type: ignore

  def test_lookup_error(self):
    """Tests that lookup raises the correct error."""
    registry = enact.Registry()
    with self.assertRaises(enact.ResourceNotFound):
      registry.lookup('SimpleResource')

  def test_singleton_and_decorator(self):
    """Tests the Registry singleton."""
    self.assertEqual(
      enact.Registry.get().lookup(SimpleResource.type_id()),
      SimpleResource)

  def test_wrap_field_values(self):
    """Tests wrapping and unwrapping field values."""
    field_values = [
      (None, type(None)), (1, int), (1.0, float), ('a', str),
      (bytes([1, 2, 3]), bytes), ([1, 2, 3], list),
      ({'1': 1, '2': 2}, dict)]
    for value, value_type in field_values:
      with self.subTest(str(value_type)):
        wrapped = enact.wrap(value)
        assert isinstance(wrapped, enact.ResourceWrapperBase)
        self.assertEqual(enact.unwrap(wrapped), value)

  def test_wrap_noop_on_resources(self):
    """Tests that wrapping does nothing on resources."""
    resource = SimpleResource()
    self.assertEqual(enact.wrap(resource), resource)

  def test_wrap_custom_type(self):
    """Tests wrapping custom types."""
    custom = CustomType((1, 2, 3))
    wrapped = enact.wrap(custom)
    assert isinstance(wrapped, CustomWrapper)
    self.assertEqual(enact.unwrap(wrapped), custom)

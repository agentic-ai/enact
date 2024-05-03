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
import types
from typing import Any, List, Tuple, Type
import unittest

import enact
from enact import resource_registry
from enact import version

import register_enact_distribution  # pylint: disable=unused-import


@enact.register
@dataclasses.dataclass
class SimpleResource(enact.Resource):
  """A simple resource for testing."""


@dataclasses.dataclass
class CustomType:
  """A custom non-resource type for testing."""
  tuple_value: Tuple


@enact.register
@dataclasses.dataclass
class CustomWrapper(enact.TypeWrapper[CustomType]):
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
        assert isinstance(wrapped, enact.TypeWrapperBase)
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

  def test_wrap_nests(self):
    """Tests that wrapping nests with custom types works."""
    py_dict = {'a': [CustomType((1, 2)), {'test': CustomType((2, 3))}]}
    wrapped = enact.wrap(py_dict)
    assert isinstance(wrapped, enact.TypeWrapperBase)
    self.assertEqual(enact.unwrap(wrapped), py_dict)

  def test_wrap_types(self):
    """Tests that wrapping types works."""
    type_nest = [int, [float, bytes]]
    as_fields = resource_registry.to_field_value(type_nest)
    self.assertEqual(
      as_fields, [resource_registry.IntWrapper,
                  [resource_registry.FloatWrapper,
                   resource_registry.BytesWrapper]])
    self.assertEqual(
      resource_registry.from_field_value(as_fields), type_nest)

  def test_wrap_function(self):
    """Tests that wrapping functions works."""
    def foo(x: int) -> int:
      return x

    def foo_wrapper(x: int) -> int:
      return x

    @dataclasses.dataclass
    class MethodWrapper(resource_registry.MethodWrapper):
      @classmethod
      def wrapper_function(cls):
        return foo_wrapper

      @classmethod
      def wrapped_function(cls):
        return foo

    @dataclasses.dataclass
    class FunctionWrapper(resource_registry.FunctionWrapper):
      """The FunctionWrapper that goes with MethodWrapper."""
      @classmethod
      def wrapper_function(cls):
        return foo_wrapper

      @classmethod
      def wrapped_function(cls):
        return foo

      @classmethod
      def method_wrapper(cls) -> Type[resource_registry.MethodWrapper]:
        return MethodWrapper

    registry = enact.Registry()
    registry.register(FunctionWrapper)
    self.assertEqual(registry.wrap(foo_wrapper), FunctionWrapper())
    self.assertEqual(
      registry.unwrap(registry.wrap(foo_wrapper)), foo_wrapper)

  def test_wrap_method(self):
    """Tests that wrapping methods works."""

    @dataclasses.dataclass
    class MyClass(enact.Resource):
      y: int

      def foo(self, x: int) -> int:
        return x

      def foo_wrapper(self, x: int) -> int:
        return x

    @dataclasses.dataclass
    class MethodWrapper(resource_registry.MethodWrapper):
      instance: Any

      @classmethod
      def wrapper_function(cls):
        return MyClass.foo_wrapper

      @classmethod
      def wrapped_function(cls):
        return MyClass.foo

    @dataclasses.dataclass
    class FunctionWrapper(resource_registry.FunctionWrapper):
      """The FunctionWrapper that goes with MethodWrapper."""
      @classmethod
      def wrapper_function(cls):
        return MyClass.foo_wrapper

      @classmethod
      def wrapped_function(cls):
        return MyClass.foo

      @classmethod
      def method_wrapper(cls) -> Type[resource_registry.MethodWrapper]:
        return MethodWrapper

    registry = enact.Registry()
    registry.register(MyClass)
    registry.register(FunctionWrapper)

    self.assertEqual(registry.wrap(MyClass.foo_wrapper), FunctionWrapper())
    self.assertEqual(
      registry.unwrap(registry.wrap(MyClass.foo_wrapper)), MyClass.foo_wrapper)

    instance = MyClass(69)
    wrapped_method = registry.wrap(instance.foo_wrapper)
    self.assertEqual(wrapped_method, MethodWrapper(instance))
    unwrapped_method = registry.unwrap(wrapped_method)
    self.assertIsInstance(unwrapped_method, types.MethodType)
    self.assertEqual(unwrapped_method.__self__, instance)
    self.assertEqual(unwrapped_method.__func__, MyClass.foo_wrapper)

    # Check that original instance and self are aliased.
    instance.y = 420
    self.assertEqual(unwrapped_method.__self__.y, 420)

  def test_wrap_type_value(self):
    """Tests that wrapping methods works."""
    @enact.register
    class MyResource(enact.Resource):
      pass

    # Test for resource subclasses.
    self.assertIsInstance(
      enact.wrap(MyResource),
      resource_registry.ResourceTypeWrapper)

    # Test for types that have associated wrappers.
    self.assertIsInstance(
      enact.wrap(int),
      resource_registry.ResourceTypeWrapper)

    with enact.Store():
      resource_type_ref = enact.commit(MyResource)
      int_type_ref = enact.commit(int)

      # Clear cache
      resource_type_ref = enact.Ref.from_id(resource_type_ref.id)
      int_type_ref = enact.Ref.from_id(int_type_ref.id)

      # Checkout
      self.assertEqual(resource_type_ref.checkout(), MyResource)
      self.assertEqual(int_type_ref.checkout(), int)

  def test_wrap_unregistered_type_fails(self):
    """Tests that wrapping an unregistered type fails."""
    class MyUnregisteredClass:
      pass
    with enact.Store():
      with self.assertRaises(resource_registry.MissingWrapperError):
        enact.wrap(MyUnregisteredClass)

  def test_deepcopy_primitives(self):
    """Tests that deep copying primitives works."""
    primitives = [1, 1.0, 'a', bytes([1, 2, 3]), True, False, None]
    for p in primitives:
      with self.subTest(str(p)):
        self.assertEqual(resource_registry.deepcopy(p), p)

  def test_deepcopy_nests(self):
    """Tests that deep copying primitives works."""
    nest: List[Any] = [[1, 2], {'a': [1, 2], 'b': True}]
    copy = resource_registry.deepcopy(nest)
    self.assertEqual(copy, nest)
    self.assertIsNot(copy, nest)
    self.assertIsNot(copy[0], nest[0])
    self.assertIsNot(copy[1], nest[1])
    self.assertIsNot(copy[1]['a'], nest[1]['a'])

  def test_enact_types_have_distribution_info(self):
    """Tests that type distribution info is present for enact types."""
    self.assertEqual(
      resource_registry.IntWrapper.type_distribution_info(),
      enact.DistributionInfo(version.DIST_NAME, version.__version__))

  def test_auto_assign_distribution_info(self):
    """Tests that type distribution info is auto-assigned on register"""
    class MyResource(enact.Resource):
      pass
    self.assertIsNone(MyResource.type_distribution_info())
    enact.register(MyResource)
    self.assertEqual(
      MyResource.type_distribution_info(),
      enact.DistributionInfo(
        f'{version.PKG_NAME}-tests', version.__version__))


if __name__ == 'main':
  unittest.main()

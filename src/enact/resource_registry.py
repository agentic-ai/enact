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

"""Type registration functionality to allow deserialization of resources."""

import inspect
import types
from typing import (
  Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Set,
  Type, TypeVar, Union, cast)

from enact import distribution_registry
from enact import interfaces



WrappedT = TypeVar('WrappedT')
WrapperT = TypeVar('WrapperT', bound=interfaces.TypeWrapperBase)
ResourceT = TypeVar('ResourceT', bound=interfaces.ResourceBase)

FieldValueWrapperT = TypeVar('FieldValueWrapperT', bound='FieldValueWrapper')
FunctionWrapperT = TypeVar('FunctionWrapperT', bound='FunctionWrapper')
MethodWrapperT = TypeVar('MethodWrapperT', bound='MethodWrapper')

class RegistryError(Exception):
  """Raised when there is an error with the registry."""


class ResourceNotFound(RegistryError):
  """Raised when a resource is not found."""


class FunctionWrapper(interfaces.ResourceBase):
  """Base class for function wrappers.

  Should be subclassed as an Invokable or AsyncInvokable.

  Function wrapping require a three step approach:
  1) The native function is transparently wrapped using a python wrapper
     function.
  2) The wrapper function routes calls into an invokable resource, so that
     inputs are tracked.
  3) The invokable calls the native function that is being wrapped.

  The native function that is being wrapped is exposed as wrapped_function.
  The python wrapper function that calls into the invokable is exposed as
  wrapper function.
  """
  @classmethod
  def wrapper_function(cls) -> Callable:
    """The python function that routes calls into this invokable."""
    raise NotImplementedError()

  @classmethod
  def wrapped_function(cls) -> Callable:
    """The python function that is being called by this invokable."""
    raise NotImplementedError()

  @classmethod
  def wrap(cls: Type[FunctionWrapperT], c: Callable) -> Union[
      'FunctionWrapper', 'MethodWrapper']:
    """Wrap a function."""
    if inspect.ismethod(c):
      return cls.method_wrapper().wrap(c)
    assert c == cls.wrapper_function(), (
      f'Wrong wrapper class for callable: {cls} vs {c}')
    return cls()

  @classmethod
  def method_wrapper(cls) -> 'Type[MethodWrapper]':
    """The associated method wrapper."""
    raise NotImplementedError()


class MethodWrapper(interfaces.ResourceBase):
  """Base class for method wrappers."""

  @classmethod
  def wrap(cls: Type[MethodWrapperT], m: types.MethodType) -> MethodWrapperT:
    """Wrap a method."""
    assert inspect.ismethod(m), 'Expected method.'
    assert m.__self__, 'Expected __self__ attribute on bound method.'
    assert m.__func__ == cls.wrapper_function(), (
      'Expected method to be based on the same function as its '
      'function wrapper.')
    # pylint: disable=too-many-function-args
    return cls(m.__self__) # type: ignore

  @classmethod
  def wrapper_function(cls) -> Callable:
    """The python function that routes calls into this invokable."""
    raise NotImplementedError()

  @classmethod
  def wrapped_function(cls) -> Callable:
    """The python function that is being called by this invokable."""
    raise NotImplementedError()

  def get_instance(self) -> Any:
    """Added to interface to simplify typing."""
    return self.instance  # type: ignore


class MissingWrapperError(interfaces.FieldTypeError):
  """Raised when a required wrapper is missing."""


class Registry:
  """Registers resource types for deserialization."""

  _singleton: Optional['Registry'] = None

  def __init__(self):
    """Initializes a registry."""
    # Ensure that the enact distribution is registered to its semantic version
    # before registering any resource types.
    distribution_registry.ensure_enact_registered()
    self.allow_reregistration = True

    # Map from type id to resource type.
    self._type_map: Dict[str, Type[interfaces.ResourceBase]] = {}

    # Map from python types to wrapper types.
    self._wrapped_types: Dict[Type, Type[interfaces.TypeWrapperBase]] = {}
    self._wrapper_types: Set[Type[interfaces.TypeWrapperBase]] = set()
    self._function_wrappers: Dict[Callable, Type[FunctionWrapper]] = {}


  def _from_dict_value(self, value: interfaces.ResourceDictValue) -> (
      interfaces.FieldValue):
    """Transforms a resource dict value to a field value."""
    if isinstance(value, interfaces.PRIMITIVES):
      return value
    if isinstance(value, type) and issubclass(value, interfaces.ResourceBase):
      return value
    if isinstance(value, List):
      return [self._from_dict_value(x) for x in value]
    if isinstance(value, interfaces.ResourceDict):
      return self.from_resource_dict(value)
    if isinstance(value, Dict):
      def _assert_str(maybe_str: str) -> str:
        if type(maybe_str) is not str:  # pylint: disable=unidiomatic-typecheck
          raise interfaces.FieldTypeError(
            f'Expected string key, got {type(maybe_str)}')
        return maybe_str
      return {
        _assert_str(k): self._from_dict_value(v)
        for k, v in value.items()}
    raise interfaces.FieldTypeError(
      f'Encountered unsupported resource '
      f'dict value type {type(value)}: {value}')

  def deepcopy(self, resource: ResourceT) -> ResourceT:
    """Create a deep-copy of the resource."""
    return cast(ResourceT, self.from_resource_dict(resource.to_resource_dict()))

  def from_resource_dict(self, resource_dict: interfaces.ResourceDict) -> (
      interfaces.ResourceBase):
    """Constructs the resource from a ResourceDict dictionary."""
    if not isinstance(resource_dict, interfaces.ResourceDict):
      raise TypeError(f'Input is not a ResourceDict: {resource_dict}')
    resource_type = self.lookup(resource_dict.type_info)
    field_dict = {
      k: self._from_dict_value(v)
      for k, v in resource_dict.items()}
    return resource_type.from_fields(field_dict)

  def register(self, resource: Type[interfaces.ResourceBase]):
    """Registers the resource type."""
    # Check argument type.
    if not issubclass(resource, interfaces.ResourceBase):
      raise RegistryError(
        f'Cannot register non-resource type: {resource}')
    # Auto-add distribution info.
    dist_info = resource.type_distribution_info()
    if dist_info is None:
      dist_info = distribution_registry.get_distribution_info(resource)
      if dist_info is not None:
        resource.set_type_distribution_info(dist_info)
    # Record the type.
    type_id = resource.type_id()
    if (type_id in self._type_map and
        self._type_map[type_id] != resource and
        not self.allow_reregistration):
      raise RegistryError(
        f'{id(resource)} == {id(self._type_map[type_id])}\n'
        f'While registering {resource}: '
        f'Type with id {type_id} already '
        f'registered to a different type: '
        f'{self._type_map[type_id]}\n')
    self._type_map[type_id] = resource
    # Handle special types.
    if issubclass(resource, interfaces.TypeWrapperBase):
      self._register_type_wrapper(resource)
    if issubclass(resource, FunctionWrapper):
      return self._register_function_wrapper(resource)

  def lookup(self, type_id: Union[str, interfaces.TypeInfo]) -> (
      Type[interfaces.ResourceBase]):
    """Looks up a resource type by name or type_info."""
    if isinstance(type_id, interfaces.TypeInfo):
      type_id = type_id.type_id()
    resource_class = self._type_map.get(type_id)
    if not resource_class:
      raise ResourceNotFound(f'No type registered for {type_id}')
    return resource_class

  def _register_function_wrapper(self, wrapper_type: Type[FunctionWrapper]):
    """Register a function wrapper type."""
    self._function_wrappers[
        wrapper_type.wrapper_function()] = wrapper_type

  def _register_type_wrapper(self, wrapper_type: Type[WrapperT]):
    """Register a new wrapper type."""
    self._wrapped_types[wrapper_type.wrapped_type()] = wrapper_type
    self._wrapper_types.add(wrapper_type)

  def get_type_wrapper(self, t: Type[WrappedT]) -> Optional[
      Type[interfaces.TypeWrapperBase[WrappedT]]]:
    """Return a matching wrapper type if present."""
    wrapper = self._wrapped_types.get(t)
    if wrapper:
      return cast(Type[interfaces.TypeWrapperBase[WrappedT]], wrapper)
    found: Optional[Type[interfaces.TypeWrapperBase]] = None
    for k, v in self._wrapped_types.items():
      if issubclass(t, k):
        if found:
          raise RegistryError(
            f'Found multiple wrappers for type {t}: {found} and {v}')
        found = v
    return found

  def _get_function_wrapper_type(self, c: Callable) -> Type['FunctionWrapper']:
    """Return the function wrapper for c or raise an error."""
    func = c
    if inspect.ismethod(c):
      func = c.__func__
    if not isinstance(func, Hashable):
      raise interfaces.FieldTypeError(
        'Only immutable callables (e.g. python functions) can be handled '
        'using enact. If you need use callable python classes that are '
        'mutable, consider subclassing from Invokable or AsyncInvokable.')
    function_wrapper_type = self._function_wrappers.get(func)
    if not function_wrapper_type:
      raise MissingWrapperError(
        f'Cannot find wrapper for {func}. Did you register the '
        f'function or method with the @enact.register decorator?')
    return function_wrapper_type

  def wrap(self, value: Any) -> interfaces.ResourceBase:
    """Wrap a value if necessary."""
    if isinstance(value, interfaces.ResourceBase):
      return value
    type_wrapper_class = self.get_type_wrapper(type(value))
    if type_wrapper_class:
      return type_wrapper_class.wrap(value)
    if callable(value):
      return self._get_function_wrapper_type(value).wrap(value)
    if inspect.iscoroutine(value):
      raise MissingWrapperError(
        f'Cannot wrap coroutine {value}. Did you forget to await it?')
    raise MissingWrapperError(
      f'Cannot wrap value of type {type(value)}. Please register '
      f'a TypeWrapper for this type.')

  def wrap_type(self, value: Type) -> Type[interfaces.ResourceBase]:
    """Wrap a type if necessary."""
    if issubclass(value, interfaces.ResourceBase):
      return value
    wrapper = self.get_type_wrapper(value)
    if wrapper:
      return wrapper
    raise MissingWrapperError(
      f'Cannot wrap type {value}. Please register '
      f'a TypeWrapper for this type.')

  def unwrap(self, value: Any) -> Any:
    """Unwrap a value if wrapped."""
    if isinstance(value, interfaces.TypeWrapperBase):
      return value.unwrap()
    if isinstance(value, FunctionWrapper):
      return value.wrapper_function()
    if isinstance(value, MethodWrapper):
      return types.MethodType(
        value.wrapper_function(), value.get_instance())
    return value

  def unwrap_type(self, value: Type[interfaces.ResourceBase]) -> Type:
    """Unwrap a type if wrapped."""
    if issubclass(value, interfaces.TypeWrapperBase):
      return value.wrapped_type()
    return value

  @classmethod
  def get(cls) -> 'Registry':
    """Returns the singleton registry."""
    if not cls._singleton:
      cls._singleton = cls()
    return cls._singleton


def register(cls: Type[ResourceT]) -> Type[ResourceT]:
  """Decorator for resource classes."""
  Registry.get().register(cls)
  return cls


def register_wrapper(cls: Type[WrapperT]) -> Type[WrapperT]:
  """Decorator for resource wrapper classes."""
  # pylint: disable=protected-access
  Registry.get()._register_type_wrapper(cls)
  return cls


def wrap(value: Any) -> interfaces.ResourceBase:
  """Wrap a value as a resource if necessary."""
  return Registry.get().wrap(value)


def unwrap(value: Any) -> Any:
  """Unwrap a value if wrapped."""
  return Registry.get().unwrap(value)


def wrap_type(value: Type) -> Type[interfaces.ResourceBase]:
  """Wrap a type as a resource."""
  return Registry.get().wrap_type(value)


def unwrap_type(value: Type[interfaces.ResourceBase]) -> Type:
  """Wrap a type."""
  return Registry.get().unwrap_type(value)


def deepcopy(value: WrappedT) -> WrappedT:
  """Deep copy a value."""
  if isinstance(value, interfaces.ResourceBase):
    result = from_resource_dict(value.to_resource_dict())
  else:
    result = unwrap(from_resource_dict(wrap(value).to_resource_dict()))
  return cast(WrappedT, result)


def _ensure_str_key(s: Any) -> str:
  """Ensure that a value is a string."""
  if not isinstance(s, str):
    raise ValueError(
      f'Cannot auto-wrap a dictionary with non-str keys: '
      f'{s} of type {type(s)}')
  return s


def to_field_value(value: Any) -> interfaces.FieldValue:
  """Wrap a value as a field value."""
  if isinstance(value, interfaces.PRIMITIVES):
    return value
  if isinstance(value, list):
    return [to_field_value(x) for x in value]
  if isinstance(value, dict):
    return {_ensure_str_key(k): to_field_value(v) for k, v in value.items()}
  if isinstance(value, type):
    return Registry.get().wrap_type(value)
  return wrap(value)


def from_field_value(value: interfaces.FieldValue) -> Any:
  """Unwrap a field value into a python value."""
  if isinstance(value, interfaces.ResourceBase):
    return unwrap(value)
  if (
      isinstance(value, type) and
      issubclass(value, interfaces.TypeWrapperBase)):
    return unwrap_type(value)
  if isinstance(value, list):
    return [from_field_value(x) for x in value]
  if isinstance(value, dict):
    return {k: from_field_value(v) for k, v in value.items()}
  return value

def from_resource_dict(resource_dict: interfaces.ResourceDict) -> (
    interfaces.ResourceBase):
  """Constructs the resource from a ResourceDict dictionary."""
  return Registry.get().from_resource_dict(resource_dict)


class FieldValueWrapper(interfaces.TypeWrapperBase[WrappedT]):
  """Base class for field value wrappers."""
  value: interfaces.FieldValue

  def __init__(self, value: interfaces.FieldValue):
    """Initializes a wrapped field value."""
    self.value = value

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    return ('wrapped',)

  def field_values(self) -> Iterable[interfaces.FieldValue]:
    return (self.value,)

  @classmethod
  def from_fields(
      cls: Type[FieldValueWrapperT],
      field_dict: Mapping[str, interfaces.FieldValue]) -> FieldValueWrapperT:
    assert len(field_dict) == 1
    return cls(field_dict['wrapped'])

  def set_from(self, other: interfaces.ResourceBase):
    """Sets the fields of this resource from another resource."""
    if not type(self) is type(other):
      raise RegistryError(
        f'Cannot set fields from {type(other)} to {type(self)}.')
    assert isinstance(other, FieldValueWrapper)
    self.wrapped = deepcopy(other).value

  @classmethod
  def wrap(cls: Type[FieldValueWrapperT],
           value: WrappedT) -> FieldValueWrapperT:
    """Wrap a value directly."""
    assert isinstance(value, cls.wrapped_type()), (
      'Cannot wrap value of type {type(value)} with wrapper {cls}.')
    return cls(to_field_value(value))

  def unwrap(self) -> WrappedT:
    """Unwrap a value directly."""
    assert isinstance(self.value, self.wrapped_type())
    return from_field_value(self.value)


@register
class NoneWrapper(
  interfaces.TypeWrapperBase):
  """Wrapper for None."""
  @classmethod
  def wrapped_type(cls) -> Type[None]:
    return type(None)

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    return ()

  def field_values(self) -> Iterable[interfaces.FieldValue]:
    return ()

  @classmethod
  def from_fields(
      cls, field_dict: Mapping[str, interfaces.FieldValue]) -> 'NoneWrapper':
    assert len(field_dict) == 0
    return cls()

  def set_from(self, other: interfaces.ResourceBase):
    """Sets the fields of this resource from another resource."""
    if not type(self) is type(other):
      raise RegistryError(
        f'Cannot set fields from {type(other)} to {type(self)}.')

  @classmethod
  def wrap(cls, value: WrappedT) -> 'NoneWrapper':
    """Wrap a value directly."""
    assert isinstance(value, cls.wrapped_type()), (
      'Cannot wrap value of type {type(value)} with wrapper {cls}.')
    return NoneWrapper()

  def unwrap(self) -> None:
    """Unwrap a value directly."""
    return None

class PrimitiveWrapper(FieldValueWrapper[WrappedT]):
  """Wrapper for primitives."""

  @classmethod
  def is_immutable(cls) -> bool:
    return True


@register
class IntWrapper(PrimitiveWrapper[int]):
  """Wrapper for ints."""
  @classmethod
  def wrapped_type(cls) -> Type[int]:
    return int


@register
class FloatWrapper(PrimitiveWrapper[float]):
  """Wrapper for floats."""
  @classmethod
  def wrapped_type(cls) -> Type[float]:
    return float


@register
class BoolWrapper(PrimitiveWrapper[bool]):
  """Wrapper for bools."""
  @classmethod
  def wrapped_type(cls) -> Type[bool]:
    return bool


@register
class StrWrapper(PrimitiveWrapper[str]):
  """Wrapper for strs."""
  @classmethod
  def wrapped_type(cls) -> Type[str]:
    return str


@register
class BytesWrapper(PrimitiveWrapper[bytes]):
  """Wrapper for bytes."""
  @classmethod
  def wrapped_type(cls) -> Type[bytes]:
    return bytes


@register
class ListWrapper(FieldValueWrapper[list]):
  """Wrapper for lists."""
  @classmethod
  def wrapped_type(cls) -> Type[list]:
    return list

  @classmethod
  def set_wrapped_value(cls, target: list, src: list):
    target.clear()
    target.extend(src)


@register
class DictWrapper(FieldValueWrapper[dict]):
  """Wrapper for dicts."""
  @classmethod
  def wrapped_type(cls) -> Type[dict]:
    return dict

  @classmethod
  def set_wrapped_value(cls, target: dict, src: dict):
    target.clear()
    target.update(src)

@register
class ResourceTypeWrapper(FieldValueWrapper[type]):
  """Wrapper for type-valued fields."""

  @classmethod
  def wrapped_type(cls) -> Type[type]:
    return type

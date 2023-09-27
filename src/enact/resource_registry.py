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

from typing import (
  Any, Dict, Hashable, Iterable, Mapping, Optional, Set, Type, TypeVar, cast)

from enact import interfaces, resources


WrappedT = TypeVar('WrappedT')
WrapperT = TypeVar('WrapperT', bound=interfaces.ResourceWrapperBase)


class RegistryError(Exception):
  """Raised when there is an error with the registry."""


class ResourceNotFound(RegistryError):
  """Raised when a resource is not found."""



FieldValueWrapperT = TypeVar('FieldValueWrapperT', bound='FieldValueWrapper')


class FieldValueWrapper(interfaces.ResourceWrapperBase[WrappedT]):
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
    self.wrapped = other.deepcopy_resource().value

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


class NoneWrapper(interfaces.ResourceWrapperBase):
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


class IntWrapper(PrimitiveWrapper[int]):
  """Wrapper for ints."""
  @classmethod
  def wrapped_type(cls) -> Type[int]:
    return int


class FloatWrapper(PrimitiveWrapper[float]):
  """Wrapper for floats."""
  @classmethod
  def wrapped_type(cls) -> Type[float]:
    return float


class BoolWrapper(PrimitiveWrapper[bool]):
  """Wrapper for bools."""
  @classmethod
  def wrapped_type(cls) -> Type[bool]:
    return bool


class StrWrapper(PrimitiveWrapper[str]):
  """Wrapper for strs."""
  @classmethod
  def wrapped_type(cls) -> Type[str]:
    return str


class BytesWrapper(PrimitiveWrapper[bytes]):
  """Wrapper for bytes."""
  @classmethod
  def wrapped_type(cls) -> Type[bytes]:
    return bytes


class ListWrapper(FieldValueWrapper[list]):
  """Wrapper for lists."""
  @classmethod
  def wrapped_type(cls) -> Type[list]:
    return list

  @classmethod
  def set_wrapped_value(cls, target: list, src: list):
    target.clear()
    target.extend(src)


class DictWrapper(FieldValueWrapper[dict]):
  """Wrapper for dicts."""
  @classmethod
  def wrapped_type(cls) -> Type[dict]:
    return dict

  @classmethod
  def set_wrapped_value(cls, target: dict, src: dict):
    target.clear()
    target.update(src)


class MissingWrapperError(interfaces.FieldTypeError):
  """Raised when a required wrapper is missing."""


class Registry:
  """Registers resource types for deserialization."""

  _singleton: Optional['Registry'] = None

  def __init__(self):
    """Initializes a registry."""
    self.allow_reregistration = True
    # Map from type id to resource type.
    self._type_map: Dict[str, Type[interfaces.ResourceBase]] = {}
    # Map from python types to wrapper types.
    self._wrapped_types: Dict[Type, Type[interfaces.ResourceWrapperBase]] = {}
    self._wrapper_types: Set[Type[interfaces.ResourceWrapperBase]] = set()
    # Map from python instances to instance wrapper resources.
    self._wrapped_instances: Dict[Hashable, resources.ImmutableResource] = {}
    self._instance_wrappers: Dict[resources.ImmutableResource, Hashable] = {}
    self.register(NoneWrapper)
    self.register(IntWrapper)
    self.register(FloatWrapper)
    self.register(BoolWrapper)
    self.register(StrWrapper)
    self.register(BytesWrapper)
    self.register(ListWrapper)
    self.register(DictWrapper)

  def register(self, resource: Type[interfaces.ResourceBase]):
    """Registers the resource type."""
    if not issubclass(resource, interfaces.ResourceBase):
      raise RegistryError(
        f'Cannot register non-resource type: {resource}')
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
    if issubclass(resource, interfaces.ResourceWrapperBase):
      self._register_resource_wrapper(resource)

  def lookup(self, type_id: str) -> Type[interfaces.ResourceBase]:
    """Looks up a resource type by name."""
    resource_class = self._type_map.get(type_id)
    if not resource_class:
      raise ResourceNotFound(f'No type registered for {type_id}')
    return resource_class

  def register_instance_wrapper(
      self, instance: Hashable, wrapper: resources.ImmutableResource):
    """Register a wrapper type for a hashable instance.

    This is useful for registering functions to their wrappers.

    Args:
      instance: An instance. Must be hashable.
      wrapper: An immutable wrapper instance.
    """
    if isinstance(instance, interfaces.ResourceBase):
      raise TypeError('Cannot wrap a resource instance.')
    if isinstance(instance, type):
      raise TypeError(
        'Cannot wrap a type instance. Use a ResourceWrapper instead.')
    if not wrapper.type_id() in self._type_map:
      raise RegistryError(
        f'Instance wrapper type {type(wrapper)} is not registered.')
    self._instance_wrappers[wrapper] = instance
    self._wrapped_instances[instance] = wrapper

  def _register_resource_wrapper(self, wrapper_type: Type[WrapperT]):
    """Register a new wrapper type."""
    self._wrapped_types[wrapper_type.wrapped_type()] = wrapper_type
    self._wrapper_types.add(wrapper_type)

  def get_wrapper_type(self, t: Type[WrappedT]) -> Optional[
      Type[interfaces.ResourceWrapperBase[WrappedT]]]:
    """Return a matching wrapper type if present."""
    wrapper = self._wrapped_types.get(t)
    if wrapper:
      return cast(Type[interfaces.ResourceWrapperBase[WrappedT]], wrapper)
    found: Optional[Type[interfaces.ResourceWrapperBase]] = None
    for k, v in self._wrapped_types.items():
      if issubclass(t, k):
        if found:
          raise RegistryError(
            f'Found multiple wrappers for type {t}: {found} and {v}')
        found = v
    return found

  def wrap(self, value: Any) -> interfaces.ResourceBase:
    """Wrap a value if necessary."""
    if isinstance(value, interfaces.ResourceBase):
      return value
    if isinstance(value, Hashable):
      try:
        instance_wrapper = self._wrapped_instances.get(value)
      except TypeError:
        instance_wrapper = None
      if instance_wrapper:
        return instance_wrapper
    wrapper = self.get_wrapper_type(type(value))
    if wrapper:
      return wrapper.wrap(value)
    raise MissingWrapperError(
      f'Cannot wrap value of type {type(value)}. Please register '
      f'a ResourceWrapper for this type.')

  def wrap_type(self, value: Type) -> Type[interfaces.ResourceBase]:
    """Wrap a type if necessary."""
    if issubclass(value, interfaces.ResourceBase):
      return value
    wrapper = self.get_wrapper_type(value)
    if wrapper:
      return wrapper
    raise MissingWrapperError(
      f'Cannot wrap type {value}. Please register '
      f'a ResourceWrapper for this type.')

  def unwrap(self, value: interfaces.ResourceBase) -> Any:
    """Unwrap a value if wrapped."""
    if (isinstance(value, resources.ImmutableResource) and
        value in self._instance_wrappers):
      return self._instance_wrappers[value]
    if isinstance(value, interfaces.ResourceWrapperBase):
      return value.unwrap()
    return value

  def unwrap_type(self, value: Type[interfaces.ResourceBase]) -> Type:
    """Unwrap a type if wrapped."""
    if issubclass(value, interfaces.ResourceWrapperBase):
      return value.wrapped_type()
    return value

  @classmethod
  def get(cls) -> 'Registry':
    """Returns the singleton registry."""
    if not cls._singleton:
      cls._singleton = cls()
    return cls._singleton


ResourceT = TypeVar('ResourceT', bound=interfaces.ResourceBase)


def register(cls: Type[ResourceT]) -> Type[ResourceT]:
  """Decorator for resource classes."""
  Registry.get().register(cls)
  return cls


def register_instance_wrapper(
    instance: Hashable, wrapper: resources.ImmutableResource):
  """Decorator for resource classes."""
  Registry.get().register_instance_wrapper(instance, wrapper)


def register_wrapper(cls: Type[WrapperT]) -> Type[WrapperT]:
  """Decorator for resource wrapper classes."""
  # pylint: disable=protected-access
  Registry.get()._register_resource_wrapper(cls)
  return cls


def wrap(value: Any) -> interfaces.ResourceBase:
  """Wrap a value as a resource."""
  return Registry.get().wrap(value)


def unwrap(value: interfaces.ResourceBase) -> Any:
  """Wrap a value."""
  return Registry.get().unwrap(value)


def wrap_type(value: Type) -> Type[interfaces.ResourceBase]:
  """Wrap a type as a resource."""
  return Registry.get().wrap_type(value)


def unwrap_type(value: Type[interfaces.ResourceBase]) -> Type:
  """Wrap a type."""
  return Registry.get().unwrap_type(value)


def deepcopy(value: WrappedT) -> WrappedT:
  """Deep copy a value."""
  return from_field_value(to_field_value(value))


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
      issubclass(value, interfaces.ResourceWrapperBase)):
    return unwrap_type(value)
  if isinstance(value, list):
    return [from_field_value(x) for x in value]
  if isinstance(value, dict):
    return {k: from_field_value(v) for k, v in value.items()}
  return value

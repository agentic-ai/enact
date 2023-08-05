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

"""Core resource interface."""

import functools
import json
from typing import Dict, Generic, Iterable, Mapping, Sequence, Tuple, Type, TypeVar, Union


JsonLeaf = Union[int, float, str, bool, None]
Json = Union[JsonLeaf, Sequence['Json'], Dict[str, 'Json']]


PRIMITIVES = (int, float, str, bytes, bool, type(None))


Primitives = Union[
  int, float, str, bytes, bool,
  None]


FieldValue = Union[
  Primitives,
  'ResourceBase',
  Type['ResourceBase'],
  Sequence['FieldValue'],
  Mapping[str, 'FieldValue']]


ResourceDictValue = Union[
  Primitives,
  Type['ResourceBase'],
  Sequence['ResourceDictValue'],
  Mapping[str, 'ResourceDictValue']]


C = TypeVar('C', bound='ResourceBase')


class ResourceError(Exception):
  """Superclass for resource related errors."""


class FieldTypeError(Exception):
  """Superclass for errors related to field types."""


class ResourceBase:
  """Base class for resources.

  Not an abstract base class in order to avoid meta-class conflict.

  Resources have a unique type identifier. Each resource class is associated
  with a fixed list of named fields.
  """

  @classmethod
  def type_descr(cls) -> Mapping[str, Json]:
    """Returns a unique descriptor for the type."""
    return {'name': f'{cls.__module__}.{cls.__qualname__}'}

  @classmethod
  @functools.lru_cache
  def type_id(cls) -> str:
    """Returns a string descriptor of the type."""
    return json.dumps(cls.type_descr(), sort_keys=True)

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    raise NotImplementedError(f'{cls} does not implement field_names')

  def field_values(self) -> Iterable[FieldValue]:
    """Return a list of field values, aligned with field_names."""
    raise NotImplementedError(f'{type(self)} does not implement field_values')

  def field_items(self) -> Iterable[Tuple[str, FieldValue]]:
    """Iterate through the field names and values."""
    return zip(self.field_names(), self.field_values())

  @classmethod
  def from_fields(cls: Type[C],
                  field_dict: Mapping[str, FieldValue]) -> C:
    """Constructs the resource from a field dictionary."""
    raise NotImplementedError()

  @staticmethod
  def _to_dict_value(v: FieldValue) -> ResourceDictValue:
    """Transforms a field value to a resource dict value."""
    if isinstance(v, ResourceBase):
      return v.to_resource_dict()
    if isinstance(v, PRIMITIVES):
      return v
    if isinstance(v, type) and issubclass(v, ResourceBase):
      return v
    if isinstance(v, Sequence):
      return [ResourceBase._to_dict_value(x) for x in v]
    if isinstance(v, Mapping):
      def _assert_str(s: str) -> str:
        if not isinstance(s, str):
          raise FieldTypeError(
            f'Expected string key, got {type(s)}')
        return s
      return {
        _assert_str(k): ResourceBase._to_dict_value(v)
        for k, v in v.items()}
    raise FieldTypeError(
      f'Encountered unsupported field type {type(v)}: {v}')

  @staticmethod
  def _from_dict_value(v: ResourceDictValue) -> FieldValue:
    """Transforms a resource dict value to a field value."""
    if isinstance(v, PRIMITIVES):
      return v
    if isinstance(v, type) and issubclass(v, ResourceBase):
      return v
    if isinstance(v, Sequence):
      return [ResourceBase._from_dict_value(x) for x in v]
    if isinstance(v, ResourceDict):
      return v.type.from_resource_dict(v)
    if isinstance(v, Mapping):
      def _assert_str(s: str) -> str:
        if not isinstance(s, str):
          raise FieldTypeError(
            f'Expected string key, got {type(s)}')
        return s
      return {
        _assert_str(k): ResourceBase._from_dict_value(v)
        for k, v in v.items()}
    raise FieldTypeError(
      f'Encountered unsupported resource '
      f'dict value type {type(v)}: {v}')

  def deep_copy_resource(self: C) -> C:
    """Create a deep-copy of the resource."""
    return self.from_resource_dict(self.to_resource_dict())

  def to_resource_dict(self) -> 'ResourceDict':
    """Returns a ResourceDict dictionary representation."""
    result = ResourceDict(type(self))
    for k, v in self.field_items():
      result[k] = ResourceBase._to_dict_value(v)
    return result

  @classmethod
  def from_resource_dict(cls: Type[C], d: 'ResourceDict') -> C:
    """Constructs the resource from a ResourceDict dictionary."""
    if not issubclass(d.type, cls):
      raise ResourceError(
        f'Expected resource of type {cls}, got {d.type}')
    field_dict = {
      k: ResourceBase._from_dict_value(v)
      for k, v in d.items()}
    return cls.from_fields(field_dict)

  def set_from(self: C, other: C):
    """Sets the fields of this resource from another resource.

    Implementation of set_from is required to support replays of invokable
    resources that change their internal state during execution.

    Args:
      other: The resource to set fields from.
    """
    raise NotImplementedError(
      f'Setting fields from another resource is not '
      f'supported by type {type(self)}.')


class ResourceDict(Generic[C], dict, Mapping[str, ResourceDictValue]):
  """A dictionary representing a resource with attached type info."""

  def __init__(self, resource_type: Type[C], *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.type = resource_type

  def to_resource(self) -> C:
    """Constructs the resource from the dictionary."""
    return self.type.from_resource_dict(self)


class NoneResource(ResourceBase):
  """The None resource."""

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    return ()

  def field_values(self) -> Iterable[FieldValue]:
    """Return a list of field values, aligned with field_names."""
    return ()

  @classmethod
  def from_fields(
      cls: Type[C],
      field_dict: Mapping[str, FieldValue]) -> C:
    """Constructs the resource from a value dictionary."""
    assert not field_dict
    return cls()

  def set_from(self: C, other: C):
    """Sets the fields of this resource from another resource."""
    raise NotImplementedError(
      f'Setting fields from another resource is not '
      f'supported by type {type(self)}.')

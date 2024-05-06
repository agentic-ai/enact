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

import abc
import functools
import json
from typing import (
  Dict, Generic, Iterable, List, NamedTuple, Optional, Tuple, Type, TypeVar,
  Union, cast)

from enact import acyclic


JsonLeaf = Union[int, float, str, bool, None]
Json = Union[JsonLeaf, List['Json'], Dict[str, 'Json']]


PRIMITIVES = (int, float, str, bytes, bool, type(None))


Primitives = Union[
  int, float, str, bytes, bool,
  None]


FieldValue = Union[
  Primitives,
  'ResourceBase',
  Type['ResourceBase'],
  List['FieldValue'],
  Dict[str, 'FieldValue']]


ResourceDictValue = Union[
  Primitives,
  Type['ResourceBase'],
  List['ResourceDictValue'],
  Dict[str, 'ResourceDictValue'],
  'ResourceDict']


C = TypeVar('C', bound='ResourceBase')


class FrameworkError(Exception):
  """Superclass for framework related errors."""


class ResourceError(FrameworkError):
  """Superclass for resource related errors."""


class ImplementationMissing(FrameworkError):
  """Superclass for errors where an implementaton is missing."""


class FieldTypeError(FrameworkError):
  """Superclass for errors related to field types."""


class TypeInfo(NamedTuple):
  """Information about a resource type."""
  name: str
  distribution_info: Optional['DistributionInfo']

  def type_id(self) -> str:
    """Returns a unique string identifier for the type."""
    return json.dumps(self.as_dict(), sort_keys=True)

  def as_dict(self) -> Dict[str, Json]:
    """Returns a dictionary representation of the distribution info."""
    dist_info = (
      self.distribution_info.as_dict() if self.distribution_info else None)
    return {
      'name': self.name,
      'distribution_info': dist_info}

  @staticmethod
  def from_dict(d: Dict[str, Json]) -> 'TypeInfo':
    """Instantiate TypeInfo from a dictionary."""
    r: Dict = dict(d)
    r['distribution_info'] = DistributionInfo.from_dict(r['distribution_info'])
    return TypeInfo(**r)


class DistributionInfo(NamedTuple):
  """Information about a package where a resource is defined.

  This information (together with qualified class names) is used to identify
  compatible resources across different versions of a package.
  """
  name: str
  version: str

  def as_dict(self) -> Dict[str, Json]:
    """Returns a dictionary representation of the distribution info."""
    return {'name': self.name, 'version': self.version}

  @staticmethod
  def from_dict(d: Dict[str, Json]) -> 'DistributionInfo':
    """Instantiate DistributionInfo from a dictionary."""
    return DistributionInfo(**cast(Dict[str, str], d))


class ResourceBase:
  """Base class for resources.

  Not an abstract base class in order to avoid meta-class conflict.

  Resources have a unique type identifier. Each resource class is associated
  with a fixed list of named fields. These fields must have value semantics,
  that is, replacing a field by a copy should not change the meaning of the
  resource. This means, for example, that code using resources should not rely
  on aliasing assumptions (e.g., two resources sharing the same list instance).

  In particular, this also means that resources may not mutually reference each
  other.
  """
  _enact_distribution_info: Optional[DistributionInfo] = None

  @classmethod
  def type_info(cls) -> TypeInfo:
    """Returns a descriptor for the type."""
    return TypeInfo(
      name=f'{cls.__module__}.{cls.__qualname__}',
      distribution_info=cls.type_distribution_info())

  @classmethod
  def type_distribution_info(cls) -> Optional[DistributionInfo]:
    """Returns package information for the type if set."""
    return cls._enact_distribution_info

  @classmethod
  def set_type_distribution_info(cls, info: DistributionInfo):
    """Sets the package information for the type."""
    cls._enact_distribution_info = info

  @classmethod
  @functools.lru_cache
  def type_id(cls) -> str:
    """Returns a string descriptor of the type."""
    return cls.type_info().type_id()

  @classmethod
  @abc.abstractmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    raise NotImplementedError(f'{cls} does not implement field_names')

  @abc.abstractmethod
  def field_values(self) -> Iterable[FieldValue]:
    """Return a list of field values, aligned with field_names."""
    raise NotImplementedError(f'{type(self)} does not implement field_values')

  def field_items(self) -> Iterable[Tuple[str, FieldValue]]:
    """Iterate through the field names and values."""
    return zip(self.field_names(), self.field_values())

  @classmethod
  @abc.abstractmethod
  def from_fields(cls: Type[C],
                  field_dict: Dict[str, FieldValue]) -> C:
    """Constructs the resource from a field dictionary."""
    raise NotImplementedError()

  @staticmethod
  def _to_dict_value(value: FieldValue) -> ResourceDictValue:
    """Transforms a field value to a resource dict value."""
    with acyclic.AcyclicContext(value):
      if isinstance(value, ResourceBase):
        return value.to_resource_dict()
      if isinstance(value, PRIMITIVES):
        return value
      if isinstance(value, type) and issubclass(value, ResourceBase):
        return value
      if isinstance(value, List):
        return [ResourceBase._to_dict_value(x) for x in value]
      if isinstance(value, Dict):
        def _assert_str(maybe_str: str) -> str:
          if type(maybe_str) is not str:  # pylint: disable=unidiomatic-typecheck
            raise FieldTypeError(
              f'Expected string key, got {type(maybe_str)}')
          return maybe_str
        return {
          _assert_str(k): ResourceBase._to_dict_value(v)
          for k, v in value.items()}
      raise FieldTypeError(
        f'Encountered unsupported field type {type(value)}: {value}')

  def to_resource_dict(self: C) -> 'ResourceDict[C]':
    """Returns a ResourceDict dictionary representation."""
    result = ResourceDict(type(self))
    for field_name, value in self.field_items():
      result[field_name] = ResourceBase._to_dict_value(value)
    return result

  def set_from(self, other: 'ResourceBase'):
    """Sets the fields of this resource from another resource.

    Implementation of set_from is required to support replays of invokable
    resources that change their internal state during execution.

    Args:
      other: The resource to set fields from.
    """

    raise ImplementationMissing(
      f'Setting fields from another resource is not '
      f'supported by type {type(self)}.')


class ResourceDict(Generic[C], Dict[str, ResourceDictValue]):
  """A dictionary representing a resource with attached TypeInfo."""

  def __init__(
      self, resource_type: Union[Type[C], TypeInfo], *args, **kwargs):
    super().__init__(*args, **kwargs)
    if not isinstance(resource_type, TypeInfo):
      resource_type = resource_type.type_info()
    self.type_info = resource_type


WrappedT = TypeVar('WrappedT')
WrapperT = TypeVar('WrapperT', bound='TypeWrapperBase')


class TypeWrapperBase(ResourceBase, Generic[WrappedT]):
  """Interface for resource classes that wrap python classes"""
  @classmethod
  @abc.abstractmethod
  def wrapped_type(cls) -> Type[WrappedT]:
    """Returns the type of the wrapped value."""
    raise NotImplementedError()

  @classmethod
  @abc.abstractmethod
  def wrap(cls: Type[WrapperT], value: WrappedT) -> WrapperT:
    """Wrap a value."""
    raise NotImplementedError()

  @abc.abstractmethod
  def unwrap(self) -> WrappedT:
    """Wrap a value."""
    raise NotImplementedError()

  @classmethod
  def is_immutable(cls) -> bool:
    """Whether the wrapped value is immutable."""
    return False

  @classmethod
  def set_wrapped_value(cls, target: WrappedT, src: WrappedT):
    """Set a wrapped value target to correspond to source."""
    raise ImplementationMissing(
      f'Please implement set_wrapped_value for TypeWrapper'
      f'{cls} to enable advanced features, e.g., replays.')


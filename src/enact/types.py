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
"""A serializable type description language."""

import abc
import dataclasses
import typing
import json


JsonLeaf = typing.Union[int, float, str, bool, None]
Json = typing.Union[JsonLeaf, typing.List['Json'], typing.Dict[str, 'Json']]

PRIMITIVES = (int, float, str, bytes, bool, type(None))

Primitives = typing.Union[
  int, float, str, bytes, bool, None]


class TypeKey(typing.NamedTuple):
  """Information about a resource type."""
  name: str
  distribution_key: typing.Optional['DistributionKey']

  def type_id(self) -> str:
    """Returns a unique string identifier for the type."""
    return json.dumps(self.as_dict(), sort_keys=True)

  def as_dict(self) -> typing.Dict[str, Json]:
    """Returns a dictionary representation of the distribution key."""
    dist_info = (
      self.distribution_key.as_dict() if self.distribution_key else None)
    return {
      'name': self.name,
      'distribution_key': dist_info}

  @staticmethod
  def from_dict(d: typing.Dict[str, Json]) -> 'TypeKey':
    """Instantiate TypeKey from a dictionary."""
    r: typing.Dict = dict(d)
    dist_key = r['distribution_key']
    if dist_key is not None:
      r['distribution_key'] = DistributionKey.from_dict(r['distribution_key'])
    return TypeKey(**r)


class DistributionKey(typing.NamedTuple):
  """Information about a package where a resource is defined.

  This information (together with qualified class names) is used to identify
  compatible resources across different versions of a package.
  """
  name: str
  version: str

  def as_dict(self) -> typing.Dict[str, Json]:
    """Returns a dictionary representation of the distribution key."""
    return {'name': self.name, 'version': self.version}

  @staticmethod
  def from_dict(d: typing.Dict[str, Json]) -> 'DistributionKey':
    """Instantiate DistributionKey from a dictionary."""
    return DistributionKey(**typing.cast(typing.Dict[str, str], d))


@dataclasses.dataclass(frozen=True)
class TypeDescriptor(abc.ABC):
  """Interface for type descriptors."""
  NAME: typing.ClassVar[str] = ''

  def to_json(self) -> Json:
    """JSON representation."""
    return self.NAME

  @staticmethod
  def from_json(json_value: Json) -> 'TypeDescriptor':
    """Constructs a descriptor from JSON."""
    cls: typing.Type[TypeDescriptor]
    for cls in BASIC_TYPE_DESCRIPTOR_CLASSES:
      if json_value == cls.NAME:
        return cls()
    if not isinstance(json_value, dict) or len(json_value) != 1:
      raise ValueError(
        f'Expected a dictionary of length 1 for complex type descriptor: '
        f'{json_value}')
    if ResourceType.NAME in json_value:
      value = json_value[ResourceType.NAME]
      if not isinstance(value, dict):
        raise ValueError(
          f'Expected a dictionary for resource type descriptor: '
          f'{json_value}')
      return ResourceType(TypeKey.from_dict(value))
    elif List.NAME in json_value:
      value = json_value[List.NAME]
      if value is None:
        return List(None)
      return List(TypeDescriptor.from_json(value))
    elif Dict.NAME in json_value:
      value = json_value[Dict.NAME]
      if value is None:
        return Dict(None)
      return Dict(TypeDescriptor.from_json(value))
    elif Union.NAME in json_value:
      values = json_value[Union.NAME]
      assert isinstance(values, list)
      return Union(
        tuple(TypeDescriptor.from_json(value) for value in values))
    raise ValueError(f'Unknown type descriptor: {json_value}')

  def pformat(self) -> str:
    """Pretty formats the descriptor."""
    return self.NAME


@dataclasses.dataclass(frozen=True)
class Int(TypeDescriptor):
  """Describes an int value."""
  NAME: typing.ClassVar[str] = 'int'


@dataclasses.dataclass(frozen=True)
class Float(TypeDescriptor):
  """Describes a float value."""
  NAME: typing.ClassVar[str] = 'float'


@dataclasses.dataclass(frozen=True)
class Str(TypeDescriptor):
  """Describes a string value."""
  NAME: typing.ClassVar[str] = 'str'


@dataclasses.dataclass(frozen=True)
class Bool(TypeDescriptor):
  """Describes a boolean value."""
  NAME: typing.ClassVar[str] = 'bool'


@dataclasses.dataclass(frozen=True)
class Bytes(TypeDescriptor):
  """Describes a bytes value."""
  NAME: typing.ClassVar[str] = 'bytes'


@dataclasses.dataclass(frozen=True)
class List(TypeDescriptor):
  """Describes a list value."""
  NAME: typing.ClassVar[str] = 'list'
  value_type: typing.Optional[TypeDescriptor] = None

  def to_json(self) -> Json:
    """JSON representation."""
    if self.value_type is None:
      return self.NAME
    return {self.NAME: self.value_type.to_json()}

  def pformat(self) -> str:
    """Pretty formats the descriptor."""
    if self.value_type is not None:
      return f'{self.NAME}[{self.value_type.pformat()}]'
    return self.NAME


@dataclasses.dataclass(frozen=True)
class Dict(TypeDescriptor):
  """Describes a dictionary value."""
  NAME: typing.ClassVar[str] = 'dict'
  value_type: typing.Optional[TypeDescriptor] = None

  def to_json(self) -> Json:
    """JSON representation."""
    if self.value_type is None:
      return self.NAME
    return {self.NAME: self.value_type.to_json()}

  def pformat(self) -> str:
    """Pretty formats the descriptor."""
    if self.value_type is not None:
      return f'{self.NAME}[{self.value_type.pformat()}]'
    return self.NAME


@dataclasses.dataclass(frozen=True)
class Union(TypeDescriptor):
  """Describes an optional value."""
  NAME: typing.ClassVar[str] = 'optional'
  value_types: typing.Tuple[TypeDescriptor, ...]

  def to_json(self) -> Json:
    """JSON representation."""
    return {self.NAME: [
      value_type.to_json() for value_type in self.value_types]}

  def pformat(self) -> str:
    """Pretty formats the descriptor."""
    value_pformats = [value_type.pformat() for value_type in self.value_types]
    return f'{self.NAME}[{", ".join(value_pformats)}]'


@dataclasses.dataclass(frozen=True)
class NoneType(TypeDescriptor):
  """Describes a None type."""
  NAME: typing.ClassVar[str] = 'none'


@dataclasses.dataclass(frozen=True)
class ResourceType(TypeDescriptor):
  """Describes a resource type."""
  NAME: typing.ClassVar[str] = 'resource'
  type_key: TypeKey

  def to_json(self):
    return {self.NAME: self.type_key.as_dict()}

  def pformat(self) -> str:
    """Pretty formats the descriptor."""
    if self.type_key.distribution_key:
      return f'{self.type_key.name} ({self.type_key.distribution_key.name})'
    return self.type_key.name


BASIC_TYPE_DESCRIPTOR_CLASSES = (
  Int, Float, Str, Bool, Bytes, List, Dict, NoneType)

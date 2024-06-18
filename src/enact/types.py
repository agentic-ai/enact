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
"""A serializable type description language."""

import abc
import dataclasses
import typing
import json


JsonLeaf = typing.Union[int, float, str, bool, None]
Json = typing.Union[JsonLeaf, typing.List['Json'], typing.Dict[str, 'Json']]

INT_NAME = 'int'
STR_NAME = 'str'
FLOAT_NAME = 'float'
BOOL_NAME = 'bool'
BYTES_NAME = 'bytes'
LIST_NAME = 'list'
DICT_NAME = 'dict'
RESOURCE_NAME = 'resource'


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


@dataclasses.dataclass  # Type descriptors are dataclasses for value-semantics.
class TypeDescriptor(abc.ABC):
  """Interface for type descriptors."""

  @abc.abstractmethod
  def to_json(self) -> Json:
    """JSON representation."""

  @staticmethod
  def from_json(json_value: Json) -> 'TypeDescriptor':
    """Constructs a descriptor from JSON."""
    if json_value == INT_NAME:
      return Int()
    if json_value == STR_NAME:
      return Str()
    if json_value == FLOAT_NAME:
      return Float()
    if json_value == BOOL_NAME:
      return Bool()
    if json_value == BYTES_NAME:
      return Bytes()
    if json_value == LIST_NAME:
      return List()
    if json_value == DICT_NAME:
      return Dict()
    if not isinstance(json_value, dict) or len(json_value) != 1:
      raise ValueError(
        f'Expected a dictionary of length 1 for complex type descriptor: '
        f'{json_value}')
    if RESOURCE_NAME in json_value:
      value = json_value[RESOURCE_NAME]
      if not isinstance(value, dict):
        raise ValueError(f'Expected a dictionary for resource type descriptor: '
                         f'{json_value}')
      return ResourceType(TypeKey.from_dict(value))
    raise ValueError(f'Unknown type descriptor: {json_value}')

@dataclasses.dataclass
class Int(TypeDescriptor):
  """Describes an int value."""

  def to_json(self):
    return INT_NAME


@dataclasses.dataclass
class Float(TypeDescriptor):
  """Describes a float value."""

  def to_json(self):
    return FLOAT_NAME


@dataclasses.dataclass
class Str(TypeDescriptor):
  """Describes a string value."""

  def to_json(self):
    return STR_NAME


@dataclasses.dataclass
class Bool(TypeDescriptor):
  """Describes a boolean value."""

  def to_json(self):
    return BOOL_NAME


@dataclasses.dataclass
class Bytes(TypeDescriptor):
  """Describes a bytes value."""

  def to_json(self):
    return BYTES_NAME


@dataclasses.dataclass
class List(TypeDescriptor):
  """Describes a list value."""

  def to_json(self):
    return LIST_NAME


@dataclasses.dataclass
class Dict(TypeDescriptor):
  """Describes a dictionary value."""

  def to_json(self):
    return DICT_NAME


@dataclasses.dataclass
class ResourceType(TypeDescriptor):
  """Describes a resource type."""

  def __init__(self, type_key: TypeKey):
    self.type_key = type_key

  def to_json(self):
    return {RESOURCE_NAME: self.type_key.as_dict()}

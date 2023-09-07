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

"""Serialization functionality."""

import abc
import base64
import json
from typing import Dict, Mapping, Optional, Sequence, Union, cast

from enact import interfaces
from enact import resource_registry


JsonLeaf = Union[int, float, str, bool, None]
Json = Union[JsonLeaf, Sequence['Json'], Dict[str, 'Json']]

JSON_LEAF_TYPES = (int, float, str, bool, type(None))


class SerializationError(Exception):
  """Raised when serialization fails."""


class DeserializationError(Exception):
  """Raised when deserialization fails."""


class Serializer(abc.ABC):
  """Serializes and deserializes resources."""

  @abc.abstractmethod
  def serialize(self, resource_dict: interfaces.ResourceDict) -> bytes:
    """Serializes a resource."""

  @abc.abstractmethod
  def deserialize(self, data: bytes) -> interfaces.ResourceDict:
    """Deserializes a packed resource."""


class JsonSerializer(Serializer):
  """A JSON serializer."""

  _ESCAPE = '#'

  def __init__(self,
               registry: Optional[resource_registry.Registry]=None,
               encoding='utf-8'):
    """Initializes a JSON serializer."""
    self._registry = registry or resource_registry.Registry.get()
    self._encoding = encoding

  def _escape(self, s: str):
    """Escapes a string."""
    return f'{self._ESCAPE}{s}'

  def to_json(self, value: interfaces.ResourceDictValue) -> Json:
    """Converts a value to a JSON compatible value recursively."""
    if isinstance(value, interfaces.ResourceDict):
      result: Dict[str, Json] = {
        self._escape('res'): (value.type.type_id())}
      for k, v in value.items():
        if not isinstance(k, str):
          raise interfaces.FieldTypeError(
            f'Resource has non-string keys: {value}')
        if k.startswith(self._ESCAPE):
          raise SerializationError(
            f'Cannot serialize resource with fields starting with '
            f'{self._ESCAPE}: {value}')
        result[k] = self.to_json(v)
      return result
    if isinstance(value, JSON_LEAF_TYPES):
      return value
    if isinstance(value, bytes):
      return {
        self._escape('b85'): base64.b85encode(value).decode('ascii')}
    if isinstance(value, type):
      if not issubclass(value, interfaces.ResourceBase):
        raise SerializationError(
          f'Cannot serialize type: {value} which is not a resource')
      return {
        self._escape('type'): value.type_id()}
    if isinstance(value, Mapping):
      result = {}
      for k, v in value.items():
        if not isinstance(k, str):
          raise interfaces.FieldTypeError(
            f'Mapping has non-string keys: {value}')
        if k.startswith(self._ESCAPE):
          raise SerializationError(
            f'Cannot serialize mapping with keys starting with '
            f'{self._ESCAPE}: {value}')
        result[k] = self.to_json(v)
      return result
    if isinstance(value, Sequence):
      return [self.to_json(v) for v in value]
    raise SerializationError(
      f'Cannot serialize value: {value} of type {type(value)}')

  def _resource_dict_from_json(
      self,
      value: Mapping[str, Json]) -> interfaces.ResourceDict:
    """Parse a resource from a json representation."""
    type_id_key = self._escape('res')
    fields = dict(value)
    type_id = fields.pop(type_id_key, None)
    if not type_id:
      raise DeserializationError(
        f'Could not find {type_id_key} in JSON: {value}')
    if not isinstance(type_id, str):
      raise DeserializationError(
        f'Expected {type_id_key} to be a string, '
        f'got {type(type_id)}: {value}')
    resource_class = self._registry.lookup(type_id)
    return interfaces.ResourceDict(resource_class, self.from_json(fields))

  def from_json(self, value: Json) -> interfaces.ResourceDictValue:
    """Turn a json encodable dictionary into a field value."""
    # Check encoded resource:
    type_id_key = self._escape('res')
    b85_key = self._escape('b85')
    resource_type_key = self._escape('type')
    if isinstance(value, JSON_LEAF_TYPES):
      return value
    if isinstance(value, Mapping):
      if type_id_key in value:
        ref = self._resource_dict_from_json(value)
        return ref
      if b85_key in value:
        b85str = cast(str, value[b85_key])
        return base64.b85decode(b85str)
      if resource_type_key in value:
        type_id = cast(str, value[resource_type_key])
        return self._registry.lookup(type_id)
      return {k: self.from_json(v) for k, v in value.items()}
    elif isinstance(value, Sequence):
      return [self.from_json(v) for v in value]
    else:
      raise DeserializationError(
        f'Cannot deserialize value: {value} of type {type(value)}')

  def serialize(self, resource_dict: interfaces.ResourceDict) -> bytes:
    """Serializes a resource dictionary."""
    json_dict = self.to_json(resource_dict)
    return json.dumps(json_dict, sort_keys=True).encode(self._encoding)

  def deserialize(self, data: bytes) -> interfaces.ResourceDict:
    """Deserializes a resource."""
    json_dict = json.loads(data.decode(self._encoding))
    assert isinstance(json_dict, dict)
    return self._resource_dict_from_json(json_dict)

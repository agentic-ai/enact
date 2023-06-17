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

"""Computes hash digests of resources."""

import hashlib
from typing import Any, List, Mapping, Sequence, Type, Union

from enact import interfaces


Value = Union[
  interfaces.FieldValue,
  interfaces.ResourceDictValue]


def type_digest(cls: Type[interfaces.ResourceBase]) -> str:
  """Return a digest of the type of resource."""
  hash = hashlib.sha256()
  hash.update(cls.__module__.encode('utf-8'))
  hash.update(b'.')
  hash.update(cls.__qualname__.encode('utf-8'))
  for field in cls.field_names():
    hash.update(repr(field).encode('utf-8'))
  return hash.hexdigest()


def _digest(
    value: Value,
    hash: Any,
    stack: List[int]):
  """Recursively compute digest over a field value.

  Args:
    value: The value to digest.
    hash: A hash object with an update method, e.g., hashlib.sha256().
  """
  if id(value) in stack:
    raise interfaces.FieldTypeError(
      'Cyclic references are not allowed in field values.')
  stack.append(id(value))
  if isinstance(value, (interfaces.ResourceBase,
                        interfaces.ResourceDict)):
    if isinstance(value, interfaces.ResourceBase):
      res_type = type(value)
      items = value.field_items()
    else:
      assert isinstance(value, interfaces.ResourceDict)
      res_type = value.type
      items = value.items()
    hash.update(b'res[')
    hash.update(res_type.type_id().encode('utf-8'))
    for k, v in items:
      hash.update(repr(k).encode('utf-8'))
      _digest(v, hash, stack)
    hash.update(b']')
  elif isinstance(value, int):
    hash.update(b'i')
    hash.update(repr(int(value)).encode('utf-8'))
  elif isinstance(value, float):
    hash.update(b'f')
    hash.update(repr(value).encode('utf-8'))
  elif isinstance(value, str):
    hash.update(b's')
    hash.update(repr(value).encode('utf-8'))
  elif isinstance(value, bytes):
    hash.update(b'b')
    hash.update(value)
  elif value is True:
    hash.update(b'1')
  elif value is False:
    hash.update(b'0')
  elif value is None:
    hash.update(b'n')
  elif isinstance(value, Sequence):
    hash.update(b'seq[')
    for item in value:
      _digest(item, hash, stack)
    hash.update(b']')
  elif isinstance(value, Mapping):
    hash.update(b'map[')
    for k, v in value.items():
      if not isinstance(k, str):
        raise interfaces.FieldTypeError('Map keys must be strings')
      hash.update(repr(k).encode('utf-8'))
      _digest(v, hash, stack)
    hash.update(b']')
  else:
    raise interfaces.FieldTypeError(
      f'Got unexpected field type: {type(value)}. '
      f'Allowed fields types are: int, float, str, bytes, bool, None and '
      f'Ref, and nested maps from strings or sequences of these types.')
  stack.pop()


def digest(resource: Union[interfaces.ResourceDict,
                           interfaces.ResourceBase]) -> str:
  """Compute a digest of a resource or a dict representation."""
  hash = hashlib.sha256()
  _digest(resource, hash, [])
  return hash.hexdigest()

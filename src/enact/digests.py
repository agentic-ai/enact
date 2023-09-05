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
  hash_obj = hashlib.sha256()
  hash_obj.update(cls.__module__.encode('utf-8'))
  hash_obj.update(b'.')
  hash_obj.update(cls.__qualname__.encode('utf-8'))
  for field in sorted(cls.field_names()):
    hash_obj.update(repr(field).encode('utf-8'))
  return hash_obj.hexdigest()


def _digest(
    value: Value,
    hash_obj: Any,
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
      # Use alphabetical ordering.
      items = sorted(value.field_items(), key=lambda x: x[0])
    else:
      assert isinstance(value, interfaces.ResourceDict)
      res_type = value.type
      items = sorted(value.items(), key=lambda x: x[0])
    hash_obj.update(b'res[')
    hash_obj.update(res_type.type_id().encode('utf-8'))
    for k, v in items:
      hash_obj.update(repr(k).encode('utf-8'))
      _digest(v, hash_obj, stack)
    hash_obj.update(b']')
  elif isinstance(value, int):
    hash_obj.update(b'i')
    hash_obj.update(repr(int(value)).encode('utf-8'))
  elif isinstance(value, float):
    hash_obj.update(b'f')
    hash_obj.update(repr(value).encode('utf-8'))
  elif isinstance(value, str):
    hash_obj.update(b's')
    hash_obj.update(repr(value).encode('utf-8'))
  elif isinstance(value, bytes):
    hash_obj.update(b'b')
    hash_obj.update(value)
  elif value is True:
    hash_obj.update(b'1')
  elif value is False:
    hash_obj.update(b'0')
  elif value is None:
    hash_obj.update(b'n')
  elif isinstance(value, Sequence):
    hash_obj.update(b'seq[')
    for item in value:
      _digest(item, hash_obj, stack)
    hash_obj.update(b']')
  elif isinstance(value, Mapping):
    hash_obj.update(b'map[')
    for k, v in sorted(value.items(), key=lambda x: x[0]):  # type: ignore
      if not isinstance(k, str):
        raise interfaces.FieldTypeError('Map keys must be strings')
      hash_obj.update(repr(k).encode('utf-8'))
      _digest(v, hash_obj, stack)
    hash_obj.update(b']')
  elif issubclass(value, interfaces.ResourceBase):
    # Type of resource.
    hash_obj.update(b'type[')
    hash_obj.update(value.type_id().encode('utf-8'))
    hash_obj.update(b']')
  else:
    raise interfaces.FieldTypeError(
      f'Got unexpected field type: {type(value)}. '
      f'Allowed fields types are: int, float, str, bytes, bool, None and '
      f'Ref, and nested maps from strings or sequences of these types.')
  stack.pop()


def digest(resource: Union[interfaces.ResourceDict,
                           interfaces.ResourceBase]) -> str:
  """Compute a digest of a resource or a dict representation."""
  hash_obj = hashlib.sha256()
  _digest(resource, hash_obj, [])
  return hash_obj.hexdigest()

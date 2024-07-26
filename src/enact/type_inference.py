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
"""Automatic field type inference."""

from typing import Any, Optional, List, Dict, Tuple, Union, TypeVar, cast

from enact import interfaces
from enact import types
from enact import resource_registry

class TypeInferenceFailed(Exception):
  """Raised when type inference fails."""
  pass

def from_annotation(t: Any, strict: bool = False) -> (
    Optional[types.TypeDescriptor]):
  """Attempts to infer a type descriptor from a python type annotation.

  Args:
    t: A type annotation.
    strict: If True, raise an exception if the type cannot be inferred.

  Returns:
    A type descriptor or None if the type cannot be inferred.
  Raises:
    TypeInferenceFailed: If strict is True and the type cannot be inferred.
  """
  result: Optional[types.TypeDescriptor] = None
  failure_reason: str = ''
  if t == int:
    result = types.Int()
  elif t == str:
    result = types.Str()
  elif t == float:
    result = types.Float()
  elif t == bool:
    result = types.Bool()
  elif t == bytes:
    result = types.Bytes()
  elif t == list:
    result = types.List()
  elif t == dict:
    result = types.Dict()
  elif t == type(None):
    result = types.NoneType()
  elif hasattr(t, '__origin__') and t.__origin__ in (List, list):
    if hasattr(t, '__args__'):
      if len(t.__args__) != 1:
        failure_reason = 'List must have exactly one type argument'
      else:
        if type(  # pylint: disable=unidiomatic-typecheck
            t.__args__[0]) is TypeVar:
          result = types.List()
          failure_reason = 'Untyped list'
        else:
          result = types.List(from_annotation(t.__args__[0], strict=strict))
    else:
      result = types.List()
  elif hasattr(t, '__origin__') and t.__origin__ in (Dict, dict):
    if hasattr(t, '__args__'):
      if len(t.__args__) != 2:
        failure_reason = 'Dict should have exactly two type arguments.'
      else:
        arg1, arg2 = t.__args__
        # pylint: disable=unidiomatic-typecheck
        if type(arg1) is TypeVar and type(arg2) is TypeVar:
          # Just 'Dict' will have two type vars as args.
          result = types.Dict()
        elif t.__args__[0] != str:
          failure_reason = (
            'Only dicts from str are supported at the moment.')
        else:
          result = types.Dict(from_annotation(t.__args__[1], strict=strict))
    else:
      result = types.Dict()
  elif hasattr(t, '__origin__') and t.__origin__ == Union:
    if not hasattr(t, '__args__') or len(t.__args__) == 0:
      failure_reason = 'Could not parse Union type'
    else:
      value_types = tuple(
        from_annotation(arg, strict=strict) for arg in t.__args__)
      if any(value_type is None for value_type in value_types):
        failure_reason = 'Failed to parse value type of Union'
      else:
        value_types = cast(Tuple[types.TypeDescriptor, ...], value_types)
        result = types.Union(value_types)
  elif t is None:
    failure_reason = 'Missing type annotation'
  elif isinstance(t, type):
    try:
      t = resource_registry.wrap_type(t)
    except resource_registry.MissingWrapperError:
      if strict:
        failure_reason = 'Unsupported type'
    if not failure_reason and issubclass(t, interfaces.ResourceBase):
      result = types.ResourceType(t.type_key())
  else:
    failure_reason = 'Unsupported type'
  if strict and failure_reason:
    raise TypeInferenceFailed(
      f'Failed to infer type from annotation. {failure_reason}: {t}')
  return result

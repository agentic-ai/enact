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

from typing import Any, Optional

from enact import interfaces, types, resource_registry


def from_annotation(t: Any) -> Optional[types.TypeDescriptor]:
  """Attempts to infer a type descriptor from a python type annotation."""
  if t == int:
    return types.Int()
  if t == str:
    return types.Str()
  if t == float:
    return types.Float()
  if t == bool:
    return types.Bool()
  if t == bytes:
    return types.Bytes()
  if t == list:
    return types.List()
  if t == dict:
    return types.Dict()
  if hasattr(t, '__origin__') and t.__origin__ in (types.List, list):
    return types.List()
  if hasattr(t, '__origin__') and t.__origin__ in (types.Dict, dict):
    return types.Dict()
  if isinstance(t, type):
    try:
      t = resource_registry.wrap_type(t)
    except resource_registry.MissingWrapperError:
      # We could raise here since this is not allowed, but we'll just ignore it
      # to not annoy users unless they need enact-based functionality, e.g.,
      # commits.
      return None
    if issubclass(t, interfaces.ResourceBase):
      return types.ResourceType(t.type_key())
  return None  # Could not infer type.

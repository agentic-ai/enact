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
"""Utility functions and classes."""
from typing import Any, Dict, Iterable, Optional

from enact import interfaces
from enact import resource_digests
from enact import resource_registry

# pylint: disable=invalid-name
class cached_property(property):
  """A property that caches its result based on the digest of the resource.

  NOTE: This is effectively an LRU cache with a maxsize of 1, so the cache
  is cleared when the digest of the resource changes.
  """
  def __init__(self, user_function):
    self._user_function = user_function
    self._cache: Dict[str, Any] = {}
    super().__init__()

  def __get__(self, instance: Any, unused_owner: Optional[type] = None):
    """Cached property based on the digest of the resource."""
    digest = resource_digests.resource_digest(instance)
    result = self._cache.get(digest, None)
    if result is None:
      result = self._user_function(instance)
      self._cache = {digest: result}
    return result


def walk_resource_dict(
    value: interfaces.ResourceDictValue,
    include_self: bool = True) -> Iterable[interfaces.ResourceDict]:
  """Recursively yields all ResourceDict occurrences in the dictionary tree.

  Args:
    value: The value to walk.
    include_self: Whether to also yield 'value' if it is a resource dict.

  Yields:
    All instances of ResourceDict in the subtree defined by 'value' in
    depth-first order.
  """
  if isinstance(value, interfaces.ResourceDict) and include_self:
    yield value
  if isinstance(value, dict):  # Both normal dicts and resource dicts.
    for v in value.values():
      yield from walk_resource_dict(v, include_self=True)
  elif isinstance(value, list):
    for v in value:
      yield from walk_resource_dict(v, include_self=True)


def walk_resource(
    value: Any,
    include_self: bool = True) -> Iterable[interfaces.ResourceBase]:
  """Recursively yields all sub-resources in depth-first order.

  Will auto-wrap non-primitive python values.

  Args:
    value: The value to walk.
    include_self: Whether to also yield 'value' if it is a resource dict.

  Yields:
    All instances of ResourceBase in the subtree defined by 'value' in
    depth-first order.
  """
  if isinstance(value, dict):
    for v in value.values():
      yield from walk_resource(v, include_self=True)
  elif isinstance(value, list):
    for v in value:
      yield from walk_resource(v, include_self=True)
  elif isinstance(value, interfaces.PRIMITIVES):
    pass
  elif isinstance(value, type):
    pass
  else:
    value = resource_registry.wrap(value)
    if include_self:
      yield value
    assert isinstance(value, interfaces.ResourceBase)
    for v in value.field_values():
      yield from walk_resource(v, include_self=True)

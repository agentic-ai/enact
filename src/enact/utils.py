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
from typing import Any, Dict, Optional

import enact

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
    digest = enact.resource_digest(instance)
    result = self._cache.get(digest, None)
    if result is None:
      result = self._user_function(instance)
      self._cache = {digest: result}
    return result

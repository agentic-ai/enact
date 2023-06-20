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

"""Type registration functionality to allow deserialization of resources."""

from typing import Dict, Optional, Type, TypeVar

from enact import interfaces


class RegistryError(Exception):
  """Raised when there is an error with the registry."""


class ResourceNotFound(RegistryError):
  """Raised when a resource is not found."""


class Registry:
  """Registers resource types for deserialization."""

  _singleton: Optional['Registry'] = None

  def __init__(self):
    """Initializes a registry."""
    self.allow_reregistration = False
    self._type_map: Dict[str, Type[interfaces.ResourceBase]] = {}

  def register(self, resource: Type[interfaces.ResourceBase]):
    """Registers the resource or reference type."""
    if not issubclass(resource, interfaces.ResourceBase):
      raise RegistryError(
        f'Cannot register non-resource type: {resource}')
    import traceback as tb
    type_id = resource.type_id()
    if (type_id in self._type_map and
        self._type_map[type_id] != resource and
        not self.allow_reregistration):
      raise RegistryError(
        f'{id(resource)} == {id(self._type_map[type_id])}\n'
        f'While registering {resource}: '
        f'Type with id {type_id} already '
        f'registered to a different type: '
        f'{self._type_map[type_id]}\n')
    self._type_map[type_id] = resource

  def lookup(self, type_id: str) -> Type[interfaces.ResourceBase]:
    """Looks up a resource type by name."""
    resource_class = self._type_map.get(type_id)
    if not resource_class:
      raise ResourceNotFound(f'No type registered for {type_id}')
    return resource_class

  @classmethod
  def get(cls) -> 'Registry':
    """Returns the singleton registry."""
    if not cls._singleton:
      cls._singleton = cls()
    return cls._singleton


R = TypeVar('R', bound=interfaces.ResourceBase)


def register(cls: Type[R]) -> Type[R]:
  """Decorator for resource classes."""
  Registry.get().register(cls)
  return cls

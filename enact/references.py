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

"""References and stores."""

import abc
import contextlib
import json
import os
from typing import Any, Dict, Generic, Iterable, Iterator, Mapping, NamedTuple, Optional, Type, TypeVar, cast

from enact import contexts
from enact import digests
from enact import interfaces
from enact import resource_registry
from enact import serialization


class RefError(Exception):
  """Superclass for reference related errors."""


class PackedResource(NamedTuple):
  """A resource packed for storage with a corresponding reference."""
  # A resource packed for storage. This may be just the resource itself or some
  # other representation that encodes it, e.g., compressed or encrypted.
  data: interfaces.ResourceDict
  # A reference to the packed resource. The reference class must be compatible
  # with the packed resource, i.e., ref.verify(resource) must succeed.
  ref: 'Ref'


P = TypeVar('P', bound='Ref')
R = TypeVar('R', bound=interfaces.ResourceBase)


def get(ref: Optional['Ref[R]']) -> R:
  """Gets the reference or asserts false if None."""
  assert ref
  return ref.get()


def commit(resource: R) -> 'Ref[R]':
  """Commits a resource to the store and returns a reference."""
  return Store.current().commit(resource)


@resource_registry.register
class Ref(Generic[R], interfaces.ResourceBase):
  """Represents a reference to other resources.

  References are JSON encodable objects and their key-sorted json encoding
  serves as a string key for the underlying resource.

  Ref subclasses may pack resources before storage and unpack them when they are
  retrieved. This allows implementation of subtypes of Ref that perform
  end-to-end encryption or compression.
  """

  def __init__(self, digest: str, resource: Optional[R]=None):
    """Initializes the reference from a digest and optionally the resource."""
    self._digest = digest
    self._cached: Optional[R] = resource

  @property
  def digest(self) -> str:
    """Returns a unique, deterministic digest of the referenced resource."""
    return self._digest

  @property
  def id(self) -> str:
    """Returns a string version of this reference."""
    return json.dumps(dict(self.field_items()), sort_keys=True)

  def __hash__(self) -> int:
    """Hash representation."""
    return hash(self.id)

  @classmethod
  def from_id(cls: Type[P], id: str) -> P:
    """Returns a string version of this reference."""
    return cls.from_resource_dict(
      interfaces.ResourceDict(cls, **json.loads(id)))

  @contextlib.contextmanager
  def modify(self) -> Iterator[R]:
    """Context manager for modifying the resource."""
    resource = self.get()
    yield resource
    self.set(resource)

  def get(self) -> R:
    """Fetches the resource from the cache or active store."""
    if self._cached is None or self.from_resource(self._cached) != self:
      self._cached = Store.current().get(self)
    return self._cached

  def __call__(self) -> R:
    """Alias for get."""
    return self.get()

  def set(self, resource: R):
    """Sets the reference to point to the given resource."""
    self._cached = resource
    self._digest = digests.digest(resource)

  @classmethod
  def from_resource(cls: Type[P], resource: interfaces.ResourceBase) -> P:
    """Constructs a reference from a resource."""
    return cls(digest=digests.digest(resource), resource=resource)

  def __eq__(self, other: Any):
    """Returns true if the referenced object is the same."""
    if not isinstance(other, Ref):
      return False
    if type(self) != type(other):
      other = self.from_resource(other.get())
    return self.digest == other.digest

  @classmethod
  def verify(cls, packed_resource: PackedResource):
    """Check the integrity of a packed resource.

    Args:
      packed_resource: The packed resource to verify.

    Raises:
      RefError: If integrity check fails.
    """
    if digests.digest(packed_resource.data) != packed_resource.ref.digest:
      raise RefError(
        f'Reference {packed_resource.ref} does not match packed data '
        f'{packed_resource.data}.')

  @classmethod
  def unpack(cls: Type['Ref[R]'],
             packed_resource: PackedResource) -> R:
    """Unpacks the referenced resource."""
    if not isinstance(packed_resource.ref, cls):
      raise RefError(
        f'Reference type mismatch: {packed_resource.ref} is not a {cls}.')
    cls.verify(packed_resource)
    return cast(
      R, packed_resource.data.to_resource())

  @classmethod
  def pack(cls, resource: R) -> PackedResource:
    """Packs the resource."""
    return PackedResource(
      resource.to_resource_dict(),
      ref=cls.from_resource(resource))

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    yield 'digest'

  def field_values(self) -> Iterable[interfaces.FieldValue]:
    """Return a list of field values, aligned with field_names."""
    yield self._digest

  @classmethod
  def from_fields(cls: Type[P],
                  field_values: Mapping[str, interfaces.FieldValue]) -> P:
    """Constructs the resource from a value dictionary."""
    return cls(**field_values)  # type: ignore

  def __repr__(self) -> str:
    return f'<{type(self).__name__}: {self.digest}>'


class StorageBackend(abc.ABC):
  """A storage backend."""

  @abc.abstractmethod
  def commit(self, packed_resource: PackedResource):
    """Stores a packed resource."""

  @abc.abstractmethod
  def has(self, ref: Ref) -> bool:
    """Returns whether the storage backend has the resource."""

  @abc.abstractmethod
  def get(self, ref: Ref) -> Optional[interfaces.ResourceDict]:
    """Returns the packed resource or None if not available."""


class NotFound(Exception):
  """Raised when a resource is not found."""


class InMemoryBackend(StorageBackend):
  """A backend that stores resources in memory."""

  def __init__(self):
    """Create a new in-memory backend."""
    self._resources: Dict[str, interfaces.ResourceDict] = {}

  def commit(self, packed_resource: PackedResource):
    """Stores a packed resource."""
    packed_resource.ref.verify(packed_resource)
    self._resources[packed_resource.ref.id] = packed_resource.data

  def has(self, ref: Ref) -> bool:
    """Returns whether the backend has the referenced resource."""
    return ref.id in self._resources

  def get(self, ref: Ref) -> Optional[interfaces.ResourceDict]:
    """Returns the packed resource or None if not available."""
    return self._resources.get(ref.id)

  def __len__(self) -> int:
    """Returns the number of resources in the backend."""
    return len(self._resources)


class FileBackend(StorageBackend):
  """A backend that stores resources in files."""

  def __init__(self,
               root_dir: str,
               serializer: Optional[serialization.Serializer] = None):
    """Create a new in-memory backend."""
    os.makedirs(root_dir, exist_ok=True)
    self._root_dir = root_dir
    self._serializer = serializer or serialization.JsonSerializer()

  def commit(self, packed_resource: PackedResource):
    """Stores a packed resource."""
    packed_resource.ref.verify(packed_resource)
    with open(os.path.join(self._root_dir, packed_resource.ref.id), 'wb') as f:
      f.write(self._serializer.serialize(
        packed_resource.data))

  def has(self, ref: Ref) -> bool:
    """Returns whether the backend has the referenced resource."""
    return os.path.exists(os.path.join(self._root_dir, ref.id))

  def get(self, ref: Ref) -> Optional[interfaces.ResourceDict]:
    """Returns the packed resource or None if not available."""
    with open(os.path.join(self._root_dir, ref.id), 'rb') as f:
      return self._serializer.deserialize(f.read())


@contexts.register
class Store(contexts.Context):
  """A store for resources."""

  def __init__(
      self,
      backend: Optional[StorageBackend]=None,
      registry: Optional[resource_registry.Registry]=None,
      ref_type: Type[Ref]=Ref):
    """Initializes the store."""
    self._backend = backend or InMemoryBackend()
    self._registry = registry
    self._ref_type = ref_type

  def commit(self, resource: R) -> Ref[R]:
    """Commits a resource to the store."""
    packed_resource = self._ref_type.pack(resource)
    self._backend.commit(packed_resource)
    return packed_resource.ref

  def has(self, ref: Ref) -> bool:
    """Returns whether the store has a resource."""
    return self._backend.has(ref)

  def get(self, ref: Ref[R]) -> R:
    """Retrieves a resource from the store."""
    resource_data = self._backend.get(ref)
    if resource_data is None:
      raise NotFound(ref.id)
    packed_resource = PackedResource(resource_data, ref=ref)
    return ref.unpack(packed_resource)

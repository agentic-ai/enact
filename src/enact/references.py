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
import asyncio
import base64
import contextlib
import json
import os
import pickle

from typing import (
  Any, Awaitable, Dict, Generic, Iterable, Iterator, List, Mapping, NamedTuple,
  Optional, Set, Tuple, Type, TypeVar, Union, cast)

from enact import contexts
from enact import digests
from enact import interfaces
from enact import resource_registry
from enact import serialization
from enact import types
from enact import utils



class RefError(Exception):
  """Superclass for reference related errors."""


R = TypeVar('R')
ResourceT = TypeVar('ResourceT', bound=interfaces.ResourceBase)
RefT = TypeVar('RefT', bound='Ref')


class PackedResource(NamedTuple):
  """A resource packed for storage with a corresponding reference."""
  # A resource packed for storage. This may be just the resource itself or some
  # other representation that encodes it, e.g., compressed or encrypted.
  data: interfaces.ResourceDict
  # The reference to this object as a resource dict.
  ref_dict: interfaces.ResourceDict
  # The outgoing reference ids as a set of strings.
  links: Set[str]
  # The referenced type keys.
  type_keys: Set[types.TypeKey]

  def unpack(self) -> Any:
    """Unpacks the resource."""
    return self.ref().unpack(self)

  def ref(self) -> 'Ref':
    """Returns the reference to this resource."""
    r = resource_registry.from_resource_dict(self.ref_dict)
    if not isinstance(r, Ref):
      raise RefError(f'Expected a Ref, got {r}.')
    return r


def checkout(ref: 'Ref[R]') -> R:
  """Gets the reference or asserts false if None."""
  return ref.checkout()


async def checkout_async(ref: 'Ref[R]') -> R:
  """Gets the reference or asserts false if None."""
  return await ref.checkout_async()


def commit(resource: R) -> 'Ref[R]':
  """Commits a value to the store and returns a reference."""
  return Store.current().commit(resource)


async def commit_async(resource: R) -> 'Ref[R]':
  """Commits a value to the store and returns a reference."""
  return await Store.current().commit_async(resource)


class _PackHelper:
  """Collects references and type keys while walking a resource."""

  def __init__(self):
    self.links: Set[str] = set()
    self.type_keys: Set[types.TypeKey] = set()

  def __call__(self, value: interfaces.FieldValue):
    """Collects references and type keys."""
    if isinstance(value, Ref):
      self.links.add(value.id)
    elif isinstance(value, interfaces.ResourceBase):
      self.type_keys.add(value.type_key())
    elif isinstance(value, type) and issubclass(value, interfaces.ResourceBase):
      self.type_keys.add(value.type_key())


@resource_registry.register
class Ref(Generic[R], interfaces.ResourceBase):
  """Represents a reference to a resource or wrappable python object.

  References are JSON encodable objects and their key-sorted json encoding
  serves as a string key for the underlying resource.

  Ref subclasses may pack resources before storage and unpack them when they are
  retrieved. This allows implementation of subtypes of Ref that perform
  end-to-end encryption or compression.
  """

  def __init__(self, digest: str):
    """Initializes the reference from a digest and optionally the resource."""
    assert isinstance(digest, str), (
      'Must instantiate Ref with a string digest.')
    self._digest = digest
    self._cached: List[R] = []

  def _clear_cache(self):
    """Clear the cache."""
    self._cached = []

  def _set_cache(self, value: R):
    """Set the cache."""
    self._cached = [resource_registry.unwrap(value)]

  @property
  def digest(self) -> str:
    """Returns a unique, deterministic digest of the referenced resource."""
    return self._digest

  @property
  def id(self) -> str:  # pylint: disable=invalid-name
    """Returns a string version of this reference."""
    return json.dumps(dict(self.field_items()), sort_keys=True)

  def __hash__(self) -> int:
    """Hash representation."""
    return hash(self.id)

  @classmethod
  def from_id(cls: Type[RefT], ref_id: str) -> RefT:
    """Returns a reference from a reference ID."""
    try:
      return cast(RefT, resource_registry.from_resource_dict(
        interfaces.ResourceDict(cls, **json.loads(ref_id))))
    except json.JSONDecodeError as error:
      raise RefError(f'Invalid ref id: {ref_id}') from error

  @contextlib.contextmanager
  def modify(self) -> Iterator[R]:
    """Context manager for modifying the resource."""
    resource = self.checkout()
    yield resource
    commit(resource)
    self.set(resource)

  def is_cached(self) -> bool:
    """Check whether the reference is cached."""
    return (
      bool(self._cached) and
      self.from_resource(resource_registry.wrap(self._cached[0])) == self)

  def checkout(self) -> R:
    """Fetches the resource from the cache or active store."""
    if not self.is_cached():
      self._set_cache(Store.current().checkout(self))
    return cast(R, self._cached[0])

  async def checkout_async(self) -> R:
    """Fetches the resource from the cache or active store."""
    if not self.is_cached():
      self._set_cache(await Store.current().checkout_async(self))
    return cast(R, self._cached[0])

  def __call__(self) -> R:
    """Alias for get."""
    return self.checkout()

  def set(self, resource: R):
    """Sets the reference to point to the given resource."""
    self._digest = digests.digest(resource_registry.wrap(resource))
    self._set_cache(resource)

  @classmethod
  def from_resource(cls: Type[RefT], resource: interfaces.ResourceBase) -> RefT:
    """Constructs a reference to the resource."""
    ref = cls(digest=digests.digest(resource))
    ref._set_cache(resource_registry.unwrap(resource))
    return ref

  @classmethod
  def from_resource_dict(
      cls: Type[RefT], resource_dict: interfaces.ResourceDict) -> RefT:
    """Constructs a reference to the resource represented by resource_dict."""
    return cls(digest=digests.digest(resource_dict))

  def __eq__(self, other: Any):
    """Returns true if the other object is the same reference."""
    if not isinstance(other, Ref):  # pylint: disable=unidiomatic-typecheck
      return False
    return self.digest == other.digest

  def verify(self, resource_dict: interfaces.ResourceDict):
    """Check that this reference matches the resource dict.

    Args:
      resource_dict: The resource dict to verify.

    Raises:
      RefError: If integrity check fails.
    """
    if self != self.from_resource_dict(resource_dict):
      raise RefError(
        f'Reference {self} does not match resource dict {resource_dict}.')

  def unpack(self: 'Ref[R]', packed_resource: PackedResource) -> R:
    """Unpacks the referenced resource."""
    self.verify(packed_resource.data)
    return cast(
      R,
      resource_registry.unwrap(
        resource_registry.from_resource_dict(packed_resource.data)))

  @classmethod
  def pack(cls: Type[RefT], resource: ResourceT) -> Tuple[RefT, PackedResource]:
    """Wraps and packs the resource."""
    callback = _PackHelper()
    resource_dict = resource.to_resource_dict(callback)
    ref = cls.from_resource(resource)
    return ref, PackedResource(
      data=resource_dict,
      ref_dict=ref.to_resource_dict(),
      links=callback.links,
      type_keys=callback.type_keys)

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    yield 'digest'

  def field_values(self) -> Iterable[interfaces.FieldValue]:
    """Return a list of field values, aligned with field_names."""
    yield self._digest

  @classmethod
  def from_fields(cls: Type[RefT],
                  field_dict: Mapping[str, interfaces.FieldValue]) -> RefT:
    """Constructs the resource from a value dictionary."""
    return cls(**field_dict)  # type: ignore

  def __repr__(self) -> str:
    return f'<{type(self).__name__}: {self.digest}>'

  def set_from(self, other: Any):
    """Sets the fields of this resource from another resource."""
    if type(other) != type(self):  # pylint: disable=unidiomatic-typecheck
      raise ValueError(f'Cannot set {self} from {other}: types do not match.')
    self._digest = other._digest  # pylint: disable=protected-access
    self._cached = list(other._cached)  # pylint: disable=protected-access


class StorageBackend(abc.ABC):
  """A storage backend."""

  @abc.abstractmethod
  def register_type(self,
                    type_key: types.TypeKey,
                    attributes: Dict[str, Optional[types.TypeDescriptor]]):
    """Register a new type to allow resource to be committed."""

  async def register_type_async(
      self,
      type_key: types.TypeKey,
      attributes: Dict[str, Optional[types.TypeDescriptor]]):
    """Register a new type to allow resource to be committed."""
    return self.register_type(type_key, attributes)

  @abc.abstractmethod
  def get_type(self, type_key: types.TypeKey) -> (
      Optional[Dict[str, Optional[types.TypeDescriptor]]]):
    """Returns the type, if known."""

  async def get_type_async(self, type_key: types.TypeKey) -> (
      Optional[Dict[str, Optional[types.TypeDescriptor]]]):
    """Returns the type, if known."""
    return self.get_type(type_key)

  @abc.abstractmethod
  def commit(self, ref_id: str, packed_resource: PackedResource):
    """Stores a packed resource."""

  async def commit_async(self, ref_id: str, packed_resource: PackedResource):
    """Stores a packed resource."""
    self.commit(ref_id, packed_resource)

  @abc.abstractmethod
  def has(self, ref_ids: Iterable[str]) -> List[bool]:
    """Returns whether the storage backend has the resource."""

  async def has_async(self, ref_ids: Iterable[str]) -> List[bool]:
    """Returns whether the storage backend has the resource."""
    return self.has(ref_ids)

  @abc.abstractmethod
  def checkout(
      self, ref_ids: Iterable[str]) -> (
        List[Optional[PackedResource]]):
    """Returns a dictionary of resources or None if not available.

    Args:
      ref_ids: The reference IDs to retrieve.

    Returns:
      A list of packed resources or None if the resource is not available.
      Returned in the order of the ref_ids argument.
    """

  async def checkout_async(
      self, ref_ids: Iterable[str]) -> (
        List[Optional[PackedResource]]):
    """Returns a dictionary of resources or None if not available.

    Args:
      ref_ids: The reference IDs to retrieve.

    Returns:
      A list of packed resources or None if the resource is not available.
      Returned in the order of the ref_ids argument.
    """
    return self.checkout(ref_ids)

  def get_type_keys(
      self, ref_ids: Iterable[str]) -> List[
        Optional[Set[types.TypeKey]]]:
    """Returns the types required to unpack the resources.

    The default implementation will load all resource data and extract only
    the type info. This should be overridden if more efficiency is required.

    Args:
      ref_ids: The reference IDs to retrieve.

    Returns:
      A list of sets of types or None, in the order of the ref_ids argument.
      None is returned if a reference cannot be resolved.
    """
    result: List[Optional[Set[types.TypeKey]]] = []
    for packed in self.checkout(ref_ids):
      if packed:
        # Add resource types.
        type_set = {
          value.type_info for value in utils.walk_resource_dict(packed.data)}
        # Add reference type.
        type_set.add(packed.ref_dict.type_info)
        result.append(type_set)
      else:
        result.append(None)
    return result

  async def get_type_keys_async(
      self, ref_ids: Iterable[str]) -> List[
        Optional[Set[types.TypeKey]]]:
    """Returns the types required to unpack the resources.

    The default implementation will load all resource data and extract only
    the type info. This should be overridden if more efficiency is required.

    Args:
      ref_ids: The reference IDs to retrieve.

    Returns:
      A list of sets of types or None, in the order of the ref_ids argument.
      None is returned if a reference cannot be resolved.
    """
    return self.get_type_keys(ref_ids)

  def get_dependency_graph(
      self,
      ref_ids: Iterable[str],
      max_depth: Optional[int]=None) -> Dict[str, Optional[Set[str]]]:
    """Return the dependency graph for the input references.

    The default implementation will load all resource data and extract only
    the reference graph. This should be overridden if more efficiency is
    required.

    Args:
      ref_dict: The reference to retrieve dependencies for, encoded as a
        resource_dict.
      max_depth: The maximum depth to retrieve dependencies. If None, all
        dependencies will be retrieved.

    Returns:
      A dictionary from reference IDs to sets of references that the key
      depends on directly. An unknown reference will map to None. The set of
      keys in the returned dictionary will contain the IDs of the references
      in ref_dicts and all references that they depend on up to the specified
      depth.
    """
    result: Dict[str, Optional[Set[str]]] = {}

    # Set up BFS 'queue', which is going to be batch processed.
    seen: Set[str] = set(ref_ids)
    this_level = set(seen)
    depth = 0

    while this_level and (max_depth is None or depth <= max_depth):
      # Batch fetch all unfetched references at this depth.
      packed_resources = self.checkout(this_level)
      next_level: Set[str] = set()

      for ref_id, packed in zip(this_level, packed_resources):
        if packed is None:
          result[ref_id] = None
        else:
          result[ref_id] = packed.links
          next_level.update(packed.links - seen)
          seen.update(packed.links)

      # Update loop variables
      depth += 1
      this_level = next_level
    return result

  async def get_dependency_graph_async(
      self,
      ref_ids: Iterable[str],
      max_depth: Optional[int]=None) -> Dict[str, Optional[Set[str]]]:
    """Return the dependency graph for the input references.

    The default implementation will load all resource data and extract only
    the reference graph. This should be overridden if more efficiency is
    required.

    Args:
      ref_dict: The reference to retrieve dependencies for, encoded as a
        resource_dict.
      max_depth: The maximum depth to retrieve dependencies. If None, all
        dependencies will be retrieved.

    Returns:
      A dictionary from reference IDs to sets of references that the key
      depends on directly. An unknown reference will map to None. The set of
      keys in the returned dictionary will contain the IDs of the references
      in ref_dicts and all references that they depend on up to the specified
      depth.
    """
    return self.get_dependency_graph(ref_ids, max_depth)


class NotFound(Exception):
  """Raised when a resource is not found."""


class InMemoryBackend(StorageBackend):
  """A backend that stores resources in memory."""

  def __init__(self):
    """Create a new in-memory backend."""
    self._resources: Dict[str, PackedResource] = {}
    self._types: Dict[
      types.TypeKey, Dict[str, Optional[types.TypeDescriptor]]] = {}

  def register_type(self,
                    type_key: types.TypeKey,
                    attributes: Dict[str, Optional[types.TypeDescriptor]]):
    """Register a new type to allow resource to be committed."""
    if type_key in self._types and self._types[type_key] != attributes:
      raise TypeKeyError(
        f'Type {type_key} already registered with different attributes.')
    self._types[type_key] = attributes

  def get_type(self, type_key: types.TypeKey) -> (
      Optional[Dict[str, Optional[types.TypeDescriptor]]]):
    """Returns the type, if known."""
    if type_key not in self._types:
      return None
    return self._types[type_key]

  def commit(self, ref_id: str, packed_resource: PackedResource):
    """Stores a packed resource."""
    self._resources[ref_id] = packed_resource

  def has(self, ref_ids: Iterable[str]) -> List[bool]:
    """Returns whether the backend has the referenced resource."""
    return [ref_id in  self._resources for ref_id in ref_ids]

  def checkout(self, ref_ids: Iterable[str]) -> (
      List[Optional[PackedResource]]):
    """Returns a dictionary with resource data or None if not available."""
    return [self._resources.get(ref_id) for ref_id in ref_ids]

  def __len__(self) -> int:
    """Returns the number of resources in the backend."""
    return len(self._resources)


class FileBackend(StorageBackend):
  """A backend that stores resources in files."""

  def __init__(self,
               root_dir: str,
               serializer: Optional[serialization.Serializer] = None,
               use_base64_names: bool=True):
    """Create a new file-backed backend.

    Args:
      root_dir: The directory where resources will be stored.
      serialized: The serializer to use. Will default to JsonSerializer if not
        provided.
      use_base64_names: Use base64 encoded filenames for resources. This is
        useful on windows, since windows does not allow certain characters in
        file names.
    """
    os.makedirs(root_dir, exist_ok=True)
    self._root_dir = root_dir
    self._serializer = serializer or serialization.JsonSerializer()
    self._use_base64_names = use_base64_names

  def register_type(self,
                    type_key: types.TypeKey,
                    attributes: Dict[str, Optional[types.TypeDescriptor]]):
    """Register a new type to allow resource to be committed."""
    with open(self._get_type_path(type_key), 'wb') as file:
      encoded_attrs = {
        key: value.to_json() if value else None
        for key, value in attributes.items()}
      pickle.dump(encoded_attrs, file)

  def get_type(self, type_key: types.TypeKey) -> (
      Optional[Dict[str, Optional[types.TypeDescriptor]]]):
    """Returns the type, if known."""
    if not os.path.exists(self._get_type_path(type_key)):
      return None
    with open(self._get_type_path(type_key), 'rb') as f:
      encoded_attrs = pickle.load(f)
    assert isinstance(encoded_attrs, dict)
    return {
      key: types.TypeDescriptor.from_json(value)
      for key, value in encoded_attrs.items()}

  def _get_type_path(self, type_key: types.TypeKey) -> str:
    type_id = json.dumps(type_key.as_dict()).encode('utf-8')
    basename = f'type_{base64.b64encode(type_id).decode("utf-8")}'
    return os.path.join(self._root_dir, basename)

  def _get_path(self, ref_id: str) -> str:
    basename = ref_id
    if self._use_base64_names:
      basename = base64.b64encode(basename.encode('utf-8')).decode('utf-8')
    return os.path.join(self._root_dir, basename)

  def commit(self, ref_id: str, packed_resource: PackedResource):
    """Stores a packed resource."""
    data_bytes = self._serializer.serialize(packed_resource.data)
    ref_bytes = self._serializer.serialize(packed_resource.ref_dict)
    links = packed_resource.links
    with open(self._get_path(ref_id), 'wb') as file:
      pickle.dump((data_bytes, ref_bytes,
                   links, packed_resource.type_keys), file)

  def has(self, ref_ids: Iterable[str]) -> List[bool]:
    """Returns whether the backend has the referenced resource."""
    return [os.path.exists(self._get_path(ref_id)) for ref_id in ref_ids]

  def checkout(self, ref_ids: Iterable[str]) -> (
      List[Optional[PackedResource]]):
    """Returns a dictionary with resource data or None if not available."""
    return [self._get_packed(ref_id) for ref_id in ref_ids]

  def _get_packed(self, ref_id: str) -> Optional[PackedResource]:
    """Return the packed resource for a reference."""
    path = self._get_path(ref_id)
    if not os.path.exists(path):
      return None
    with open(path, 'rb') as file:
      data_bytes, ref_bytes, links, type_keys = pickle.load(file)
    data: interfaces.ResourceDict = self._serializer.deserialize(data_bytes)
    ref_dict: interfaces.ResourceDict = self._serializer.deserialize(ref_bytes)
    return PackedResource(data, ref_dict, links, type_keys)


class DistributionKeyError(Exception):
  """Raised when there are issues with distribution key objects.."""


class TypeKeyError(Exception):
  """Raised when there is an issue with type key objects."""


@contexts.register
class Store(contexts.Context):
  """A store for resources."""

  def __init__(
      self,
      backend: Optional[StorageBackend]=None,
      registry: Optional[resource_registry.Registry]=None,
      ref_type: Type[Ref]=Ref):
    """Initializes the store."""
    super().__init__()
    self._backend = backend if backend is not None else InMemoryBackend()
    self._registry = registry
    self._ref_type = ref_type
    # Tracks types known to exist on the backend.
    self._types_in_backend: Set[types.TypeKey] = set()

  def _register_type_helper(self, value: Any) -> (
    Tuple[types.TypeKey, Dict[str, Optional[types.TypeDescriptor]]]):
    """Helper for register type sync / async implementation."""
    as_resource: Union[Type[interfaces.ResourceBase], interfaces.ResourceBase]
    if isinstance(value, type):
      as_resource = resource_registry.wrap_type(value)
    else:
      as_resource = resource_registry.wrap(value)
    attributes = dict(
      zip(as_resource.field_names(), as_resource.field_descriptors()))
    return as_resource.type_key(), attributes

  def register_type(self, resource: Union[Type[R], R]):
    """Explicitly register a type with the store."""
    type_key, attributes = self._register_type_helper(resource)
    self._backend.register_type(type_key, attributes)

  async def register_type_async(self, resource: Union[Type[R], R]):
    """Explicitly register a type with the store."""
    type_key, attributes = self._register_type_helper(resource)
    await self._backend.register_type_async(type_key, attributes)

  def commit(self, resource: R) -> Ref[R]:
    """Commits a resource to the store."""
    as_resource = resource_registry.wrap(resource)
    self.register_type(as_resource)
    ref, packed_resource = self._ref_type.pack(as_resource)
    new_types = packed_resource.type_keys - self._types_in_backend
    registry = resource_registry.Registry.get()
    for type_key in new_types:
      _, attributes = self._register_type_helper(
        registry.lookup(type_key))
      self._backend.register_type(type_key, attributes)
      self._types_in_backend.add(type_key)
    self._backend.commit(ref.id, packed_resource)
    return ref

  async def commit_async(self, resource: R) -> Ref[R]:
    """Commits a resource to the store."""
    as_resource = resource_registry.wrap(resource)
    await self.register_type_async(as_resource)
    ref, packed_resource = self._ref_type.pack(as_resource)
    new_types = packed_resource.type_keys - self._types_in_backend
    register_coros: List[Awaitable] = []
    registry = resource_registry.Registry.get()
    for type_key in new_types:
      _, attributes = self._register_type_helper(
        registry.lookup(type_key))
      coro = self._backend.register_type_async(type_key, attributes)
      register_coros.append(coro)
    await asyncio.gather(*register_coros)
    self._types_in_backend.update(new_types)
    await self._backend.commit_async(ref.id, packed_resource)
    return ref

  def has(self, ref: Ref) -> bool:
    """Returns whether the store has a resource."""
    return self._backend.has((ref.id,))[0]

  async def has_async(self, ref: Ref) -> bool:
    """Returns whether the store has a resource."""
    return (await self._backend.has_async((ref.id,)))[0]

  def _checkout_verify_packed(
    self, ref: Ref[R], packed_resource: Optional[PackedResource]) -> R:
    """Verify and return the packed object.."""
    if packed_resource is None:
      raise NotFound(ref.id)
    if packed_resource.ref() != ref:
      raise RefError(
        f'Backend returned wrong reference:\ngot {packed_resource.ref()}\n'
        f'expected: {ref}')
    result = ref.unpack(packed_resource)
    self._types_in_backend.update(packed_resource.type_keys)
    return result

  def checkout(self, ref: Ref[R]) -> R:
    """Retrieves a resource from the store."""
    packed_resource = self._backend.checkout((ref.id,))[0]
    return self._checkout_verify_packed(ref, packed_resource)

  async def checkout_async(self, ref: Ref[R]) -> R:
    """Retrieves a resource from the store."""
    packed_resource = (await self._backend.checkout_async((ref.id,)))[0]
    return self._checkout_verify_packed(ref, packed_resource)

  def _get_transitive_ref_ids(
    self, ref: Ref, graph: Dict[str, Optional[Set[str]]]) -> (
      Set[str]):
    """Return a set of transitive ref IDs from a dependency graph."""
    all_references = {ref.id}
    for ref_id, deps in graph.items():
      if deps is None:
        raise NotFound(f'Could not resolve transitive reference {ref_id}.')
      all_references.update(deps)
    return all_references

  def _get_transitive_type_requirements(
    self,
    ref_and_typesets: Iterable[Tuple[str, Optional[Set[types.TypeKey]]]]) -> (
      Set[types.TypeKey]):
    """Return the set of type keys.."""
    result: Set[types.TypeKey] = set()
    for typed_ref, type_set in ref_and_typesets:
      if type_set is None:
        raise TypeKeyError(
          f'Could not resolve types for transitive reference {typed_ref}.')
      result.update(type_set)
    return result

  def get_transitive_type_requirements(self, ref: Ref) -> (
      Set[types.TypeKey]):
    """Return a set of transitive type requirements for the reference."""
    all_references = self._get_transitive_ref_ids(
      ref, self._backend.get_dependency_graph([ref.id]))
    type_sets = self._backend.get_type_keys(all_references)
    return self._get_transitive_type_requirements(
      zip(all_references, type_sets))

  async def get_transitive_type_requirements_async(self, ref: Ref) -> (
      Set[types.TypeKey]):
    """Return a set of transitive type requirements for the reference."""
    all_references = self._get_transitive_ref_ids(
      ref, await self._backend.get_dependency_graph_async([ref.id]))
    type_sets = await self._backend.get_type_keys_async(all_references)
    return self._get_transitive_type_requirements(
      zip(all_references, type_sets))

  def _get_distribution_requirements(
      self,
      type_requirements: Set[types.TypeKey],
      expect_distribution_key: bool) -> (
        Set[types.DistributionKey]):
    """Return distribution requirements from a set of type keys."""
    result: Set[types.DistributionKey] = set()
    for type_info in type_requirements:
      if type_info.distribution_key is None:
        if expect_distribution_key:
          raise DistributionKeyError(
            f'Type {type_info} does not have distribution key.')
      else:
        result.add(type_info.distribution_key)
    return result

  def get_distribution_requirements(
    self, ref: Ref, expect_distribution_key: bool=True) -> (
      Set[types.DistributionKey]):
    """Return the distribution requirements of a reference.

    Args:
      ref: The reference to check.
      expect_distribution_key: If True, raise an error if the reference does
        not have a distribution key.
    """
    return self._get_distribution_requirements(
      self.get_transitive_type_requirements(ref), expect_distribution_key)

  async def get_distribution_requirements_async(
    self, ref: Ref, expect_distribution_key: bool=True) -> (
      Set[types.DistributionKey]):
    """Return the distribution requirements of a reference.

    Args:
      ref: The reference to check.
      expect_distribution_key: If True, raise an error if the reference does
        not have a distribution key.
    """
    return self._get_distribution_requirements(
      await self.get_transitive_type_requirements_async(ref),
      expect_distribution_key)

  def _get_dependency_graph(self, graph: Dict[str, Optional[Set[str]]]) -> (
      Dict[Ref, Optional[Set[Ref]]]):
    """Translate a backend dependency graph to a reference dependency graph."""
    return {
      Ref.from_id(ref):
        {Ref.from_id(dep) for dep in deps} if deps is not None else None
      for ref, deps in graph.items()}

  def get_dependency_graph(
      self,
      refs: Iterable[Ref],
      max_depth: Optional[int]=None) -> Dict[Ref, Optional[Set[Ref]]]:
    """Return a dependency graph over references."""
    return self._get_dependency_graph(
      self._backend.get_dependency_graph([ref.id for ref in refs], max_depth))

  async def get_dependency_graph_async(
      self,
      refs: Iterable[Ref],
      max_depth: Optional[int]=None) -> Dict[Ref, Optional[Set[Ref]]]:
    """Return a dependency graph over references."""
    return self._get_dependency_graph(
      await self._backend.get_dependency_graph_async(
        [ref.id for ref in refs], max_depth))


@contexts.register_to_superclass(Store)
class FileStore(Store):
  """A file-based store."""

  def __init__(self, root_dir: str):
    """Initializes the store."""
    super().__init__(backend=FileBackend(root_dir))


@contexts.register_to_superclass(Store)
class InMemoryStore(Store):
  """An in-memory store."""

  def __init__(self):
    """Initializes the store."""
    super().__init__(backend=InMemoryBackend())

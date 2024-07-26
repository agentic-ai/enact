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

"""Tests for references and stores."""

import dataclasses
import tempfile
from typing import Awaitable, Callable, List, TypeVar
import unittest

import enact
from enact import interfaces
from enact import references
from enact import serialization
from enact import contexts
from enact import resource_registry
from enact import acyclic
from enact import type_wrappers
from enact import types
from enact import version

# pylint: disable=invalid-name,missing-class-docstring

@enact.register
@dataclasses.dataclass
class SimpleResource(enact.Resource):
  x: int
  y: float


@enact.register
@dataclasses.dataclass
class JsonPackedResource(enact.Resource):
  contents: bytes


@enact.register
class JsonPackedRef(enact.Ref):
  """A reference to a JSON packed resource."""

  @classmethod
  def verify(cls, packed_resource: references.PackedResource):
    """Verifies that the packed resource is valid."""

  @classmethod
  def unpack(cls, packed_resource: references.PackedResource) -> (
      interfaces.ResourceBase):
    """Unpacks the referenced resource."""
    data = packed_resource.data
    if data.type_info != JsonPackedResource.type_key():
      raise enact.RefError('Resource is not a JsonPackedResource.')
    json_packed: JsonPackedResource = resource_registry.from_resource_dict(data)
    unpacked_dict = serialization.JsonSerializer().deserialize(
      json_packed.contents)
    return resource_registry.from_resource_dict(unpacked_dict)

  @classmethod
  def pack(cls, resource: interfaces.ResourceBase) -> references.PackedResource:
    """Packs the resource."""
    ref = cls.from_resource(resource)
    return ref, references.PackedResource(
      JsonPackedResource(serialization.JsonSerializer().serialize(
        resource.to_resource_dict())).to_resource_dict(),
      ref_dict=ref.to_resource_dict(),
      links=set(),
      type_keys=set())


class RefTest(unittest.TestCase):
  """A test for refs."""

  def test_verify_pack_unpack(self):
    """Test that packing and unpacking works."""
    resource = SimpleResource(x=1, y=2.0)
    ref, packed = enact.Ref.pack(resource)
    packed.ref().verify(packed.data)
    self.assertEqual(ref, packed.ref())
    unpacked = packed.ref().unpack(packed)
    self.assertEqual(unpacked, resource)

  def test_custom_ref_type(self):
    """Tests custom ref types works."""
    resource = SimpleResource(x=1, y=2.0)
    ref, packed = JsonPackedRef.pack(resource)
    self.assertEqual(ref, packed.ref())
    unpacked = packed.ref().unpack(packed)
    packed.ref().verify(packed.data)
    self.assertEqual(unpacked, resource)

  def test_id_from_id(self):
    """Test that references can be cast to and from ids."""
    ref: enact.Ref = enact.Ref('fake_digest')
    ref2 = ref.from_id(ref.id)
    self.assertEqual(ref, ref2)

  def test_ref_non_string_digest(self):
    """Test that references cannot be constructed from a non-string digest."""
    with self.assertRaisesRegex(AssertionError, 'string digest'):
      _ = enact.Ref(1234)  # type: ignore

  def test_ref_to_wrapped_type(self):
    """Tests references to wrapped python types."""
    value = 1
    ref, packed = enact.Ref.pack(resource_registry.wrap(value))
    self.assertIsInstance(
      resource_registry.from_resource_dict(packed.data),
      resource_registry.IntWrapper)
    self.assertIsInstance(ref, enact.Ref)
    self.assertEqual(ref, packed.ref())
    packed.ref().verify(packed.data)
    self.assertEqual(packed.ref().unpack(packed), 1)

T = TypeVar('T')

class StoreTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    """Initializes the test."""
    self._async = False

  def _as_async(self, fun: Callable[..., T]) -> (
      Callable[..., Awaitable[T]]):
    """Returns either the sync or corresponding async function."""
    assert getattr(fun, '__self__') is not None, (
      'Callable must be a bound instance method.')
    if not self._async:
      async def wrapper(*args, **kwargs):
        return fun(*args, **kwargs)
      return wrapper
    return getattr(fun.__self__, fun.__name__ + '_async')

  async def test_commit_has_get(self):
    """Create a store and test that commit, has, and cehckout work."""
    for async_ in (False, True):
      self._async = async_
      store = enact.Store()
      resource = SimpleResource(x=1, y=2.0)
      ref = await self._as_async(store.commit)(resource)
      self.assertEqual(await self._as_async(store.checkout)(ref), resource)
      self.assertNotEqual(id(await self._as_async(store.checkout)(ref)),
                          id(resource))

  async def test_commit_stores_types(self):
    """Tests that types are stored in he backend."""
    for async_ in (False, True):
      self._async = async_
      store = enact.Store()
      self.assertIsNone(
        # pylint: disable=protected-access
        await self._as_async(store._backend.get_type)(
          SimpleResource.type_key()))
      @enact.register
      @dataclasses.dataclass
      class LocalResource(enact.Resource):
        z: int
      with store:
        enact.commit(SimpleResource(x=[LocalResource(1)], y=0.0))

      # pylint: disable=protected-access
      simple_type = await self._as_async(store._backend.get_type)(
          SimpleResource.type_key())
      assert simple_type is not None
      self.assertCountEqual(['x', 'y'], simple_type)

      # pylint: disable=protected-access
      local_type = await self._as_async(store._backend.get_type)(
        LocalResource.type_key())
      assert local_type is not None
      self.assertCountEqual(['z'], local_type)


  def test_store_provided_backend(self):
    """Tests that constructing stores with a provided backend works."""
    b1 = enact.InMemoryBackend()
    self.assertEqual(
      enact.Store(b1)._backend,  # pylint: disable=protected-access
      b1)

    with tempfile.TemporaryDirectory() as tmpdir:
      b2 = enact.FileBackend(tmpdir)
      self.assertEqual(
        enact.Store(b2)._backend,  # pylint: disable=protected-access
        b2)

  def test_custom_ref(self):
    """Test stores with custom ref types."""
    store = enact.Store(ref_type=JsonPackedRef)
    resource = SimpleResource(x=1, y=2.0)
    ref = store.commit(resource)
    self.assertIsInstance(ref, JsonPackedRef)
    self.assertTrue(store.has(ref))
    self.assertEqual(store.checkout(ref), resource)
    self.assertNotEqual(id(store.checkout(ref)), id(resource))

  async def test_wrapped_ref(self):
    store = enact.Store()
    for val in [None, 0, 0.0, 'str', True, bytes([1, 2, 3]),
                [1, 2, 3], {'a': 1}]:
      with self.subTest(val):
        ref = await self._as_async(store.commit)(val)
        self.assertEqual(await self._as_async(store.checkout)(ref), val)

  async def test_caching_none(self):
    """Tests that caching None works correctly."""
    store = enact.Store()
    ref = await self._as_async(store.commit)(None)
    self.assertTrue(ref.is_cached())
    self.assertIsNone(ref.checkout())

  async def test_caching(self):
    """Test that caching works correctly."""
    store = enact.Store()
    resource = SimpleResource(x=1, y=2.0)
    ref = await self._as_async(store.commit)(resource)

    # checkout() works because cached reference is correct.
    self.assertTrue(ref.is_cached())
    (await self._as_async(ref.checkout)()).x = 10
    self.assertFalse(ref.is_cached())

    # checkout() fails because cached reference is incorrect.
    with self.assertRaises(contexts.NoActiveContext):
      await self._as_async(ref.checkout())()

    with store:
      # Refetches the correct resource from the store.
      self.assertEqual((await self._as_async(ref.checkout)()).x, 1)

  def test_modify(self):
    """Tests the modify context."""
    store = enact.Store()
    resource = SimpleResource(x=1, y=2.0)
    ref = store.commit(resource)
    old_digest = ref.digest

    with store:
      with ref.modify() as resource:
        resource.x = 10
      self.assertEqual(ref.checkout().x, 10)
    self.assertNotEqual(ref.digest, old_digest)

  def test_modify_wrapped(self):
    """Tests that modifying wrapped resources works."""
    store = enact.Store()
    with store:
      ref = store.commit([0, 1, 2])
      old_ref = enact.deepcopy(ref)
      with ref.modify() as elems:
        elems.append(3)
      self.assertEqual(ref.checkout(), [0, 1, 2, 3])
      self.assertNotEqual(ref, old_ref)

  def test_pack_none(self):
    """Tests packing none."""
    store = enact.Store()
    value = None
    ref = store.commit(value)
    _, packed = ref.pack(resource_registry.wrap(value))
    ref.verify(packed.data)

  def test_packed_links(self):
    """Tests links are properly tracked during packing."""
    store = enact.Store()
    with store:
      x = enact.commit(1)
      y = enact.commit(2.0)
      r1 = SimpleResource(x, [{'test': (y, y)}])  # type: ignore
      _, packed = enact.Ref.pack(r1)
      got = packed.links
      want = {x.id, y.id}
      self.assertEqual(got, want)

  async def test_file_backend(self):
    """Tests the file backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
      for async_ in (False, True):
        self._async = async_
        store = enact.Store(backend=enact.FileBackend(tmpdir))
        resource = SimpleResource(x=1, y=2.0)
        ref = await self._as_async(store.commit)(resource)
        self.assertTrue(await self._as_async(store.has)(ref))
        self.assertEqual(await self._as_async(store.checkout)(ref), resource)

  def test_commit_cyclic_fails(self):
    """Tests that commits with cylic resource graphs fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
      r1 = SimpleResource(x=1, y=2.0)
      r2 = SimpleResource(x=1, y=2.0)
      r1.x = r2  # type: ignore
      r2.x = r1  # type: ignore

      with enact.Store(backend=enact.FileBackend(tmpdir)):
        with self.assertRaises(acyclic.CycleDetected):
          enact.commit(r1)

  def test_ref_distribution_key(self):
    """Makes sure that refs have a distribution key."""
    self.assertEqual(
      enact.Ref.type_key().distribution_key,
      types.TypeKey(version.DIST_NAME, version.__version__))

  async def test_backend_get_types(self):
    """Tests that getting types work."""
    backend = enact.InMemoryBackend()
    with enact.Store(backend):
      r1 = enact.commit(
        SimpleResource(
          1,
          SimpleResource(
            2,
            SimpleResource(0, {1, 2}))))  # set  # type: ignore
      r2 = enact.commit([(1,2)])  # tuple in list
      r3 = enact.Ref.from_id('{"digest": "fake_id"}')
      # pylint: disable=unbalanced-tuple-unpacking
      for async_ in (False, True):
        self._async = async_
        t1, t2, t3 = await self._as_async(
          backend.get_type_keys)((r1.id, r2.id, r3.id))
        self.assertIsNone(t3)

        expected_t1 = {
          enact.Ref.type_key(),
          SimpleResource.type_key(),
          type_wrappers.SetWrapper.type_key(),
        }
        self.assertEqual(t1, expected_t1)

        expected_t2 = {
          enact.Ref.type_key(),
          resource_registry.ListWrapper.type_key(),
          type_wrappers.TupleWrapper.type_key()
        }
        self.assertEqual(t2, expected_t2)

  async def test_backend_get_dependency_graph(self):
    """Tests that getting dependency graphs works."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      for store in (enact.InMemoryStore(), enact.FileStore(tmp_dir)):
        with store:
          r1 = enact.commit(5)
          r2 = enact.commit([r1])
          r3 = enact.commit([r1])
          r4 = enact.commit([r2, r3, r1])
          fake_ref = enact.Ref.from_id('{"digest": "fake_id"}')

          for async_ in (False, True):
            self._async = async_
            graph = await self._as_async(store.get_dependency_graph)(
              (r1, r2, r3, r4, fake_ref))
            expected_graph = {
              r4: {r2, r3, r1},
              r3: {r1},
              r2: {r1},
              r1: set(),
              fake_ref: None}

            self.assertEqual(graph, expected_graph)

  def test_backend_get_dependency_graph_depth(self):
    """Tests that getting dependency graphs up to a certain depth works."""
    backend = enact.InMemoryBackend()

    with enact.Store(backend):
      refs = [enact.commit(0)]
      for _ in range(20):
        refs.append(enact.commit(refs[-1]))
      graph = backend.get_dependency_graph((refs[-1].id,), max_depth=10)
      self.assertEqual(set(graph.keys()), {ref.id for ref in refs[-11:]})

  async def test_get_transitive_type_requirements(self):
    """Tests that getting transitive type requirements works."""
    with enact.Store() as store:
      r1 = enact.commit(5)
      r2 = enact.commit([r1])
      r3 = enact.commit(SimpleResource(1, 2.0))
      r4 = enact.commit({r2, r3})

      expected_type_requirements = {
        enact.Ref.type_key(),
        resource_registry.IntWrapper.type_key(),
        resource_registry.ListWrapper.type_key(),
        SimpleResource.type_key(),
        type_wrappers.SetWrapper.type_key(),
      }

      for async_ in (False, True):
        self._async = async_
        type_requirements = await self._as_async(
          store.get_transitive_type_requirements)(r4)
        self.assertEqual(type_requirements, expected_type_requirements)

  def test_get_transitive_type_requirements_fails(self):
    """Tests that the transitive type fails if a reference is not present."""
    with enact.Store() as store:
      fake_ref = enact.Ref.from_id('{"digest": "fake_id"}')
      r = enact.commit(fake_ref)
      with self.assertRaises(references.NotFound):
        store.get_transitive_type_requirements(r)

  async def test_get_distribution_requirements(self):
    """Tests that getting distribution requirements works."""
    with enact.Store() as store:
      r1 = enact.commit(5)
      r2 = enact.commit([r1])
      r3 = enact.commit(SimpleResource(1, 2.0))
      r4 = enact.commit({r2, r3})

      # We don't have a distribution key for SimpleResource.
      with self.assertRaises(references.DistributionKeyError):
        store.get_distribution_requirements(r4)

      expected_dist_requirements = {
        types.DistributionKey(version.DIST_NAME, version.__version__),
      }

      for async_ in (False, True):
        self._async = async_
        dist_requirements = await self._as_async(
            store.get_distribution_requirements)(
          r4, expect_distribution_key=False)

        self.assertEqual(dist_requirements, expected_dist_requirements)

  async def test_type_registration(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      backends: List[references.StorageBackend] = [
        enact.FileBackend(tmp_dir), enact.InMemoryBackend()]

      for backend in backends:
        for async_ in (False, True):
          with self.subTest(backend=type(backend).__name__, async_=async_):
            self._async = async_
            with enact.Store(backend) as store:
              _ = await self._as_async(store.commit)(SimpleResource(1, 2.0))
              attributes = await self._as_async(backend.get_type)(
                SimpleResource.type_key())
              self.assertEqual(
                attributes,
                {
                  'x': types.Int(),
                  'y': types.Float()
                }
              )

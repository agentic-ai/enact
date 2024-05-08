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
import unittest

import enact
from enact import interfaces
from enact import references
from enact import serialization
from enact import contexts
from enact import resource_registry
from enact import acyclic
from enact import type_wrappers
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
    if data.type_info != JsonPackedResource.type_info():
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
      links={})


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
    ref, packed = enact.Ref.pack(value)
    self.assertIsInstance(
      resource_registry.from_resource_dict(packed.data),
      resource_registry.IntWrapper)
    self.assertIsInstance(ref, enact.Ref)
    self.assertEqual(ref, packed.ref())
    packed.ref().verify(packed.data)
    self.assertEqual(packed.ref().unpack(packed), 1)


class StoreTest(unittest.TestCase):
  """A test for stores."""

  def test_commit_has_get(self):
    """Create a store and test that commit, has, and get work."""
    store = enact.Store()
    resource = SimpleResource(x=1, y=2.0)
    ref = store.commit(resource)
    self.assertTrue(store.has(ref))
    self.assertEqual(store.checkout(ref), resource)
    self.assertNotEqual(id(store.checkout(ref)), id(resource))

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

  def test_wrapped_ref(self):
    store = enact.Store()
    for val in [None, 0, 0.0, 'str', True, bytes([1, 2, 3]),
                [1, 2, 3], {'a': 1}]:
      with self.subTest(val):
        ref = store.commit(val)
        self.assertEqual(store.checkout(ref), val)

  def test_caching(self):
    """Test that caching works correctly."""
    store = enact.Store()
    resource = SimpleResource(x=1, y=2.0)
    ref = store.commit(resource)

    # checkout() works because cached reference is correct.
    ref.checkout().x = 10

    # checkout() fails because cached reference is incorrect.
    with self.assertRaises(contexts.NoActiveContext):
      ref.checkout()

    with store:
      # Refetches the correct resource from the store.
      self.assertEqual(ref.checkout().x, 1)

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
    _, packed = ref.pack(value)
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

  def test_file_backend(self):
    """Tests the file backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
      store = enact.Store(backend=enact.FileBackend(tmpdir))
      resource = SimpleResource(x=1, y=2.0)
      ref = store.commit(resource)
      self.assertTrue(store.has(ref))
      self.assertEqual(store.checkout(ref), resource)

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

  def test_ref_distribution_info(self):
    """Makes sure that refs have a distribution info."""
    self.assertEqual(
      enact.Ref.type_info().distribution_info,
      interfaces.TypeInfo(version.DIST_NAME, version.__version__))

  def test_backend_get_types(self):
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
      t1, t2, t3 = backend.get_types((r1.id, r2.id, r3.id))
      self.assertIsNone(t3)

      expected_t1 = {
        enact.Ref.type_info(),
        SimpleResource.type_info(),
        type_wrappers.SetWrapper.type_info(),
      }
      self.assertEqual(t1, expected_t1)

      expected_t2 = {
        enact.Ref.type_info(),
        resource_registry.ListWrapper.type_info(),
        type_wrappers.TupleWrapper.type_info()
      }
      self.assertEqual(t2, expected_t2)

  def test_backend_get_dependency_graph(self):
    """Tests that getting dependency graphs works."""
    backend = enact.InMemoryBackend()

    with enact.Store(backend):
      r1 = enact.commit(5)
      r2 = enact.commit([r1])
      r3 = enact.commit([r1])
      r4 = enact.commit([r2, r3, r1])
      fake_ref = enact.Ref.from_id('{"digest": "fake_id"}')

      graph = backend.get_dependency_graph(
        (r1.id, r2.id, r3.id, r4.id, fake_ref.id))
      expected_graph = {
        r4.id: {r2.id, r3.id, r1.id},
        r3.id: {r1.id},
        r2.id: {r1.id},
        r1.id: set(),
        fake_ref.id: None}

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

  def test_get_transitive_type_requirements(self):
    """Tests that getting transitive type requirements works."""
    with enact.Store() as store:
      r1 = enact.commit(5)
      r2 = enact.commit([r1])
      r3 = enact.commit(SimpleResource(1, 2.0))
      r4 = enact.commit({r2, r3})

      expected_type_requirements = {
        enact.Ref.type_info(),
        resource_registry.IntWrapper.type_info(),
        resource_registry.ListWrapper.type_info(),
        SimpleResource.type_info(),
        type_wrappers.SetWrapper.type_info(),
      }

      type_requirements = store.get_transitive_type_requirements(r4)
      self.assertEqual(type_requirements, expected_type_requirements)

  def test_get_transitive_type_requirements_fails(self):
    """Tests that the transitive type fails if a reference is not present."""
    with enact.Store() as store:
      fake_ref = enact.Ref.from_id('{"digest": "fake_id"}')
      r = enact.commit(fake_ref)
      with self.assertRaises(references.NotFound):
        store.get_transitive_type_requirements(r)

  def test_get_distribution_requirements(self):
    """Tests that getting distribution requirements works."""
    with enact.Store() as store:
      r1 = enact.commit(5)
      r2 = enact.commit([r1])
      r3 = enact.commit(SimpleResource(1, 2.0))
      r4 = enact.commit({r2, r3})

      # We don't have a distribution info for SimpleResource.
      with self.assertRaises(references.DistributionInfoError):
        store.get_distribution_requirements(r4)

      expected_dist_requirements = {
        interfaces.DistributionInfo(version.DIST_NAME, version.__version__),
      }

      dist_requirements = store.get_distribution_requirements(
        r4, expect_distribution_info=False)

      self.assertEqual(dist_requirements, expected_dist_requirements)

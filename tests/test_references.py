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
    if not issubclass(data.type, JsonPackedResource):
      raise enact.RefError('Resource is not a JsonPackedResource.')
    json_packed = data.type.from_resource_dict(data)
    unpacked_dict = serialization.JsonSerializer().deserialize(
      json_packed.contents)
    return unpacked_dict.to_resource()

  @classmethod
  def pack(cls, resource: interfaces.ResourceBase) -> references.PackedResource:
    """Packs the resource."""
    return references.PackedResource(
      JsonPackedResource(serialization.JsonSerializer().serialize(
        resource.to_resource_dict())).to_resource_dict(),
      ref=cls.from_resource(resource),
      links=set())


class RefTest(unittest.TestCase):
  """A test for refs."""

  def test_verify_pack_unpack(self):
    """Test that packing and unpacking works."""
    resource = SimpleResource(x=1, y=2.0)
    packed = enact.Ref.pack(resource)
    packed.ref.verify(packed)
    unpacked = packed.ref.unpack(packed)
    self.assertEqual(unpacked, resource)

  def test_custom_ref_type(self):
    """Tests custom ref types works."""
    resource = SimpleResource(x=1, y=2.0)
    packed = JsonPackedRef.pack(resource)
    unpacked = packed.ref.unpack(packed)
    packed.ref.verify(packed)
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
    packed = enact.Ref.pack(value)
    self.assertIsInstance(
      packed.data.to_resource(),
      resource_registry.IntWrapper)
    packed.ref.verify(packed)
    self.assertEqual(packed.ref.unpack(packed), 1)


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

    # get() works because cached reference is correct.
    ref.checkout().x = 10

    # get() fails because cached reference is incorrect.
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
      old_ref = ref.deepcopy_resource()
      with ref.modify() as elems:
        elems.append(3)
      self.assertEqual(ref.checkout(), [0, 1, 2, 3])
      self.assertNotEqual(ref, old_ref)

  def test_pack_none(self):
    """Tests packing none."""
    store = enact.Store()
    value = None
    ref = store.commit(value)
    packed = ref.pack(value)
    ref.verify(packed)

  def test_packed_links(self):
    """Tests links are properly tracked during packing."""
    store = enact.Store()
    with store:
      x = enact.commit(1)
      y = enact.commit(2.0)
      r1 = SimpleResource(x, [{'test': (y, y)}])  # type: ignore
      self.assertEqual(
        enact.Ref.pack(r1).links,
        {x.id: x.to_resource_dict(),
         y.id: y.to_resource_dict()})

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
        with self.assertRaises(interfaces.FieldTypeError):
          enact.commit(r1)

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
      ref=cls.from_resource(resource))


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

  def test_pack_none(self):
    """Tests packing the none resource."""
    store = enact.Store()
    resource = interfaces.NoneResource()
    ref = store.commit(resource)
    packed = ref.pack(resource)
    ref.verify(packed)

  def test_file_backend(self):
    """Tests the file backend."""
    with tempfile.TemporaryDirectory() as tmpdir:
      store = enact.Store(backend=enact.FileBackend(tmpdir))
      resource = SimpleResource(x=1, y=2.0)
      ref = store.commit(resource)
      self.assertTrue(store.has(ref))
      self.assertEqual(store.checkout(ref), resource)

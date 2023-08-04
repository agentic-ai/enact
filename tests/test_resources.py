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

"""Tests for resources."""


import dataclasses
from typing import Any
import unittest

import enact


@dataclasses.dataclass
class SimpleResource(enact.Resource):
  a: Any
  b: Any
  c: Any


@dataclasses.dataclass
class OtherSimpleResource(enact.Resource):
  a: Any
  b: Any
  c: Any


class ResourceTest(unittest.TestCase):
  """Tests for the resource protocol."""

  def test_field_names(self):
    """Tests that the fields are correct."""
    self.assertEqual(
      list(SimpleResource.field_names()),
      ['a', 'b', 'c'])

  def test_field_values(self):
    """Tests that the field values are correct."""
    r = SimpleResource(1, 2, 3)
    self.assertEqual(
      list(r.field_values()),
      [1, 2, 3])

  def test_field_items(self):
    """Tests that field_items works."""
    r = SimpleResource(1, 2, 3)
    self.assertEqual(
      list(r.field_items()),
      [('a', 1), ('b', 2), ('c', 3)])

  def test_type_id_different_fields(self):
    """Tests that the type id differs based on field."""
    @dataclasses.dataclass
    class R(enact.Resource):
      a: Any
    type_id = R.type_id()
    @dataclasses.dataclass
    class R(enact.Resource):  # type: ignore
      b: Any
    self.assertNotEqual(type_id, R.type_id())

  def test_type_id_same_fields(self):
    """Tests that the type id is correct."""
    @dataclasses.dataclass
    class R(enact.Resource):
      a: Any
    type_id = R.type_id()
    @dataclasses.dataclass
    class R(enact.Resource):  # type: ignore
      a: Any
    self.assertEqual(type_id, R.type_id())

class RefTest(unittest.TestCase):

  def test_digest_identical(self):
    """Test that digest is identical for identical resources."""
    a = SimpleResource(1, 2, 3)
    b = SimpleResource(1, 2, 3)
    r1 = enact.Ref.pack(a)
    r2 = enact.Ref.pack(b)
    self.assertEqual(r1, r2)

  def test_typename_changes_hash(self):
    """Tests that changing the typename changes the hash."""
    a = SimpleResource(1, 2, 3)
    b = OtherSimpleResource(1, 2, 3)
    r1 = enact.Ref.pack(a)
    r2 = enact.Ref.pack(b)
    self.assertNotEqual(r1.ref, r2.ref)

  def test_value_changes_hash(self):
    """Tests that changing values changes the hash."""
    a = SimpleResource(1, 2, 3)
    b = SimpleResource(1, 2, 4)
    r1 = enact.Ref.pack(a)
    r2 = enact.Ref.pack(b)
    self.assertNotEqual(r1.ref, r2.ref)

  def test_hash_complex_nested(self):
    """Tests that nested resources are hashed correctly."""
    a = SimpleResource(
      [1, None, 2.0, True, False],
      [{'a': {'b': {'c': 1}}}],
      enact.Ref('1234'))
    b = SimpleResource(
      [1, None, 2.0, True, False],
      [{'a': {'b': {'c': 1}}}],
      enact.Ref('1234'))
    self.assertEqual(
      enact.Ref.pack(a),
      enact.Ref.pack(b))

  def test_nested_resource(self):
    """Tests nesting resources."""
    a = SimpleResource(SimpleResource(1, 2, 3), 2, 3)
    enact.Ref.pack(a)

  def test_unknown_field_error(self):
    """Tests that including an unknown type will raise an error."""
    class X:
      pass
    a = SimpleResource(X(), 2, 3)
    with self.assertRaises(enact.FieldTypeError):
      enact.Ref.pack(a)

  def test_deepy_copy_resource(self):
    """Tests that the resource can be deep-copied."""
    a = SimpleResource(SimpleResource(1, 2, 3), [4, None],
                       {'a': 0.0, 'b': [True, False]})
    b = a.deep_copy_resource()
    self.assertEqual(a, b)
    self.assertNotEqual(id(a), id(b))
    self.assertNotEqual(id(a.a), id(b.a))
    self.assertNotEqual(id(a.b), id(b.b))
    self.assertNotEqual(id(a.c), id(b.c))
    self.assertEqual(enact.Ref.pack(a), enact.Ref.pack(b))

  def test_set_from(self):
    """Tests that set-from works as expected."""
    x = SimpleResource(SimpleResource(1, 2, 3), [4, None],
                       {'a': 0.0, 'b': [True, False]})
    y = SimpleResource(None, None, None)
    y.set_from(x)
    self.assertEqual(y, x)
    x.a.a = 5
    self.assertNotEqual(y, x)

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

"""Tests for the serialize module."""
import dataclasses
from typing import Any, Dict, List, Type
import unittest

import enact
from enact import serialization
from enact import resource_registry

import random_value

@enact.register
@dataclasses.dataclass
class AllTypesResource(enact.Resource):
  i: int
  f: float
  bl: bool
  b: bytes
  s: str
  n: Any
  r: enact.Ref
  m: Dict
  l: List
  t: Type[enact.Resource]


class JsonSerializerTest(unittest.TestCase):
  """Tests the JSON serializer."""

  def setUp(self):
    """Sets up the test case."""
    self.registry = resource_registry.Registry()
    self.registry.register(enact.Ref)
    self.serializer = serialization.JsonSerializer(self.registry)

  def test_serialize_deserialize(self):
    """Tests that serialization/deserialization works."""
    self.registry.register(AllTypesResource)
    resource = AllTypesResource(
      i=2, f=3.0, bl=True, b=b'bytes', s='test', n=None,
      r=enact.Ref("12314"), m={'a': ['test']}, l=[{'b': 1}, {'c': 2}],
      t=AllTypesResource)
    got = self.serializer.serialize(resource.to_resource_dict())
    deserialized = self.serializer.deserialize(got).to_resource()
    self.assertEqual(deserialized, resource)

  def test_serialize_deserialize_fuzz(self):
    """Fuzz test serialization and deserialization."""
    self.registry.register(random_value.R)
    for _ in range(100):
      resource = random_value.rand_resource()
      got = self.serializer.serialize(resource.to_resource_dict())
      deserialized = self.serializer.deserialize(got).to_resource()
      self.assertEqual(deserialized, resource)

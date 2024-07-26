# Copyright 2024 Agentic.AI Corporation.
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
"""Tests for utils."""

import unittest

from enact import types
from enact import resource_registry


class TypesTest(unittest.TestCase):
  """Tests for the types module."""

  def test_from_to_json(self):
    """Tests that the to_json and from_json methods work as expected."""
    for t in (types.Int(), types.Str(), types.Float(), types.Bool(),
              types.Bytes(),
              types.List(None), types.List(types.List(types.Int())),
              types.Dict(None), types.Dict(types.Dict(types.Str())),
              types.NoneType(),
              types.ResourceType(resource_registry.BoolWrapper.type_key()),
              types.Union(tuple([types.Int(), types.Str(), types.NoneType()]))):
      got = types.TypeDescriptor.from_json(t.to_json())
      self.assertEqual(got, t)

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

"""Tests for the resource_registry module."""

from typing import cast
import unittest

import numpy as np
import PIL.Image

import enact


class TestResourceTypes(unittest.TestCase):
  """Tests all the basic resource types."""

  TYPES = [
    (enact.Int, 3),
    (enact.Float, 2.5),
    (enact.Bytes, b'babc'),
    (enact.Str, 'abc'),
    (enact.NPArray, np.array([0.0, 1.0, 2.0])),
    (enact.Image, PIL.Image.new('RGB', (10, 10), 'red')),
    (enact.List, [1, 2, 3]),
  ]

  def setUp(self):
    """Sets up the test."""
    self.store = enact.Store()

  def test_ref_deref(self):
    """Test storing and dereferencing a resource from a store."""
    for resource_type, value in self.TYPES:
      with self.subTest(resource_type=resource_type):
        resource = cast(enact.ResourceBase, resource_type(value))
        ref = self.store.commit(resource)
        ref2 = self.store.commit(ref.checkout())
        self.assertEqual(ref, ref2)

        for v1, v2 in zip(resource.field_values(),
                          ref2.checkout().field_values()):
          self.assertEqual(v1, v2)

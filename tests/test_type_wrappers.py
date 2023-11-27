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

"""Tests for the type_wrappers module."""

import unittest

import numpy as np
import PIL.Image

import enact

from enact import type_wrappers


class TestTypeWrappers(unittest.TestCase):
  """Tests all the non-fieldvalue wrappers."""

  TYPES = [
    (type_wrappers.NPArrayWrapper, np.array([0.0, 1.0, 2.0])),
    (type_wrappers.PILImageWrapper, PIL.Image.new('RGB', (10, 10), 'red')),
    (type_wrappers.TupleWrapper, (1, 2, 3)),
    (type_wrappers.SetWrapper, {1, 2, 3}),
    (type_wrappers.NPFloat16Wrapper, np.float16(1.0)),
    (type_wrappers.NPFloat32Wrapper, np.float32(1.0)),
    (type_wrappers.NPFloat64Wrapper, np.float64(1.0)),
  ]

  def setUp(self):
    """Sets up the test."""
    self.store = enact.Store()

  def test_ref_deref(self):
    """Test storing and dereferencing a resource from a store."""
    for resource_type, value in self.TYPES:
      with self.subTest(resource_type=resource_type):
        restored = self.store.commit(value).checkout()
        self.assertIsInstance(restored, type(value))
        if isinstance(restored, np.ndarray):
          assert isinstance(value, np.ndarray)
          self.assertSequenceEqual(restored.tolist(), value.tolist())
        elif isinstance(restored, PIL.Image.Image):
          assert isinstance(value, PIL.Image.Image)
          self.assertSequenceEqual(list(restored.getdata()),
                                   list(value.getdata()))
        else:
          self.assertEqual(restored, value)

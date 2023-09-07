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

"""Tests for the pretty printer."""

import unittest

import enact


class PrettyPrinterTest(unittest.TestCase):
  """Tests for the pretty printer."""

  def test_pformat_field_value(self):
    """Tests that printing field values works."""
    test_cases = [
      (None, 'None'),
      (True, 'True'),
      (False, 'False'),
      (42, '42'),
      (3.14, '3.14'),
      (b'hello', '<5 bytes>'),
      ("hello", '\'hello\''),
      ([1, 2, 3], "[\n  1\n  2\n  3]"),
      ({"a": 1, "b": 2}, '{\n  a: 1\n  b: 2}')]
    for field_value, expected in test_cases:
      with self.subTest(field_value=field_value):
        formatted = enact.pformat(field_value)
        self.assertEqual(formatted, expected)

  def test_pformat_wrapped_resource(self):
    """Tests that printing wrapped values works."""
    formatted = enact.pformat((1, 2, 3))
    self.assertEqual(
      formatted,
'''TupleWrapper:
  value:
    [
      1
      2
      3]''')

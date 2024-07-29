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

"""Tests for the type_inference module."""

import unittest
import typing

from enact import type_inference
from enact import types


class UnsupportedType:
  pass

FULLY_SUPPORTED_TESTCASES = [
  (int, types.Int()),
  (str, types.Str()),
  (float, types.Float()),
  (bytes, types.Bytes()),
  (bool, types.Bool()),
  (list, types.List()),
  (typing.List, types.List()),
  (dict, types.Dict()),
  (typing.Dict, types.Dict()),
  # Generic types:
  (typing.List[typing.List[int]], types.List(types.List(types.Int()))),
  (typing.Dict[str, typing.List[int]], types.Dict(types.List(types.Int()))),
  # Optional types:
  (typing.Optional[int], types.Union(
    tuple([types.Int(), types.NoneType()]))),
  # Union types:
  (typing.Union[int, str, float], types.Union(
    tuple([types.Int(), types.Str(), types.Float()]))),
]

PARTIALLY_SUPORTED_TESTCASES = [
  (typing.Dict[str, UnsupportedType], types.Dict()),
  (typing.List[UnsupportedType], types.List()),
  (typing.Dict[str, typing.Union[UnsupportedType, int]], types.Dict()),
  (typing.List[typing.Dict[str, UnsupportedType]],
   types.List(types.Dict())),
]

UNSUPPORTED_TESTCASES = [
  (typing.Union, None),
  (typing.Dict[int, typing.List[int]], None),
  (UnsupportedType, None),
]

class TestTypeInference(unittest.TestCase):
  """Tests the type inference module."""

  def test_type_inference(self):
    """Test type inference."""
    test_cases = (
      FULLY_SUPPORTED_TESTCASES +
      PARTIALLY_SUPORTED_TESTCASES +
      UNSUPPORTED_TESTCASES)
    for annotation, type_descriptor in test_cases:
      with self.subTest(annotation=str(annotation)):
        got = type_inference.from_annotation(annotation, strict=False)
        self.assertEqual(
          got,
          type_descriptor)

  def test_type_inference_strict_passes(self):
    """Test type inference with strict parsing on passing values."""
    test_cases = (
      FULLY_SUPPORTED_TESTCASES)
    for annotation, type_descriptor in test_cases:
      with self.subTest(annotation=str(annotation)):
        got = type_inference.from_annotation(annotation, strict=False)
        self.assertEqual(
          got,
          type_descriptor)

  def test_type_inference_strict_fails(self):
    """Test type inference with strict parsing on failing values."""
    raising_test_cases = (
      PARTIALLY_SUPORTED_TESTCASES +
      UNSUPPORTED_TESTCASES)
    for annotation, _ in raising_test_cases:
      with self.subTest(annotation=str(annotation)):
        with self.assertRaises(type_inference.TypeInferenceFailed):
          type_inference.from_annotation(annotation, strict=True)

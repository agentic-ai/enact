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

"""Tests registration module."""

import unittest

import enact

class RegistrationTest(unittest.TestCase):
  """Tests for the enact.register function."""

  def test_type_inference(self):
    """Tests invokable type inference."""
    @enact.register
    class MyInvokable(enact.Invokable):
      """Invokable for testing."""

      def call(self, x: int) -> str:
        return str(x)

    self.assertEqual(MyInvokable.get_input_type(), int)
    self.assertEqual(MyInvokable.get_output_type(), str)

  def test_type_inference_none_return(self):
    """Tests invokable type inference if none is returned."""
    @enact.register
    class MyInvokable(enact.Invokable):
      """Invokable for testing."""

      def call(self, x: int) -> None:
        pass

    self.assertEqual(MyInvokable.get_input_type(), int)
    self.assertEqual(MyInvokable.get_output_type(), type(None))

  def test_type_inference_no_type(self):
    """Tests invokable type inference if no types are set."""
    @enact.register
    class MyInvokable(enact.Invokable):
      """Invokable for testing."""

      def call(self, x):
        pass

    self.assertEqual(MyInvokable.get_input_type(), None)
    self.assertEqual(MyInvokable.get_output_type(), None)

  def test_type_inference_no_arg(self):
    """Tests invokable type inference if the call function has no args."""
    @enact.register
    class MyInvokable(enact.Invokable):
      """Invokable for testing."""
      def call(self):
        pass

    self.assertEqual(MyInvokable.get_input_type(), type(None))
    self.assertEqual(MyInvokable.get_output_type(), None)

  def test_register_too_many_args(self):
    """Test too many argument errors on invokables."""
    class TooManyArgs(enact.Invokable):
      """Invokable for testing."""
      def call(self, x, y):
        pass
    with self.assertRaises(TypeError):
      enact.register(TooManyArgs)

  def test_register_keyword_only_args(self):
    """Test wrong kind error on invokables."""
    class KeywordOnly(enact.Invokable):
      """Invokable for testing."""
      def call(self, *, x):
        pass
    with self.assertRaises(TypeError):
      enact.register(KeywordOnly)

  def test_register_no_self_arg(self):
    """Test wrong kind error on invokables."""
    class StaticCall(enact.Invokable):
      """Invokable for testing."""
      @staticmethod
      def call(x):
        pass
    with self.assertRaises(TypeError):
      enact.register(StaticCall)

  def test_register_type_not_resource(self):
    """Test type error on non-resource."""
    class NotAResource:
      """A class that is not a resource."""
      pass
    with self.assertRaises(TypeError):
      enact.register(NotAResource)

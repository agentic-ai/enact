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

"""Tests for invocation generators."""

import dataclasses
import tempfile
import unittest

import enact


@enact.typed_invokable(int, str)
@dataclasses.dataclass
class IntToStr(enact.Invokable):
  """An invokable that converts an int to a string."""
  salt: str = ''

  def call(self, value: int) -> str:
    return str(value) + self.salt


class InvocationGeneratorTest(unittest.TestCase):
  """Tests invocation_generator"""

  def setUp(self):
    # pylint: disable=consider-using-with
    self.dir = tempfile.TemporaryDirectory()
    self.backend = enact.FileBackend(self.dir.name)
    self.store = enact.Store(self.backend)

  def tearDown(self):
    self.dir.cleanup()

  def test_send_without_next(self):
    """Tests that send without next fails on invokable generator."""
    with self.store:
      inv_gen = enact.InvocationGenerator(
        IntToStr(), enact.commit(3))
      with self.assertRaisesRegex(TypeError, '.*non-None.*'):
        inv_gen.send(3)

  def test_send_none_without_next(self):
    """Tests that send None without next works."""
    with self.store:
      inv_gen = enact.InvocationGenerator(
        IntToStr(), enact.commit(3))
      with self.assertRaises(StopIteration):
        inv_gen.send(None)

  def test_send_flow(self):
    """Test an invokable generator in a send-based flow."""
    @enact.typed_invokable(type(None), int)
    class SumUserRequests(enact.Invokable):
      def call(self):
        return (
          enact.request_input(int) +
          enact.request_input(int) +
          enact.request_input(int))

    with self.store:
      inv_gen = enact.InvocationGenerator(
        SumUserRequests(), enact.commit(None))
      input_request = next(inv_gen)
      for i in range(5):
        assert isinstance(input_request, enact.InputRequest)
        self.assertFalse(inv_gen.complete)
        try:
          input_request = inv_gen.send(i)
        except StopIteration:
          break
      self.assertEqual(
        inv_gen.invocation.get_output(), 0 + 1 + 2)

  def test_set_input_flow(self):
    """Test an invokable generator in a send-based flow."""
    @enact.typed_invokable(type(None), int)
    class SumUserRequests(enact.Invokable):
      def call(self):
        return (
          enact.request_input(int) +
          enact.request_input(int) +
          enact.request_input(int))

    with self.store:
      inv_gen = enact.InvocationGenerator(
        SumUserRequests(), enact.commit(None))
      for i, _ in enumerate(inv_gen):
        self.assertFalse(inv_gen.complete)
        inv_gen.set_input(i)
      self.assertEqual(
        inv_gen.invocation.get_output(), 0 + 1 + 2)

  def test_wrapped_functions(self):
    """Tests invocation generator for wrapped functions."""
    @enact.register
    def request_plus_one() -> int:
      return enact.request_input(int) + 1

    @enact.register
    def sum_requests() -> int:
      return request_plus_one() + request_plus_one()

    with self.store:
      inv_gen: enact.InvocationGenerator = (
          enact.InvocationGenerator.from_callable(sum_requests))
      for i, _ in enumerate(inv_gen):
        self.assertFalse(inv_gen.complete)
        inv_gen.set_input(i)
      invocation = inv_gen.invocation
      self.assertEqual(invocation.get_output(), 1 + 2)
      self.assertEqual(invocation.get_child(0).get_output(), 1)
      self.assertEqual(invocation.get_child(1).get_output(), 2)

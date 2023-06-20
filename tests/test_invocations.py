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

"""Tests for invocations."""

import dataclasses
import time
from typing import Optional
import unittest
from unittest import mock

import enact
from enact import invocations


@dataclasses.dataclass
class Int(enact.Resource):
  """An int resource."""
  v: int


@dataclasses.dataclass
class Str(enact.Resource):
  """A string resource."""
  v: str


@enact.typed_invokable(Int, Str)
@dataclasses.dataclass
class IntToStr(enact.Invokable):
  """An invokable that converts an int to a string."""
  salt: str = ''

  def call(self, input: Int) -> Str:
    return Str(v=str(input.v) + self.salt)


@enact.typed_invokable(Int, Str)
class WrongOutputType(enact.Invokable):

  def call(self, input: Int) -> Int:
    return input


@enact.typed_invokable(Int, Int)
class AddOne(enact.Invokable):
  def call(self, input: Int) -> Int:
    return Int(v=input.v + 1)


@enact.typed_invokable(Int, Int)
@dataclasses.dataclass
class NestedFunction(enact.Invokable):
  fun: AddOne = AddOne()
  iter: int = 10
  fail_on: Optional[int] = None

  def call(self, input: Int) -> Int:
    for i in range(self.iter):
      if i == self.fail_on:
        # Produce wrong input type to cause error.
        input = Str(v=str(input.v))  # type: ignore
      input = self.fun(input)
    return input


class InvocationsTest(unittest.TestCase):
  """Tests invocations."""

  def setUp(self):
    self.backend = enact.InMemoryBackend()
    self.store = enact.Store(self.backend)

  def test_typed_invokable(self):
    """"Test that the decorator works as expected."""
    fun = IntToStr('salt')
    self.assertEqual(fun.get_input_type(), Int)
    self.assertEqual(fun.get_output_type(), Str)
    output = fun.call(Int(v=1))
    # Input and output types should not show up as fields.
    self.assertEqual(list(fun.field_names()), ['salt'])
    # The following line should typecheck correctly.
    self.assertEqual(output.v, '1salt')

  def test_typecheck_input(self):
    fun = IntToStr('salt')
    with self.assertRaises(TypeError):
      fun(Str(v='1'))

  def test_typecheck_output(self):
    fun = WrongOutputType()
    with self.assertRaises(TypeError):
      fun(Str(v='1'))

  def test_auto_input_args(self):
    self.assertEqual(
      IntToStr()(1).v, '1')

  def test_auto_input_kwargs(self):
    self.assertEqual(
      IntToStr()(v=1).v, '1')

  def test_invoke_simple(self):
    with self.store:
      fun = IntToStr('salt')
      invocation = fun.invoke(
        enact.commit(Int(v=1)))
      want: enact.Invocation = enact.Invocation(
        request=enact.commit(
          enact.Request(enact.commit(fun),
                        enact.commit(Int(v=1)))),
        response=enact.commit(
          enact.Response(
            invokable=enact.commit(fun),
            output=enact.commit(Str(v='1salt')),
            raised=None,
            raised_here=False,
            children=[])),
        timestamp_ns=invocation.timestamp_ns)
    self.assertEqual(
      invocation,
      want)

  def test_call_nested(self):
    """Test calling a nested function."""
    self.assertEqual(NestedFunction()(1).v, 11)

  def test_invoke_nested(self):
    """Test invoking a nested function."""
    with self.store:
      invocation = NestedFunction().invoke(
        enact.commit(Int(v=1)))
      output = invocation.get_output()
      self.assertEqual(output.v, 11)
      self.assertEqual(len(list(invocation.get_children())), 10)
      for i, child in enumerate(invocation.get_children()):
        self.assertEqual(child.request.get().input.get().v, i + 1)
        output = child.get_output()
        self.assertEqual(output.v, i + 2)

  def test_invoke_fail(self):
    with self.store:
      invocation = NestedFunction(fail_on=3).invoke(
        enact.commit(Int(v=1)))
      self.assertFalse(invocation.successful())
      exception = invocation.get_raised()
      self.assertIsInstance(
        exception,
        enact.ExceptionResource)
      assert invocation.response.get().children
      self.assertEqual(len(invocation.response.get().children), 3)
      for i, child in enumerate(invocation.get_children()):
        if i < 3:
          self.assertEqual(child.request.get().input.get().v, i + 1)
          output = child.get_output()
          self.assertEqual(output.v, i + 2)
        else:
          self.assertEqual(child.request.get().input.get().v, 3)
          self.assertFalse(child.successful())
          exception = child.get_raised()
          self.assertIsInstance(
            exception,
            enact.ExceptionResource)

  def test_no_parent_on_invoke(self):
    """Tests that invocations are tracked in a fresh context."""
    with mock.patch.object(time, 'time_ns', return_value=0):
      with self.store as store:
        fun = IntToStr()
        with invocations.Builder(fun, store.commit(Int(v=1))) as builder:
          # Enter an unrelated context.
          nested = NestedFunction()
          inner_invocation = nested.invoke(store.commit(Int(1)))
          # The invocation should not be tracked in the builder.
          self.assertEqual(builder.children, [])
        # Redo the invocation outside the builder.
        outer_invocation = nested.invoke(store.commit(Int(1)))
        # Ensure that executing in the misleading context made no difference.
        self.assertEqual(outer_invocation, inner_invocation)

  def test_meta_invoke(self):
    """Tests that meta-invocations are tracked correctly."""
    @dataclasses.dataclass
    class MetaInvoke(enact.Invokable):
      invokable: enact.InvokableBase
      def call(self, input: enact.ResourceBase):
        return self.invokable.invoke(enact.commit(input))

    with self.store as store:
      fun = NestedFunction()
      meta_invocation = MetaInvoke(fun).invoke(store.commit(Int(v=1)))
      self.assertEqual(
        len(list(meta_invocation.get_children())), 0)
      invocation = meta_invocation.get_output()
      assert isinstance(invocation, enact.Invocation)
      self.assertEqual(
        len(list(invocation.get_children())), 10)

  def test_wrapped_resource(self):
    """Tests that exceptions are tracked as wrapped."""
    class PythonErrorOnInvoke(enact.Invokable):
      def call(self, input: enact.ResourceBase):
        raise ValueError('foo')
    with self.store as store:
      fun = PythonErrorOnInvoke()
      invocation = fun.invoke(store.commit(enact.Int(5)))
      self.assertIsInstance(
        invocation.get_raised(),
        enact.WrappedException)

  def test_raise_native_error(self):
    """Tests that exceptions are raised in native format."""
    class PythonErrorOnInvoke(enact.Invokable):
      def call(self, input: enact.ResourceBase):
        raise ValueError('foo')
    class ExpectValueError(enact.Invokable):
      def call(self, input: enact.ResourceBase):
        try:
          PythonErrorOnInvoke()(enact.Int(3))
        except ValueError:
          return enact.Str('Got value error')
        raise Exception('Expected ValueError')

    with self.store as store:
      fun = ExpectValueError()
      invocation = fun.invoke(store.commit(enact.Int(5)))
      self.assertEqual(
        invocation.get_output(),
        'Got value error')

  def test_raised_here(self):
    """Tests that the raised_here field is set correctly."""
    class PythonErrorOnInvoke(enact.Invokable):
      def call(self, input: enact.ResourceBase):
        raise ValueError('foo')

    @dataclasses.dataclass
    class SubCall(enact.Invokable):
      invokable: enact.Ref[enact.InvokableBase]
      def call(self, input: enact.ResourceBase):
        self.invokable.get()(input)

    with self.store as store:
      error_fun = PythonErrorOnInvoke()
      subcall_1 = SubCall(store.commit(error_fun))
      subcall_2 = SubCall(store.commit(subcall_1))

      invocation = subcall_2.invoke(store.commit(enact.Int(5)))
      self.assertFalse(invocation.get_raised_here())
      (subinvocation,) = invocation.get_children()
      self.assertFalse(subinvocation.get_raised_here())
      (subsubinvocation,) = subinvocation.get_children()
      self.assertTrue(subsubinvocation.get_raised_here())

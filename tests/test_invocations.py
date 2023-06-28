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
from typing import Optional, cast
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


@dataclasses.dataclass
@enact.typed_invokable(Int, Int)
class AddOne(enact.Invokable):
  fail: bool = False
  def call(self, input: Int) -> Int:
    if self.fail:
      raise ValueError('fail')
    return Int(v=input.v + 1)


@dataclasses.dataclass
class Fail(enact.Invokable):
  def call(self, arg: enact.ResourceBase) -> Int:
    raise ValueError('fail')


@enact.typed_invokable(Int, Int)
@dataclasses.dataclass
class NestedFunction(enact.Invokable):
  fun: enact.InvokableBase = AddOne()
  iter: int = 10
  fail_on: Optional[int] = None

  def call(self, input: Int) -> Int:
    for i in range(self.iter):
      if i == self.fail_on:
        input = Fail()(input)
      else:
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
      assert invocation.response().children
      self.assertEqual(len(invocation.response().children), 4)
      for i, child in enumerate(invocation.get_children()):
        if i < 3:
          self.assertEqual(child.request().input().v, i + 1)
          output = child.get_output()
          self.assertEqual(output.v, i + 2)
        else:
          self.assertEqual(child.request().input().v, 4)
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

  def test_input_changed_error(self):
    @dataclasses.dataclass
    class Changeable(enact.Resource):
      x: int = 0

    class ChangesInput(enact.Invokable):
      def call(self, input: Changeable):
        input.x += 1
        return input

    with self.assertRaises(enact.InputChanged):
      with self.store as store:
        ChangesInput().invoke(store.commit(Changeable()))

  def test_replay_call(self):
    """Test replaying a call."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(Int(v=0)))
      with enact.ReplayContext(
          subinvocations=[invocation],
          exception_override=lambda x: Int(v=100)):
        result = fun(Int(v=0))
      self.assertEqual(result.v, 106)

  def test_replay_modifies_invokable(self):
    @dataclasses.dataclass
    class Counter(enact.Invokable):
      call_count: int = 0
      def call(self, input: enact.ResourceBase):
        self.call_count += 1
        return input

    with self.store:
      counter = Counter()
      fun = NestedFunction(counter, iter=10)
      invocation = fun.invoke(
        enact.commit(Int(v=0)))
      self.assertEqual(counter.call_count, 10)

      # Modify output to ensure we got a replay and
      # not a reexecution.
      with invocation.response.modify() as response:
        response.output = enact.commit(Int(v=100))

      fun.fun = Counter()
      result = invocation.replay()
      self.assertEqual(result.get_output().v, 100)
      self.assertEqual(counter.call_count, 10)

  def test_replay_partial(self):
    """Test replaying a partial invocation."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(Int(v=0)))
      with invocation.response.modify() as response:
        del response.children[4:]
        with response.children[-1].modify() as child:
          with child.response.modify() as child_response:
            assert child_response.raised
            child_response.raised = None
            child_response.output = enact.commit(Int(v=100))
      with enact.ReplayContext(
          subinvocations=[invocation]):
        result = fun(Int(v=0))
      self.assertEqual(result.v, 106)

  def test_replay_call_on_mismatch_nonstrict(self):
    """Test non-strict replays are ignored if arguments don't match."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(Int(v=0)))
      with enact.ReplayContext(
          subinvocations=[invocation],
          exception_override=lambda x: Int(v=100), strict=False):
        with self.assertRaises(ValueError):
          # Exception override is not active since we're ignoring the
          # replay.
          fun(Int(v=1))

  def test_replay_call_on_mismatch_strict(self):
    """Test strict replays raise."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(Int(v=0)))
      with enact.ReplayContext(
          subinvocations=[invocation],
          exception_override=lambda x: Int(v=100), strict=True):
        with self.assertRaises(enact.ReplayError):
          fun(Int(v=1))

  def test_invoke_with_replay(self):
    """Test replays are ignored if arguments don't match."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(Int(v=0)))
      invocation = fun.invoke(
        enact.commit(Int(v=0)),
        replay_from=invocation,
        exception_override=lambda x: Int(v=100))
      self.assertEqual(invocation.get_output().v, 106)

  def test_request_input(self):
    class TextInput(enact.Invokable):
      def call(self, resource: enact.ResourceBase):
        raise enact.request_input(Str)

    fun = NestedFunction(fun=TextInput(), iter=2)
    with self.store:
      inputs = ['foo', 'bar', 'bish']
      invocation = fun.invoke(enact.commit(Str(v=inputs[0])))
      for cur_input, next_input in zip(inputs[:-1], inputs[1:]):
        raised = invocation.get_raised()
        assert isinstance(raised, enact.InputRequest)
        self.assertEqual(
          cast(Str, raised.input.get()).v, cur_input)
        invocation = raised.continue_invocation(
          invocation, Str(v=next_input))
      self.assertEqual(invocation.get_output().v, 'bish')
      child_outputs = [
        c.get_output().v for c in invocation.get_children()]
      self.assertEqual(child_outputs, inputs[1:])

  def test_input_requested_outside_invocation(self):
    class TextInput(enact.Invokable):
      def call(self, resource: enact.ResourceBase):
        raise enact.request_input(Str)
    with self.assertRaises(enact.InputRequestOutsideInvocation):
      TextInput()(Str(v='foo'))

  def test_request_input_infer_type(self):
    @enact.typed_invokable(Str, Str)
    class TextInput(enact.Invokable):
      def call(self, resource: enact.ResourceBase):
        raise enact.request_input(Str)
    with self.store:
      fun = TextInput()
      invocation = fun.invoke(enact.commit(Str(v='foo')))
      raised = invocation.get_raised()
      assert isinstance(raised, enact.InputRequest)
      self.assertEqual(raised.requested_type, Str)

  def test_request_inference_fails(self):
    class TextInput(enact.Invokable):
      def call(self, resource: enact.ResourceBase):
        raise enact.request_input()
    with self.store:
      fun = TextInput()
      with self.assertRaises(enact.RequestedTypeUndetermined):
        fun.invoke(enact.commit(Str(v='foo')))

  def test_request_input_class(self):
    with self.store:
      fun = enact.RequestInput(enact.Str, enact.Str('Context'))
      invocation = fun.invoke(enact.commit(enact.Str('foo')))
      raised = invocation.get_raised()
      assert isinstance(raised, enact.InputRequest)
      assert raised.context == 'Context'
      self.assertEqual(raised.requested_type, enact.Str)

  def test_empty_call_args(self):
    """Tests that empty call args are ok for NoneResource inputs."""
    @enact.typed_invokable(enact.NoneResource, Str)
    class TextInput(enact.Invokable):
      def call(self):
        return enact.Str('Foo')

    with self.store:
      fun = TextInput()
      invocation = fun.invoke()
      self.assertEqual(invocation.get_output(), 'Foo')


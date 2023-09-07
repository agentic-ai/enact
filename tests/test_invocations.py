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

import asyncio
import dataclasses
import random
import tempfile
import time
from typing import List, Optional, cast
import unittest
from unittest import mock

import enact
from enact import invocations


@enact.typed_invokable(int, str)
@dataclasses.dataclass
class IntToStr(enact.Invokable):
  """An invokable that converts an int to a string."""
  salt: str = ''

  def call(self, value: int) -> str:
    return str(value) + self.salt


@enact.typed_invokable(int, str)
class WrongOutputType(enact.Invokable):

  def call(self, value: int) -> int:
    return value


@dataclasses.dataclass
@enact.typed_invokable(int, int)
class AddOne(enact.Invokable):
  fail: bool = False
  def call(self, input_resource: int) -> int:
    if self.fail:
      raise ValueError('fail')
    return input_resource + 1


@dataclasses.dataclass
class Fail(enact.Invokable):
  def call(self, arg: enact.ResourceBase) -> int:
    raise ValueError('fail')


@enact.typed_invokable(int, int)
@dataclasses.dataclass
class NestedFunction(enact.Invokable):
  """A nested function that repeatedly calls another invokable."""
  fun: enact.InvokableBase = AddOne()
  iter: int = 10
  fail_on: Optional[int] = None

  def call(self, input_resource: int) -> int:
    for i in range(self.iter):
      if i == self.fail_on:
        input_resource = Fail()(input_resource)
      else:
        input_resource = self.fun(input_resource)
    return input_resource


class InvocationsTest(unittest.TestCase):
  """Tests invocations."""

  def setUp(self):
    # pylint: disable=consider-using-with
    self.dir = tempfile.TemporaryDirectory()
    self.backend = enact.FileBackend(self.dir.name)
    self.store = enact.Store(self.backend)

  def tearDown(self):
    self.dir.cleanup()

  def test_typed_invokable(self):
    """"Test that the decorator works as expected."""
    fun = IntToStr('salt')
    self.assertEqual(fun.get_input_type(), int)
    self.assertEqual(fun.get_output_type(), str)
    output = fun.call(1)
    # Input and output types should not show up as fields.
    self.assertEqual(list(fun.field_names()), ['salt'])
    # The following line should typecheck correctly.
    self.assertEqual(output, '1salt')

  def test_typecheck_input(self):
    fun = IntToStr('salt')
    with self.assertRaises(TypeError):
      fun('1')

  def test_typecheck_output(self):
    fun = WrongOutputType()
    with self.assertRaises(TypeError):
      fun('1')

  def test_auto_input_args(self):
    self.assertEqual(
      IntToStr()(1), '1')

  def test_auto_input_kwargs(self):
    self.assertEqual(
      IntToStr()(1), '1')

  def test_invoke_simple(self):
    with self.store:
      fun = IntToStr('salt')
      invocation = fun.invoke(
        enact.commit(1))
      want: enact.Invocation = enact.Invocation(
        request=enact.commit(
          enact.Request(enact.commit(fun),
                        enact.commit(1))),
        response=enact.commit(
          enact.Response(
            invokable=enact.commit(fun),
            output=enact.commit('1salt'),
            raised=None,
            raised_here=False,
            children=[])))
    self.assertEqual(
      invocation,
      want)

  def test_call_nested(self):
    """Test calling a nested function."""
    self.assertEqual(NestedFunction()(1), 11)

  def test_invoke_nested(self):
    """Test invoking a nested function."""
    with self.store:
      invocation = NestedFunction().invoke(
        enact.commit(1))
      output = invocation.get_output()
      self.assertEqual(output, 11)
      self.assertEqual(len(list(invocation.get_children())), 10)
      for i, child in enumerate(invocation.get_children()):
        self.assertEqual(child.request().input(), i + 1)
        output = child.get_output()
        self.assertEqual(output, i + 2)

  def test_invoke_fail(self):
    with self.store:
      invocation = NestedFunction(fail_on=3).invoke(
        enact.commit(1))
      self.assertFalse(invocation.successful())
      exception = invocation.get_raised()
      self.assertIsInstance(
        exception,
        enact.ExceptionResource)
      assert invocation.response().children
      self.assertEqual(len(invocation.response().children), 4)
      for i, child in enumerate(invocation.get_children()):
        if i < 3:
          self.assertEqual(child.request().input(), i + 1)
          output = child.get_output()
          self.assertEqual(output, i + 2)
        else:
          self.assertEqual(child.request().input(), 4)
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
        with invocations.Builder(fun, store.commit(1)) as builder:
          # Enter an unrelated context.
          nested = NestedFunction()
          inner_invocation = nested.invoke(store.commit(1))
          # The invocation should not be tracked in the builder.
          # pylint: disable=protected-access
          self.assertEqual(builder._children, None)
        # Redo the invocation outside the builder.
        outer_invocation = nested.invoke(store.commit(1))
        # Ensure that executing in the misleading context made no difference.
        self.assertEqual(outer_invocation, inner_invocation)

  def test_meta_invoke(self):
    """Tests that meta-invocations are tracked correctly."""
    @dataclasses.dataclass
    class MetaInvoke(enact.Invokable):
      invokable: enact.InvokableBase
      def call(self, input_resource: enact.ResourceBase):
        return self.invokable.invoke(enact.commit(input_resource))

    with self.store as store:
      fun = NestedFunction()
      meta_invocation = MetaInvoke(fun).invoke(store.commit(1))
      self.assertEqual(
        len(list(meta_invocation.get_children())), 0)
      invocation = meta_invocation.get_output()
      assert isinstance(invocation, enact.Invocation)
      self.assertEqual(
        len(list(invocation.get_children())), 10)

  def test_wrapped_resource(self):
    """Tests that exceptions are tracked as wrapped."""
    class PythonErrorOnInvoke(enact.Invokable):
      def call(self, unused_input: enact.ResourceBase):
        raise ValueError('foo')
    with self.store as store:
      fun = PythonErrorOnInvoke()
      invocation = fun.invoke(store.commit(5))
      self.assertIsInstance(
        invocation.get_raised(),
        enact.WrappedException)

  def test_raise_native_error(self):
    """Tests that exceptions are raised in native format."""
    class PythonErrorOnInvoke(enact.Invokable):
      def call(self, unused_input: enact.ResourceBase):
        raise ValueError('foo')
    class ExpectValueError(enact.Invokable):
      def call(self, unused_input: enact.ResourceBase):
        try:
          PythonErrorOnInvoke()(3)
        except ValueError:
          return 'Got value error'
        raise enact.ExceptionResource('Expected ValueError')

    with self.store as store:
      fun = ExpectValueError()
      invocation = fun.invoke(store.commit(5))
      self.assertEqual(
        invocation.get_output(),
        'Got value error')

  def test_raised_here(self):
    """Tests that the raised_here field is set correctly."""
    class PythonErrorOnInvoke(enact.Invokable):
      def call(self, unused_input: enact.ResourceBase):
        raise ValueError('foo')

    @dataclasses.dataclass
    class SubCall(enact.Invokable):
      invokable: enact.Ref[enact.InvokableBase]
      def call(self, input_resource: enact.ResourceBase):
        self.invokable.checkout()(input_resource)

    with self.store as store:
      error_fun = PythonErrorOnInvoke()
      subcall_1 = SubCall(store.commit(error_fun))
      subcall_2 = SubCall(store.commit(subcall_1))

      invocation = subcall_2.invoke(store.commit(5))
      self.assertFalse(invocation.get_raised_here())
      (subinvocation,) = invocation.get_children()
      self.assertFalse(subinvocation.get_raised_here())
      (subsubinvocation,) = subinvocation.get_children()
      self.assertTrue(subsubinvocation.get_raised_here())

  def test_no_reraise_in_replay(self):
    """Tests that exceptions are replayed."""
    native_errors_raised = 0
    class PythonErrorOnInvoke(enact.Invokable):
      def call(self, unused_input: enact.ResourceBase):
        nonlocal native_errors_raised
        native_errors_raised += 1
        raise ValueError('foo')

    @dataclasses.dataclass
    class SubCall(enact.Invokable):
      invokable: enact.Ref[enact.InvokableBase]
      def call(self, input_resource: enact.ResourceBase):
        self.invokable.checkout()(input_resource)

    with self.store as store:
      error_fun = PythonErrorOnInvoke()
      subcall_1 = SubCall(store.commit(error_fun))
      subcall_2 = SubCall(store.commit(subcall_1))
      invocation = subcall_2.invoke(store.commit(5))
      self.assertEqual(native_errors_raised, 1)
      with self.assertRaises(ValueError):
        with invocations.ReplayContext(subinvocations=[
            enact.commit(invocation)]):
          subcall_2(5)
      self.assertEqual(native_errors_raised, 2)


  def test_input_changed_error(self):
    @dataclasses.dataclass
    class Changeable(enact.Resource):
      x: int = 0

    class ChangesInput(enact.Invokable):
      def call(self, input_resource: Changeable):
        input_resource.x += 1
        return input_resource

    with self.assertRaises(enact.InputChanged):
      with self.store as store:
        ChangesInput().invoke(store.commit(Changeable()))

  def test_replay_call(self):
    """Test replaying a call."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(0))
      with enact.ReplayContext(
          subinvocations=[enact.commit(invocation)],
          exception_override=lambda x: 100):
        result = fun(0)
      self.assertEqual(result, 106)

  def test_replay_invocation(self):
    """Test replaying an invocation."""
    fun = NestedFunction()
    with self.store:
      invocation = fun.invoke(enact.commit(0))
      replay = invocation.replay()
      self.assertEqual(invocation, replay)

  def test_rewind_call(self):
    """Test rewinding a call."""
    leaf_calls = 0

    @enact.typed_invokable(type(None), int)
    class Leaf(enact.Invokable):
      def call(self):
        nonlocal leaf_calls
        leaf_calls += 1
        return 1

    @enact.typed_invokable(type(None), int)
    class Nested(enact.Invokable):
      def call(self):
        leaf = Leaf()
        return leaf() + leaf() + leaf()

    fun = Nested()
    with self.store:
      invocation = fun.invoke()
      self.assertEqual(leaf_calls, 3)
      invocation.replay()
      self.assertEqual(leaf_calls, 3)
      invocation = invocation.rewind(2)
      invocation.replay()
      self.assertEqual(leaf_calls, 5)

  def test_replay_modifies_invokable(self):
    @enact.register
    @dataclasses.dataclass
    class Counter(enact.Invokable):
      call_count: int = 0
      def call(self, input_resource: enact.ResourceBase):
        self.call_count += 1
        return input_resource

    with self.store:
      counter = Counter()
      fun = NestedFunction(counter, iter=10)
      invocation = fun.invoke(
        enact.commit(0))
      self.assertEqual(counter.call_count, 10)

      # Modify output to ensure we got a replay and
      # not a reexecution.
      with invocation.response.modify() as response:
        response.output = enact.commit(100)

      fun.fun = Counter()
      result = invocation.replay()
      self.assertEqual(result.get_output(), 100)
      self.assertEqual(counter.call_count, 10)

  def test_replay_partial(self):
    """Test replaying a partial invocation."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(0))
      with invocation.response.modify() as response:
        del response.children[4:]
        with response.children[-1].modify() as child:
          with child.response.modify() as child_response:
            assert child_response.raised
            child_response.raised = None
            child_response.output = enact.commit(100)
      with enact.ReplayContext(
          subinvocations=[enact.commit(invocation)]):
        result = fun(0)
      self.assertEqual(result, 106)

  def test_replay_call_on_mismatch_nonstrict(self):
    """Test non-strict replays are ignored if arguments don't match."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(0))
      with enact.ReplayContext(
          subinvocations=[enact.commit(invocation)],
          exception_override=lambda x: 100, strict=False):
        with self.assertRaises(ValueError):
          # Exception override is not active since we're ignoring the
          # replay.
          fun(1)

  def test_replay_call_on_mismatch_strict(self):
    """Test strict replays raise."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(0))
      with enact.ReplayContext(
          subinvocations=[enact.commit(invocation)],
          exception_override=lambda x: 100, strict=True):
        with self.assertRaises(enact.ReplayError):
          fun(1)

  def test_invoke_with_replay(self):
    """Test replays are ignored if arguments don't match."""
    fun = NestedFunction(fail_on=3)
    with self.store:
      invocation = fun.invoke(
        enact.commit(0))
      invocation = fun.invoke(
        enact.commit(0),
        replay_from=invocation,
        exception_override=lambda x: 100)
      self.assertEqual(invocation.get_output(), 106)

  def test_request_input(self):
    """Tests the RequestInput invokable."""
    fun = NestedFunction(fun=enact.RequestInput(str), iter=2)
    with self.store:
      inputs = ['foo', 'bar', 'bish']
      invocation = fun.invoke(enact.commit(inputs[0]))
      for cur_input, next_input in zip(inputs[:-1], inputs[1:]):
        raised = invocation.get_raised()
        assert isinstance(raised, enact.InputRequest)
        self.assertEqual(
          cast(str, raised.for_value.checkout()), cur_input)
        invocation = raised.continue_invocation(
          invocation, next_input)
      self.assertEqual(invocation.get_output(), 'bish')
      child_outputs = [
        c.get_output() for c in invocation.get_children()]
      self.assertEqual(child_outputs, inputs[1:])

  def test_request_input_fun(self):
    """Tests the request_input function."""
    class MyInvokable(enact.Invokable):
      def call(self, value: int):
        return enact.request_input(int) + value
    with self.store:
      invocation = MyInvokable().invoke(enact.commit(3))
      raised = invocation.get_raised()
      assert isinstance(raised, enact.InputRequest)
      invocation = raised.continue_invocation(invocation, 5)
      self.assertEqual(invocation.get_output(), 8)

  def test_input_requested_outside_invocation(self):
    with self.assertRaises(enact.InputRequestOutsideInvocation):
      enact.RequestInput(str)('foo')

  def test_request_input_class(self):
    with self.store:
      fun = enact.RequestInput(str, 'Context')
      invocation = fun.invoke(enact.commit('foo'))
      raised = invocation.get_raised()
      assert isinstance(raised, enact.InputRequest)
      assert raised.context == 'Context'
      self.assertEqual(raised.requested_type, str)

  def test_empty_call_args(self):
    """Tests that empty call args are ok for None inputs."""
    @enact.typed_invokable(type(None), str)
    class TextInput(enact.Invokable):
      def call(self):
        return 'Foo'

    with self.store:
      fun = TextInput()
      invocation = fun.invoke()
      self.assertEqual(invocation.get_output(), 'Foo')

  def test_invokable_generator_send_without_next(self):
    """Tests that send without next fails on invokable generator."""
    with self.store:
      inv_gen = enact.InvocationGenerator(
        IntToStr(), enact.commit(3))
      with self.assertRaisesRegex(TypeError, '.*non-None.*'):
        inv_gen.send(3)

  def test_invokable_generator_send_none_without_next(self):
    """Tests that send None without next works."""
    with self.store:
      inv_gen = enact.InvocationGenerator(
        IntToStr(), enact.commit(3))
      with self.assertRaises(StopIteration):
        inv_gen.send(None)

  def test_invokable_generator_send_flow(self):
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

  def test_invokable_generator_set_input_flow(self):
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


@enact.register
@dataclasses.dataclass
class AsyncWrapper(enact.AsyncInvokable):
  """An async wrapper for a sync invokable."""
  fun: enact._InvokableBase
  async def call(self, *args):
    fun = self.fun
    # pylint: disable=not-callable
    if isinstance(fun, enact.InvokableBase):
      return fun(*args)
    if isinstance(fun, enact.AsyncInvokable):
      return await fun(*args)

@enact.typed_invokable(float, float)
@dataclasses.dataclass
class Sleep(enact.AsyncInvokable):
  """Calls asyncio.sleep and returns actual time slept."""
  async def call(self, value: float) -> float:
    before = time.time()
    await asyncio.sleep(value)
    after = time.time()
    return float(after - before)


@enact.register
@dataclasses.dataclass
class AwaitInTask(enact.AsyncInvokable):
  """Launches another invokable in a task and possibly awaits it."""
  fun: enact.AsyncInvokable

  async def call(self, value: enact.ResourceBase) -> enact.ResourceBase:
    return await asyncio.get_running_loop().create_task(self.fun(value))


@enact.register
@dataclasses.dataclass
class RunawayTasks(enact.AsyncInvokable):
  """Invalid invokable that returns before all tasks are complete."""

  async def call(self, unused_value: enact.ResourceBase):
    sleep = Sleep()
    sleeps = [sleep(i * 0.1) for i in range(10)]
    await asyncio.wait(sleeps, timeout=0.2)

@enact.typed_invokable(type(None), int)
class AsyncRollDie(enact.AsyncInvokable):
  async def call(self):
    await asyncio.sleep(0.1 * random.random())
    return random.randint(1, 6)


@enact.typed_invokable(int, int)
class AsyncRollConcurrentDice(enact.AsyncInvokable):
  async def call(self, n: int):
    result = await asyncio.gather(
      *[AsyncRollDie()() for _ in range(n)])
    return sum(result)


@enact.typed_invokable(type(None), type(None))
class AsyncFail(enact.AsyncInvokable):
  async def call(self):
    await asyncio.sleep(0.01)
    raise ValueError('AsyncFail')


@enact.typed_invokable(type(None), list)
@dataclasses.dataclass
class Gather(enact.AsyncInvokable):
  """Calls asyncio.gather on a list of invokables."""
  invokables: list

  async def call(self):
    results = await asyncio.gather(
      *[i(None) for i in self.invokables],
      return_exceptions=True)
    for r in results:
      if isinstance(r, Exception):
        raise r
    return list(results)


class AsyncInvocationsTest(unittest.TestCase):
  """Tests invocations."""

  def setUp(self):
    # pylint: disable=consider-using-with
    self.dir = tempfile.TemporaryDirectory()
    self.backend = enact.FileBackend(self.dir.name)
    self.store = enact.Store(self.backend)

  def test_call(self):
    """Tests calling an async invokable directly."""
    fun = AsyncWrapper(IntToStr())
    result = asyncio.run(fun(3))
    self.assertEqual(result, '3')

  def test_invoke(self):
    """Tests basic invocations."""
    fun = AsyncWrapper(IntToStr())
    with self.store:
      result = asyncio.run(fun.invoke(enact.commit(3)))
      self.assertEqual(
        result,
        enact.Invocation(
          enact.commit(enact.Request(
            enact.commit(fun),
            enact.commit(3))),
          enact.commit(enact.Response(
            enact.commit(fun),
            enact.commit('3'),
            None,
            False,
            [
              enact.commit(enact.Invocation(
                enact.commit(enact.Request(
                  enact.commit(IntToStr()),
                  enact.commit(3))),
                enact.commit(enact.Response(
                  enact.commit(IntToStr()),
                  enact.commit('3'),
                  None,
                  False,
                  []))
              ))
            ]),
          ))
        )

  def test_invoke_nested_async(self):
    """Tests basic invocations od nested async invokables."""
    fun = AsyncWrapper(AsyncWrapper(IntToStr()))
    with self.store:
      invocation = asyncio.run(fun.invoke(
        enact.commit(3)))
      self.assertEqual(invocation.get_output(), '3')

  def test_invoke_async_with_task(self):
    """Tests that created tasks are correctly tracked."""
    fun = AwaitInTask(Sleep())
    with self.store:
      invocation = asyncio.run(fun.invoke(
        enact.commit(0.01)))
      self.assertGreater(invocation.get_output(), 0.0)

  def test_invoke_async_with_runaway_task(self):
    """Tests that runaway tasks raise an error."""
    fun = RunawayTasks()
    with self.store:
      with self.assertRaises(enact.IncompleteSubinvocationError):
        asyncio.run(fun.invoke())

  def test_simple_replay_async(self):
    fun = AsyncRollConcurrentDice()
    with self.store:
      invocation = asyncio.run(fun.invoke(enact.commit(20)))
      replay = asyncio.run(invocation.replay_async())
      self.assertEqual(invocation, replay)

  def test_replay_async(self):
    """Tests async replays."""
    fun = AsyncRollConcurrentDice()
    with self.store:
      invocation = asyncio.run(fun.invoke(enact.commit(20)))
      rolls = [c.get_output() for c in invocation.get_children()]
      self.assertEqual(len(rolls), 20)

      # Delete 10 outputs in the middle.
      with invocation.response.modify() as response:
        response.output = None
        for i in range(5, 15):
          with response.children[i].modify() as child:
            child.clear_output()

      invocation = asyncio.run(invocation.replay_async())
      rerolls = [c.get_output() for c in invocation.get_children()]
      self.assertEqual(len(rerolls), 20)
      # Check that the replays are the same as the original rolls.

  def test_replay_async_call(self):
    """Tests async replays using calls."""
    fun = AsyncRollConcurrentDice()
    with self.store:
      invocation = asyncio.run(fun.invoke(enact.commit(20)))
      with enact.ReplayContext([enact.commit(invocation)]):
        result = asyncio.run(fun(20))
      self.assertEqual(invocation.get_output(), result)

  def test_replay_async_multiple_exceptions(self):
    """Tests async replays that collect multiple exceptions."""
    fun = Gather([AsyncFail() for _ in range(10)])
    with self.store:
      invocation = asyncio.run(fun.invoke())
      exceptions = [c.get_raised() for c in invocation.get_children()]
      self.assertEqual(len(exceptions), 10)

  def test_continue_invocation_async(self):
    """Tests that input requests can be continued."""
    # Request ten integers.
    fun = Gather([AsyncWrapper(enact.RequestInput(int))
                  for _ in range(10)])
    with self.store:
      invocation = asyncio.run(fun.invoke(enact.commit(None)))
      input_request = invocation.get_raised()
      assert isinstance(input_request, enact.InputRequest)
      invocation = asyncio.run(input_request.continue_invocation_async(
        invocation, 2))
      self.assertEqual(
        invocation.get_output(),
        [2 for _ in range(10)])

  def test_continue_invocation_async_targeted(self):
    """Tests that input requests can be continued with different values."""
    fun = Gather([AsyncWrapper(enact.RequestInput(int, i))
                  for i in range(10)])
    with self.store:
      invocation = asyncio.run(fun.invoke(enact.commit(None)))
      self.assertIsInstance(invocation.get_raised(), enact.InputRequest)
      def exception_override(exception_ref):
        input_request = exception_ref()
        assert isinstance(input_request, enact.InputRequest)
        context = cast(int, input_request.context)
        return context + 1
      invocation = asyncio.run(invocation.replay_async(exception_override))
      self.assertEqual(invocation.get_output(), list(range(1, 11)))

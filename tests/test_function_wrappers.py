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

"""Tests for function wrappers."""

import asyncio
import dataclasses
import random
from typing import Callable, Type
import unittest

import enact
from enact import invocations


@enact.register
def python_to_python(i: int) -> str:
  """A python function that returns a string."""
  return str(i)

@enact.register
async def async_python_to_python(i: int) -> str:
  """An async python function that returns a string."""
  return str(i)

@enact.register
@dataclasses.dataclass
class ResourceClassWithMembers(enact.Resource):
  """An enact resource with member functions."""
  calls: int = 0

  @enact.register
  def foo(self, x: int):
    self.calls += 1
    return str(x)

  @enact.register
  async def foo_async(self, x: int):
    self.calls += 1
    return str(x)

  @enact.register
  def bar(self) -> int:
    self.foo(2)
    asyncio.run(self.foo_async(3))
    return self.calls


@enact.register
class MyInvokable(enact.Invokable[int, str]):
  """An old-school invokable."""
  def call(self, value: int) -> str:
    return str(value)


@enact.register
class MyAsyncInvokable(enact.AsyncInvokable[int, str]):
  """An old-school async invokable."""
  async def call(self, value: int) -> str:
    return str(value)


class WrappedClassWithMember:
  """A class that uses wrappers and has a registered member."""
  def __init__(self, x: int):
    self.x = x

  @enact.register
  def foo(self, x: int):
    self.x += 1
    return str(x + self.x)

  @enact.register
  def bar(self):
    self.foo(1)
    self.foo(2)
    return self.x

  @enact.register
  async def async_bar(self):
    self.foo(1)
    self.foo(2)
    return self.x


@enact.register
@dataclasses.dataclass
class WrappedClassWrapper(enact.TypeWrapper[WrappedClassWithMember]):
  """A wrapper for WrappedClass."""
  x: int
  @classmethod
  def wrapped_type(cls) -> Type[WrappedClassWithMember]:
    return WrappedClassWithMember

  @classmethod
  def wrap(cls, value: WrappedClassWithMember) -> 'WrappedClassWrapper':
    return cls(value.x)

  def unwrap(self) -> WrappedClassWithMember:
    return WrappedClassWithMember(self.x)

  @classmethod
  def set_wrapped_value(
    cls, target: WrappedClassWithMember, src: WrappedClassWithMember):
    target.x = src.x


class FunctionWrappersTest(unittest.TestCase):
  """Wrap a python function."""

  def setUp(self):
    self.store = enact.InMemoryStore()

  def test_python_to_python(self):
    """Test wrapping a python function."""
    self.assertEqual(python_to_python(3), '3')
    with self.store:
      invocation = enact.invoke(python_to_python, (3,))
      self.assertEqual(invocation.get_output(), '3')
      self.assertEqual(invocation.request().invokable(), python_to_python)

  def test_async_python_to_python(self):
    """Test wrapping an async python function."""
    self.assertEqual(asyncio.run(async_python_to_python(3)), '3')
    with self.store:
      invocation = asyncio.run(enact.invoke_async(async_python_to_python, (3,)))
      self.assertEqual(invocation.get_output(), '3')
      self.assertEqual(invocation.request().invokable(), async_python_to_python)

  def test_invoke_unregistered(self):
    """Test invoking an unregistered function."""
    def unregistered(i: int) -> str:
      return str(i)
    with self.store:
      invocation = enact.invoke(unregistered, (3,))
      self.assertEqual(invocation.get_output(), '3')
      self.assertEqual(list(invocation.get_children()), [])

  def test_invoke_unregistered_async(self):
    """Test invoking an async unregistered function."""
    async def unregistered(i: int) -> str:
      return str(i)
    with self.store:
      invocation = asyncio.run(enact.invoke_async(unregistered, (3,)))
      self.assertEqual(invocation.get_output(), '3')
      self.assertEqual(list(invocation.get_children()), [])

  def test_invoke_unregistered_calls_registered(self):
    """Test an unregistered function that calls a registered function."""
    def unregistered(i: int) -> str:
      return python_to_python(i)
    with self.store:
      invocation = enact.invoke(unregistered, (3,))
      self.assertEqual(invocation.get_output(), '3')
      (child,) = invocation.get_children()
      self.assertEqual(child.request().invokable(), python_to_python)

  def test_invoke_unregistered_calls_registered_async(self):
    """Test an unregistered async_function that calls a registered function."""
    async def unregistered(i: int) -> str:
      return await async_python_to_python(i)
    with self.store:
      invocation = asyncio.run(enact.invoke_async(unregistered, (3,)))
      self.assertEqual(invocation.get_output(), '3')
      (child,) = invocation.get_children()
      self.assertEqual(child.request().invokable(), async_python_to_python)

  def test_invoke_invokable(self):
    """Tests invoking a standard invokable."""
    with self.store:
      self.assertEqual(
        enact.invoke(MyInvokable(), (3,)).get_output(), '3')

  def test_invoke_async_invokable(self):
    """Tests invoking an async invokable."""
    with self.store:
      self.assertEqual(
        asyncio.run(
          enact.invoke_async(MyAsyncInvokable(), (3,))).get_output(), '3')

  def test_replay_fails_on_unregistered(self):
    """Tests that replay will fail on an unregistered function."""
    def unregistered(i: int) -> str:
      return python_to_python(i)
    with self.store:
      invocation = enact.invoke(unregistered, (3,))
      #with self.assertRaises(invocations.InvocationError):
      invocation = invocation.rewind()
      with self.assertRaises(invocations.InvocationError):
        invocation.replay()

  def test_replay_succeeds_on_registered(self):
    """Tests that replay will succeed on a registered function."""
    @enact.register
    def rand():
      return random.randint(0, 10000000)
    @enact.register
    def randsum():
      return rand() + rand() + rand()

    with self.store:
      invocation = enact.invoke(randsum)
      self.assertGreater(invocation.get_output(), 0)
      replay = invocation.rewind().replay()
      for i in (0, 1):
        self.assertEqual(invocation.get_child(i).get_output(),
                         replay.get_child(i).get_output())
      self.assertNotEqual(invocation.get_child(2).get_output(),
                          replay.get_child(2).get_output())


  def test_member_function(self):
    """Tests that member functions can be wrapped."""
    @enact.register
    @dataclasses.dataclass
    class MyClass(enact.Resource):
      """Class with sync and async members."""
      calls: int = 0

      @enact.register
      def foo(self, x: int):
        self.calls += 1
        return str(x)

      @enact.register
      async def bar(self, x: int):
        self.calls += 1
        return str(x)

    with self.store:
      instance = MyClass()
      invocation = enact.invoke(instance.foo, (3,))
      self.assertEqual(invocation.get_output(), '3')
      self.assertEqual(instance.calls, 1)
      invocation = asyncio.run(
        enact.invoke_async(instance.bar, (3,)))
      self.assertEqual(invocation.get_output(), '3')
      self.assertEqual(instance.calls, 2)

  def test_member_replay(self):
    """Tests that updates to self are correctly replayed."""
    with self.store:
      instance = ResourceClassWithMembers()
      invocation = enact.invoke(instance.bar)
      self.assertEqual(invocation.get_output(), 2)
      invocation = invocation.rewind()
      invocation = invocation.replay()
      self.assertEqual(invocation.get_output(), 2)
      self.assertEqual(invocation.request().invokable().__self__.calls, 0)

  def test_member_of_non_enact_class_fails(self):
    """Tests that registering a member function of a non-enact class fails."""
    class UnwrappedClass:
      def __init__(self, x: int):
        self.x = x

      @enact.register
      def foo(self, x: int):
        return str(x + self.x)

    with self.store:
      instance = UnwrappedClass(3)
      self.assertEqual(instance.foo(5), '8')

      with self.assertRaises(enact.FieldTypeError) as e:
        enact.invoke(instance.foo, (5,))
      self.assertIn('Please register a TypeWrapper', str(e.exception))

  def test_member_of_wrapped_class_replay_succeeds(self):
    """Tests that wrapped classes can be replayed."""
    with self.store:
      instance = WrappedClassWithMember(0)
      invocation = enact.invoke(instance.bar)
      self.assertEqual(invocation.get_output(), 2)
      self.assertEqual(invocation.rewind().replay().get_output(), 2)

  def test_invocations_tracked(self):
    """Test various forms of tracking."""

    @enact.register
    def registered_one(inv: MyInvokable):
      rc = ResourceClassWithMembers()
      return inv(rc.bar() + rc.bar())

    @enact.register
    async def registered_two(inv: MyInvokable):
      wc = WrappedClassWithMember(2)
      return inv(wc.bar() + wc.bar() - 1)

    def unregistered(inv):
      return registered_one(inv) + asyncio.run(registered_two(inv))

    inv = MyInvokable()
    with self.store:
      invocation = enact.invoke(unregistered, (), {'inv': inv})
      self.assertEqual(invocation.get_output(), '69')

      # registered_one
      self.assertEqual(invocation.get_child(0).get_child(0).get_output(), 2)
      self.assertEqual(invocation.get_child(0).get_child(1).get_output(), 4)
      # str(2 + 4)
      self.assertEqual(invocation.get_child(0).get_child(2).get_output(), '6')

      # registered_two
      self.assertEqual(invocation.get_child(1).get_child(0).get_output(), 4)
      self.assertEqual(invocation.get_child(1).get_child(1).get_output(), 6)
      # str(4 + 6 - 1)
      self.assertEqual(invocation.get_child(1).get_child(2).get_output(), '9')

  def test_register_classmethod_fails(self):
    """Tests that classmethods can't be tracked (for now)."""
    with self.assertRaises(TypeError):
      class MyClass:  # pylint: disable=unused-variable
        @enact.register
        @classmethod
        def foo(cls, x: int):
          return str(x)

  def test_callable_argument(self):
    """Tests that registered functions can be arguments to other functions."""
    bar_suffix = 'bla'

    @enact.register
    def foo(fun):
      return fun() + fun()

    @enact.register
    def bar():
      return 'bar' + bar_suffix

    self.assertEqual(foo(bar), 'barblabarbla')
    with self.store:
      invocation = enact.invoke(foo, (bar,))
      self.assertEqual(invocation.get_output(), 'barblabarbla')
      self.assertEqual(invocation.get_child(0).get_output(), 'barbla')
      self.assertEqual(invocation.get_child(1).get_output(), 'barbla')

      bar_suffix = 'blo'
      invocation = invocation.rewind().replay()
      self.assertEqual(invocation.get_output(), 'barblabarblo')
      self.assertEqual(invocation.get_child(0).get_output(), 'barbla')
      self.assertEqual(invocation.get_child(1).get_output(), 'barblo')

  def test_member_argument(self):
    """Tests that member functions can be function arguments."""
    @enact.register
    @dataclasses.dataclass
    class MyClass(enact.Resource):
      x: int

      @enact.register
      def foo(self):
        return self.x

    @enact.register
    def bar(fun):
      return fun() + fun()

    m = MyClass(2)
    self.assertEqual(bar(m.foo), 4)

    with self.store:
      invocation = enact.invoke(bar, (m.foo,))
      self.assertEqual(invocation.get_output(), 4)
      self.assertEqual(
        invocation.request().input().args[0].__self__,
        MyClass(2))
      self.assertEqual(
        invocation.request().input().args[0].__func__,
        MyClass.foo)

  def test_assign_function_as_member_variable(self):
    """Functions assigned as members of resources can be called correctly."""

    @enact.register
    def foo(x: int) -> int:
      return x + 1

    @enact.register
    @dataclasses.dataclass
    class CallableMember(enact.Resource):
      fun: Callable[[int], int] = foo

    instance = CallableMember()
    self.assertEqual(instance.fun(3), 4)

    with self.store:
      invocation = enact.invoke(instance.fun, (3,))
      self.assertEqual(invocation.get_output(), 4)

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

"""Tests for the context module."""

import unittest
import threading
from typing import Any, Optional

from enact import contexts


@contexts.register
class SimpleContext(contexts.Context):
  """A context object for testing."""
  permissive_init: bool = True

  @classmethod
  def permissive_initialization(cls) -> bool:
    return cls.permissive_init

  def __init__(self):
    super().__init__()
    self.depth: Optional[int] = None

  def enter(self):
    cur = SimpleContext.get_current()
    if cur:
      assert cur.depth is not None
      self.depth = cur.depth + 1
    else:
      self.depth = 1

  def exit(self):
    self.depth = None


class UnregisteredContext(SimpleContext):
  """A context that was not registered."""


@contexts.register_to_superclass(SimpleContext)
class DerivedContextA(SimpleContext):
  """A derived context from SimpleContext."""


@contexts.register_to_superclass(SimpleContext)
class DerivedContextB(SimpleContext):
  """Another derived context from SimpleContext."""


class UnregisteredDerivedContext(SimpleContext):
  """A derived context that was not registered."""


class ContextTest(unittest.TestCase):
  """Test cases for the context class."""

  def test_current(self):
    """Tests that the current function works."""
    self.assertIs(SimpleContext.get_current(), None)
    with SimpleContext() as s1:
      self.assertEqual(SimpleContext.get_current(), s1)
      self.assertEqual(s1.depth, 1)
      with SimpleContext() as s2:
        self.assertEqual(SimpleContext.get_current(), s2)
        self.assertEqual(s2.depth, 2)
      self.assertEqual(SimpleContext.get_current(), s1)
    self.assertIs(SimpleContext.get_current(), None)

  def test_unregistered_fails(self):
    """Tests that using an unregistered context fails."""
    with self.assertRaises(contexts.ContextError):
      UnregisteredContext.get_current()

  def test_top_level(self):
    """Tests that top_level works as expected."""
    with SimpleContext() as s1:
      self.assertEqual(SimpleContext.current(), s1)
      with SimpleContext.top_level():
        self.assertEqual(SimpleContext.get_current(), None)
      self.assertEqual(SimpleContext.current(), s1)

  def test_derived_context(self):
    """Tests that derived contexts work as expected."""
    with DerivedContextA() as a:
      self.assertEqual(SimpleContext.get_current(), a)
      self.assertEqual(DerivedContextA.get_current(), a)

  def test_derived_context_different_subclass(self):
    """Tests that sibling derived contexts work as expected."""
    with DerivedContextA():
      with self.assertRaisesRegex(contexts.ContextError, 'not of type'):
        DerivedContextB.get_current()

  def test_derived_context_unregistered(self):
    """Tests that unregistered derived contexts fail."""
    with self.assertRaisesRegex(contexts.ContextError, 'not registered'):
      with UnregisteredDerivedContext():
        pass

  def test_register_to_superclass_unregistered_superclass(self):
    """Tests registering to an unregistered superclass."""
    class BaseContext(contexts.Context):
      pass

    class DerivedContext(BaseContext):
      pass

    with self.assertRaisesRegex(AssertionError, 'Superclass .* not registered'):
      contexts.register_to_superclass(BaseContext)(DerivedContext)

  def test_register_to_superclass_not_subclass(self):
    """Tests registering to a non-parent class."""
    @contexts.register
    class BaseContext(contexts.Context):
      pass

    class NonDerivedContext(contexts.Context):
      pass

    with self.assertRaisesRegex(AssertionError, 'must be a subclass'):
      contexts.register_to_superclass(BaseContext)(NonDerivedContext)

  def test_register_to_superclass_twice(self):
    """Tests registering to superclass twice."""
    @contexts.register
    class BaseContext(contexts.Context):
      pass

    class DerivedContext(BaseContext):
      pass

    contexts.register_to_superclass(BaseContext)(DerivedContext)
    with self.assertRaisesRegex(AssertionError, 'already registered'):
      contexts.register_to_superclass(BaseContext)(DerivedContext)

  def test_register_to_superclass_seperately_registered(self):
    """Tests registering to superclass after regular registration."""
    @contexts.register
    class BaseContext(contexts.Context):
      pass

    @contexts.register
    class DerivedContext(BaseContext):
      pass

    with self.assertRaisesRegex(AssertionError, 'already registered'):
      contexts.register_to_superclass(BaseContext)(DerivedContext)


class ContextDecoratorTest(unittest.TestCase):
  """Tests the context decorators."""

  def setUp(self):
    self.thread_result: Any = None
    self.context = SimpleContext()

  def enter_context(self):
    try:
      with self.context:
        self.thread_result = self.context.depth
    except contexts.ContextError as e:
      self.thread_result = e

  def test_threading_fails_if_not_permissive(self):
    """Tests that threading without a decorator fails."""
    SimpleContext.permissive_init = False
    t = threading.Thread(target=self.enter_context)
    t.start()
    t.join()
    self.assertIsInstance(self.thread_result, contexts.ContextError)

  def test_threading_succeeds_if_permissive(self):
    """Tests that threading without a decorator fails."""
    SimpleContext.permissive_init = True
    t = threading.Thread(target=self.enter_context)
    t.start()
    t.join()
    self.assertEqual(self.thread_result, 1)

  def test_with_new_contexts(self):
    """Tests that running with new contexts works."""
    with SimpleContext():
      t = threading.Thread(
        target=contexts.with_new_contexts(self.enter_context))
      t.start()
      t.join()
    self.assertEqual(self.thread_result, 1)

  def test_with_current_contexts(self):
    """Tests that running with the current contexts works."""
    with SimpleContext():
      t = threading.Thread(
        target=contexts.with_current_contexts(self.enter_context))
      t.start()
      t.join()
    self.assertEqual(self.thread_result, 2)

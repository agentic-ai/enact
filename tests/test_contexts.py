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

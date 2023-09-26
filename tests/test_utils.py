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
"""Tests for utils."""

import dataclasses
import unittest
from unittest import mock

import enact
from enact import utils

class UtilsTest(unittest.TestCase):
  """Tests for utils."""

  @enact.register
  @dataclasses.dataclass
  class _TestResource(enact.Resource):
    """A test resource."""
    foo: int

    def _get_foo(self):
      return self.foo

    @utils.cached_property
    def value(self):
      return self._get_foo()

  def test_cached_property(self):
    """Tests that the cached property works as expected."""
    test_resource = UtilsTest._TestResource(0)

    def mock_get_foo() -> mock.Mock:
      # pylint: disable=protected-access
      return mock.Mock(
        spec=test_resource._get_foo, side_effect=test_resource._get_foo)

    mock_getter = mock_get_foo()
    with mock.patch.object(test_resource, '_get_foo', mock_getter):
      self.assertEqual(test_resource.value, 0)
      self.assertTrue(mock_getter.called)

    mock_getter = mock_get_foo()
    with mock.patch.object(test_resource, '_get_foo', mock_getter):
      # Should not be called again as the digest of the resource hasn't changed.
      self.assertEqual(test_resource.value, 0)
      self.assertFalse(mock_getter.called)

    # This should change the digest.
    test_resource.foo = 1
    mock_getter = mock_get_foo()
    with mock.patch.object(test_resource, '_get_foo', mock_getter):
      # Should be called again as the digest of the resource has changed.
      self.assertEqual(test_resource.value, 1)
      self.assertTrue(mock_getter.called)

    # This should change the digest back to the original value, but the cache
    # will have been cleared, so the getter will be called again.
    test_resource.foo = 0
    mock_getter = mock_get_foo()
    with mock.patch.object(test_resource, '_get_foo', mock_getter):
      self.assertEqual(test_resource.value, 0)
      self.assertTrue(mock_getter.called)

if __name__ == '__main__':
  unittest.main()

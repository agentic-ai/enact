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

"""Tests for digests."""

import unittest

from enact import digests
from enact import interfaces

import tests.random_value as random_value


class DigestTest(unittest.TestCase):
  """Tests for digests."""

  def test_equal_digest_resource_and_dict(self):
    """Tests that resources and their dicts have the same digest."""
    for _ in range(100):
      resource = random_value.rand_resource()
      self.assertEqual(digests.digest(resource),
                       digests.digest(resource.to_resource_dict()))

  def test_none_digest(self):
    """Ensures that none has the same digest as dict or resource."""
    none = interfaces.NoneResource()
    resource_digest = digests.digest(none)
    dict_digest = digests.digest(none.to_resource_dict())
    self.assertEqual(resource_digest, dict_digest)

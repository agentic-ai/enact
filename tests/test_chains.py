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

"""Tests for the chains module."""

import dataclasses
import unittest

import enact

@enact.register
@dataclasses.dataclass
class Int(enact.Resource):
  """A resource that holds an integer."""
  value: int


@enact.typed_invokable(Int, Int)
@dataclasses.dataclass
class Sum(enact.Invokable[Int, Int]):
  """Computing a running sum."""
  total: int = 0

  def call(self, arg: Int) -> Int:
    self.total += arg.value
    return Int(self.total)


class ChainsTest(unittest.TestCase):
  """Test invocation chains."""

  def test_chain(self):
    """Test invocation chains."""
    with enact.Store():
      chain = enact.InvocationChain.start(
          Sum(), Int(1))
      chain = chain.step(Int(2))
      chain = chain.step(Int(3))
      self.assertEqual(len(chain.block), 3)
      self.assertEqual(chain.block[-1].get().get_output().value, 6)
      new_chain = chain(Int(4))
      self.assertEqual(new_chain.block[-1].get().get_output().value, 10)
      self.assertEqual(
        new_chain.prev,
        enact.commit(chain))

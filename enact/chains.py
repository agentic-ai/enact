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

"""Support for coroutine-style invocation chains."""

import dataclasses
from typing import Optional, List, TypeVar
from enact import interfaces
from enact import invocations
from enact import references
from enact import resource_registry


I_contra = TypeVar('I_contra', contravariant=True, bound=interfaces.ResourceBase)
O_co = TypeVar('O_co', covariant=True, bound=interfaces.ResourceBase)


@resource_registry.register
@dataclasses.dataclass
class InvocationChain(
    invocations.Invokable[I_contra,
                          'InvocationChain[I_contra, O_co]']):
  prev: Optional[references.Ref['InvocationChain[I_contra, O_co]']]
  block: List[references.Ref[invocations.Invocation[I_contra, O_co]]]

  @classmethod
  def start(
      cls,
      invokable: invocations.Invokable[I_contra, O_co],
      input: I_contra):
    """Starts a new invocation chain."""
    return cls(
      None,
      [references.commit(invokable.invoke(references.commit(input)))])

  @property
  def invokable(self) -> invocations.InvokableBase[I_contra, O_co]:
    """Returns a reference to the latest state of the invokable."""
    response = self.block[-1].get().response
    assert response
    return response.get().invokable.get()

  def step(self, arg: I_contra) -> 'InvocationChain[I_contra, O_co]':
    """Extend the current block and return self."""
    self.block.append(references.commit(self._next_invocation(arg)))
    return self

  def _next_invocation(self, arg: I_contra) -> invocations.Invocation[
      I_contra, O_co]:
    """Compute the next invocation in the chain."""
    return self.invokable.invoke(references.commit(arg))

  def call(self, arg: I_contra) -> 'InvocationChain[I_contra, O_co]':
    """Commit the current chain link and return the next one."""
    return InvocationChain(
      references.commit(self),
      [references.commit(self._next_invocation(arg))])

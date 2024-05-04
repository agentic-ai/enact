# Copyright 2024 Agentic.AI Corporation.
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
"""Context to guard against cyclic datastructures."""

from typing import List, Optional
from enact import contexts


class CycleDetected(Exception):
  """Exception raised when a cycle is detected."""


@contexts.register
class AcyclicContext(contexts.Context):
  """Helper to safeguard against cyclic data-structures."""

  def __init__(self, obj):
    """Initializes the context."""
    super().__init__()
    self.parent: Optional[AcyclicContext] = None
    self.obj = obj

  def enter(self):
    """Check if the value is already on the stack."""
    self.parent = AcyclicContext.get_current()
    parent = self.parent
    parents: List[AcyclicContext] = []
    while parent is not None:
      parents.append(parent)
      if parent.obj is self.obj:
        raise CycleDetected(
          f'Resources may not have cyclic graph structure. '
          f'Encountered cycle: '
          f'{" -> ".join(str(p.obj) for p in parents)}')
      parent = parent.parent

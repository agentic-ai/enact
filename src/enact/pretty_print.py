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

import dataclasses
from typing import Callable, List, Mapping, Optional, Sequence, Set, Tuple, Type

from enact import interfaces
from enact import references


@dataclasses.dataclass
class PPValue:
  """Marks an indented group."""
  value: str
  contents: List['PPValue'] = dataclasses.field(default_factory=list)
  open: str = ''
  close: str = ''


class PPrinter:
  """A pretty-printer."""

  def __init__(
      self, offset: int=2,
      max_ref_depth: Optional[int]=None,
      skip_repeated_refs: bool=False):
    """Initializes the pretty-printer."""
    self._formatters: List[Tuple[
        Type, Callable[[interfaces.FieldValue, int], PPValue]]] = [
      (str, self.from_primitive),
      (references.Ref, self.from_ref),
      (interfaces.ResourceBase, self.from_resource),
      (Mapping, self.from_mapping),
      (Sequence, self.from_sequence),
      (type, self.from_type),
    ]

    self.offset = offset
    self.max_ref_depth = max_ref_depth
    self.skip_repeated_refs = skip_repeated_refs
    self._seen_refs: Set[str] = set()

  def from_type(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, type)
    assert issubclass(v, interfaces.ResourceBase), (
      'Only types that are subclasses of ResourceBase are supported.')
    return PPValue(v.__name__, [])

  def primitive_to_str(self, v: interfaces.Primitives) -> str:
    """Converts a primitive to a string."""
    if isinstance(v, bytes):
      return f'<{len(v)} bytes>'
    if isinstance(v, str):
      return repr(v)
    return str(v)

  def from_sequence(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, Sequence)
    return PPValue(
      '', [self.pvalue(item, depth + 1) for item in v], '[', ']')

  def from_mapping(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, Mapping)
    return PPValue(
      '',
      [PPValue(key, [self.pvalue(item, depth + 1)]) for key, item in v.items()],
      '{', '}')

  def from_resource(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, interfaces.ResourceBase)
    if all(isinstance(field_value, interfaces.PRIMITIVES)
           for field_value in v.field_values()):
      str_values = ', '.join([
        f'{field_name}={self.primitive_to_str(field_value)}'  # type: ignore
        for field_name, field_value in v.field_items()])
      return PPValue(f'{type(v).__name__}({str_values})', [])
    result = PPValue(
      value=type(v).__name__,
      contents=[
        PPValue(field_name, [self.pvalue(field_value, depth + 1)], open=":")
        for field_name, field_value in v.field_items()],
      open=':',
      close='')
    return result

  def from_ref(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, references.Ref)
    resource: interfaces.ResourceBase = v.get()
    if ((self.max_ref_depth and depth > self.max_ref_depth) or
        (self.skip_repeated_refs and v.digest in self._seen_refs)):
      return PPValue(f'-> {type(resource).__name__}#{v.digest[0:6]}', [])
    result = self.pvalue(resource, depth + 1)
    result.value = f'-> {result.value}#{v.digest[0:6]}'
    self._seen_refs.add(v.digest)
    return result

  def from_primitive(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, interfaces.PRIMITIVES)
    return PPValue(self.primitive_to_str(v), [])

  def pvalue(
      self, v: interfaces.FieldValue, depth: int=0) -> PPValue:
    """Produces a nested string value for pretty-printing."""
    for t, formatter in self._formatters:
      if isinstance(v, t):
        return formatter(v, depth)
    return self.from_primitive(v, depth)

  def _merge(self, v: PPValue) -> List[Tuple[int, str]]:
    """Recursively merge pvalues."""
    result: List[Tuple[int, str]] = [(0, v.value + v.open)]
    for c in v.contents:
      result += [
        (d + 1, merged_c) for d, merged_c in self._merge(c)]
    result[-1] = (result[-1][0], result[-1][1] + v.close)
    if len(result) == 2:
      result = [(result[0][0], result[0][1] + ' ' + result[1][1])]
    return result

  def register(
      self,
      t: Type,
      formatter: Callable[[interfaces.FieldValue, int], PPValue]) -> None:
    """Register a new formatter."""
    self._formatters.insert(0, (t, formatter))

  def pformat(self, v: interfaces.FieldValue) -> str:
    """Return a pretty string for a field value."""
    self._seen_refs.clear()
    pvalue = self._merge(self.pvalue(v, ))
    return '\n'.join(
      ' ' * self.offset * d + s for d, s in pvalue)

  def pprint(self, v: interfaces.FieldValue) -> None:
    """Pretty-print a field value."""
    print(self.pformat(v))


def pprint(resource: interfaces.FieldValue,
           max_ref_depth: Optional[int]=None,
           skip_repeated_refs: bool=False) -> None:
  """Pretty-prints a resource, optionally resolving references."""
  PPrinter(max_ref_depth=max_ref_depth,
           skip_repeated_refs=skip_repeated_refs).pprint(resource)


def pformat(resource: interfaces.FieldValue,
            max_ref_depth: Optional[int]=None,
            skip_repeated_refs: bool=False) -> str:
  """Pretty-prints a resource, optionally resolving references."""
  return PPrinter(
    max_ref_depth=max_ref_depth,
    skip_repeated_refs=skip_repeated_refs).pformat(resource)

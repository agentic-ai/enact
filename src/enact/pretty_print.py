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
"""Pretty-printing for resources and references."""

import dataclasses
from typing import Any, Callable, List, Optional, Set, Tuple, Type

from enact import function_wrappers
from enact import interfaces
from enact import invocations
from enact import references
from enact import resource_registry
from enact import types


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
      (bytes, self.from_primitive),
      (references.Ref, self.from_ref),
      (interfaces.ResourceBase, self.from_resource),
      (dict, self.from_dict),
      (list, self.from_list),
      (type, self.from_type),
    ]

    self.offset = offset
    self.max_ref_depth = max_ref_depth
    self.skip_repeated_refs = skip_repeated_refs
    self._seen_refs: Set[str] = set()

  def from_type(self, v: interfaces.FieldValue, unused_depth: int) -> PPValue:
    assert isinstance(v, type)
    assert issubclass(v, interfaces.ResourceBase), (
      'Only types that are subclasses of ResourceBase are supported.')
    return PPValue(v.__name__, [])

  def primitive_to_str(self, v: types.Primitives) -> str:
    """Converts a primitive to a string."""
    if isinstance(v, bytes):
      return f'<{len(v)} bytes>'
    if isinstance(v, str):
      return repr(v)
    return str(v)

  def from_list(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, list)
    return PPValue(
      '', [self.pvalue(item, depth + 1) for item in v], '[', ']')

  def from_dict(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, dict)
    return PPValue(
      '',
      [PPValue(
        f'"{key}":',
        [self.pvalue(item, depth + 1)]) for key, item in v.items()],
      '{', '}')

  def from_resource(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, interfaces.ResourceBase)
    if all(isinstance(field_value, types.PRIMITIVES)
           for field_value in v.field_values()):
      str_values = ', '.join([
        f'{field_name}={self.primitive_to_str(field_value)}'  # type: ignore
        for field_name, field_value in v.field_items()])
      return PPValue(f'{type(v).__name__}({str_values})', [])
    result = PPValue(
      value=type(v).__name__,
      contents=[
        PPValue(field_name, [self.pvalue(field_value, depth + 1)], open=':')
        for field_name, field_value in v.field_items()],
      open=':',
      close='')
    return result

  def from_ref(self, v: interfaces.FieldValue, depth: int) -> PPValue:
    assert isinstance(v, references.Ref)
    resource: interfaces.ResourceBase = v()
    if ((self.max_ref_depth and depth > self.max_ref_depth) or
        (self.skip_repeated_refs and v.digest in self._seen_refs)):
      return PPValue(f'-> {type(resource).__name__}#{v.digest[0:6]}', [])
    result = self.pvalue(resource, depth + 1)
    result.value = f'-> {result.value}#{v.digest[0:6]}'
    self._seen_refs.add(v.digest)
    return result

  def from_primitive(
      self, v: interfaces.FieldValue, unused_depth: int) -> PPValue:
    assert isinstance(v, types.PRIMITIVES)
    return PPValue(self.primitive_to_str(v), [])

  def pvalue(
      self, v: Any, depth: int=0) -> PPValue:
    """Produces a nested string value for pretty-printing."""
    field_value = resource_registry.to_field_value(v)
    for t, formatter in self._formatters:
      if isinstance(field_value, t):
        return formatter(field_value, depth)
    return self.from_primitive(field_value, depth)

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

  def pformat(self, v: Any) -> str:
    """Return a pretty string for an enact value."""
    self._seen_refs.clear()
    pvalue = self._merge(self.pvalue(v))
    return '\n'.join(
      ' ' * self.offset * d + s for d, s in pvalue)

  def pprint(self, v: Any) -> None:
    """Pretty-print an enact value."""
    print(self.pformat(v))


def pprint(value: Any,
           max_ref_depth: Optional[int]=None,
           skip_repeated_refs: bool=False) -> None:
  """Pretty-prints a resource, optionally resolving references."""
  PPrinter(max_ref_depth=max_ref_depth,
           skip_repeated_refs=skip_repeated_refs).pprint(value)


def pformat(value: Any,
            max_ref_depth: Optional[int]=None,
            skip_repeated_refs: bool=False) -> str:
  """Pretty-prints a resource, optionally resolving references."""
  return PPrinter(
    max_ref_depth=max_ref_depth,
    skip_repeated_refs=skip_repeated_refs).pformat(value)


def invocation_summary(invocation: invocations.Invocation, indent: int=0):
  """Return a reader-friendly invocation summary."""
  invokable = f'->{invocation.request().invokable()}'
  input_value = invocation.get_input()
  if isinstance(input_value, function_wrappers.CallArgs):
    args, kwargs = input_value.to_python_args()
    str_args = (
      [str(a) for a in args] +
      [f'{k}={str(v)}' for k, v in kwargs.items()])
    all_args = ', '.join(str_args)
  else:
    all_args = str(input_value)
  if invocation.successful():
    output = f'= {invocation.get_output()}'
  elif invocation.response().raised:
    output = f'raised {invocation.get_raised()}'
  else:
    output = 'incomplete'

  prefix = '  ' * indent
  lines = [
    f'{prefix}{invokable}({all_args}) {output}',
    *[f'{invocation_summary(child, indent + 1)}'
      for child in invocation.get_children()]]
  return '\n'.join(lines)

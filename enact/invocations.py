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

import abc
import dataclasses
import time
import traceback
from typing import Callable, Generic, Iterable, List, Mapping, Optional, Type, TypeVar
from enact import contexts
from enact import interfaces
from enact import references
from enact import resources
from enact import resource_registry

C = TypeVar('C', bound=interfaces.ResourceBase)
E = TypeVar('E', bound='ExceptionResource')


@resource_registry.register
class ExceptionResource(interfaces.ResourceBase, Exception):
  """A resource that is also an exception."""

  def __init__(self, *args):
    Exception.__init__(self, *args)

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    yield 'args'

  def field_values(self) -> Iterable[interfaces.FieldValue]:
    """Return a list of field values, aligned with field_names."""
    yield self.args

  @classmethod
  def from_fields(cls: Type[C],
                  field_values: Mapping[str, interfaces.FieldValue]) -> C:
    """Constructs the resource from a value dictionary."""
    return cls(*field_values['args'])


I_contra = TypeVar('I_contra', contravariant=True, bound=interfaces.ResourceBase)
O_co = TypeVar('O_co', covariant=True, bound=interfaces.ResourceBase)


@resource_registry.register
class InvocationError(ExceptionResource):
  """An error during an invocation."""


@resource_registry.register
class InvokableTypeError(InvocationError, TypeError):
  """An type error on a callable input or output."""


@resource_registry.register
class InputChangedError(InvocationError):
  """Raised when the input changes during an invocation."""


@resource_registry.register
@dataclasses.dataclass
class Request(Generic[I_contra, O_co], resources.Resource):
  """An invocation request."""
  invokable: references.Ref['InvokableBase[I_contra, O_co]']
  input: references.Ref[I_contra]


@resource_registry.register
@dataclasses.dataclass
class Response(Generic[I_contra, O_co], resources.Resource):
  """An invocation response."""
  invokable: references.Ref['InvokableBase[I_contra, O_co]']
  output: Optional[references.Ref[O_co]]
  raised: Optional[references.Ref[ExceptionResource]]
  # Subinvocations associated with this invocation.
  children: List[references.Ref['Invocation']]

  def is_complete(self) -> bool:
    """Returns whether this invocation is complete."""
    return self.raised is not None or self.output is not None


@resource_registry.register
@dataclasses.dataclass
class Invocation(Generic[I_contra, O_co], resources.Resource):
  """An invocation."""
  request: references.Ref[Request[I_contra, O_co]]
  response: references.Ref[Response[I_contra, O_co]]
  timestamp_ns: int = dataclasses.field(default_factory=lambda: Invocation.now())

  @classmethod
  def now(cls) -> int:
    """Returns the current timestamp."""
    return time.time_ns()

  def successful(self) -> bool:
    """Returns true if the invocation completed successfully."""
    if not self.response:
      return False
    return self.response.get().output is not None

  def get_output(self) -> O_co:
    """Returns the output or raise assertion error."""
    output = self.response.get().output
    assert output
    return output.get()

  def get_raised(self) -> ExceptionResource:
    """Returns the raised exception or raise assertion error."""
    raised = self.response.get().raised
    assert raised
    return raised.get()

  def get_children(self) -> Iterable['Invocation']:
    """Yields the child invocations or raises assertion error."""
    children = self.response.get().children
    for child in children:
      yield child.get()


@contexts.register
class Builder(Generic[I_contra, O_co], contexts.Context):
  """A builder for invocations."""

  def __init__(
      self,
      invokable: 'InvokableBase[I_contra, O_co]',
      input: references.Ref[I_contra]):
    """Initializes the builder."""
    self._invokable = invokable
    self._input = input
    self._request = Request(
      references.commit(invokable), input)
    self.children: List[references.Ref[Invocation]] = []

  def record_child(self, invocation: Invocation):
    """Records a subinvocation."""
    self.children.append(references.commit(invocation))

  def invocation(self) -> (
      Invocation[I_contra, O_co]):
    """Return the completed invocation."""
    parent: Optional[Builder] = Builder.get_current()
    with self:
      output: Optional[references.Ref[O_co]] = None
      exception: Optional[references.Ref[ExceptionResource]] = None
      try:
        output_resource = self._invokable.call(self._input.get())
        if output_resource is None:
          output_resource = interfaces.NoneResource()
        output = references.commit(output_resource)
      except ExceptionResource as e:
        exception = references.commit(e)
      except Exception as e:
        exception = references.commit(ExceptionResource(traceback.format_exc()))
      response = Response(
        references.commit(self._invokable), output, exception, self.children)

    invocation = Invocation(
      references.commit(self._request),
      references.commit(response))

    if parent:
      parent.record_child(invocation)
    return invocation


class InvokableBase(Generic[I_contra, O_co], interfaces.ResourceBase):
  """Base class for invokable resources."""

  _input_type: Optional[Type[I_contra]] = None
  _output_type: Optional[Type[I_contra]] = None

  @classmethod
  def get_input_type(cls) -> Optional[Type[I_contra]]:
    """Returns the type of the input if known."""
    return cls._input_type

  @classmethod
  def get_output_type(cls) -> Optional[Type[O_co]]:
    """Returns the type of the output if known."""
    return cls._output_type

  @abc.abstractmethod
  def call(self, resource: I_contra) -> O_co:
    """Executes the invokable."""

  def __call__(self, *args, **kwargs) -> O_co:
    """Executes the invokable, tracking invocation metadata."""
    input_type = self.get_input_type()
    if (len(args) != 1 or kwargs or
        not all(isinstance(arg, interfaces.ResourceBase) for arg in args) or
        not all(isinstance(arg, interfaces.ResourceBase) for arg in kwargs.values())):
      if input_type:
        # Attempts to create a resource of the input type.
        arg = input_type(*args, **kwargs)
      else:
        raise InvokableTypeError(
          'Untyped invokables must be called with a single resource argument.')
    else:
      arg = args[0]
    if input_type and not isinstance(arg, input_type):
      raise InvokableTypeError(
        f'Input type {type(arg)} does not match {input_type}.')

    parent: Optional[Builder] = Builder.get_current()
    # Execution not tracked, so just call the invokable.
    if not parent:
      return self.call(arg)

    invocation = Builder(
      self, references.commit(arg)).invocation()

    assert invocation.response
    response = invocation.response.get()

    if response.raised:
      raise response.raised.get()
    assert response.output
    output = response.output.get()
    if output is None:
      output = interfaces.NoneResource()
    output_type = self.get_output_type()
    if output_type and not isinstance(output, output_type):
      raise InvokableTypeError(
        f'Output type {type(output)} does not match {output_type}.')
    return output

  def invoke(
      self, arg: references.Ref[I_contra]) -> Invocation[I_contra, O_co]:
    """Invoke the invokable, tracking invocation metadata.

    Args:
      arg: The input resource.
      replay: An optional replay to use.
    Returns
    """
    with Builder.top_level():
      # Execute in a top-level context to ensure that there are no parents.
      return Builder(self, arg).invocation()


@dataclasses.dataclass
class Invokable(InvokableBase[I_contra, O_co], resources.Resource):
  """Base class for dataclass-based invokable resources."""


I = TypeVar('I', bound=InvokableBase)

def typed_invokable(
    input_type: Type[I_contra],
    output_type: Type[O_co],
    register=True) -> Callable[
      [Type[I]], Type[I]]:
  """A decorator for creating typed invokables."""
  def _decorator(cls: Type[I]) -> Type[I]:
    """Decorates a class as a typed invokable."""
    if not issubclass(cls, InvokableBase):
      raise TypeError('Invokable must be a subclass of InvokableBase.')
    cls._input_type = input_type
    cls._output_type = output_type
    if register:
      resource_registry.register(cls)
    return cls
  return _decorator

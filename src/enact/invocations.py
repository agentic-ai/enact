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
import collections
import contextlib
import dataclasses
import inspect
import traceback
from typing import Callable, Generic, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union, cast


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


@resource_registry.register
class WrappedException(ExceptionResource):
  """A python exception wrapped as a resource."""


I_contra = TypeVar('I_contra', contravariant=True, bound=interfaces.ResourceBase)
O_co = TypeVar('O_co', covariant=True, bound=interfaces.ResourceBase)


@resource_registry.register
class InputRequest(ExceptionResource):
  """An exception indicating that external input is required."""

  def __init__(
      self,
      invokable: references.Ref['InvokableBase'],
      input: references.Ref,
      requested_output: Type[interfaces.ResourceBase],
      context: interfaces.FieldValue):
    if not references.Store.get_current():
      raise contexts.NoActiveContext(
        'InputRequest must be created within a Store context.')
    super().__init__(
      invokable,
      input,
      requested_output,
      context)

  @property
  def invokable(self) -> references.Ref['InvokableBase']:
    return self.args[0]

  @property
  def input(self) -> references.Ref:
    return self.args[1]

  @property
  def requested_type(self) -> Type[interfaces.ResourceBase]:
    return self.args[2]

  @property
  def context(self) -> interfaces.FieldValue:
    return self.args[3]

  def continue_invocation(
      self,
      invocation: 'Invocation[I_contra, O_co]',
      value: interfaces.ResourceBase,
      strict: bool=True) -> (
        'Invocation[I_contra, O_co]'):
    """Replays the invocation with the given value."""
    ref = references.commit(self)
    def _exception_override(exception_ref: references.Ref[ExceptionResource]):
      if exception_ref == ref:
        return value
      return None
    return invocation.replay(
      exception_override=_exception_override, strict=strict)


@resource_registry.register
class InputRequestOutsideInvocation(ExceptionResource):
  """Raised when input required is called outside an invocation."""


@resource_registry.register
class InvocationError(ExceptionResource):
  """An error during an invocation."""


@resource_registry.register
class InvokableTypeError(InvocationError, TypeError):
  """An type error on a callable input or output."""


@resource_registry.register
class InputChanged(InvocationError):
  """Raised when an input changes during invocation."""


@resource_registry.register
class RequestedTypeUndetermined(InvocationError):
  """Raised when the requested type cannot be determined."""


def _request_input(
    for_resource: Optional[interfaces.ResourceBase],
    requested_type: Optional[Type[interfaces.ResourceBase]],
    context: interfaces.FieldValue=None):
  """Requests an input from a user or external system.

  Args:
    for_resource: The resource for which input is requested.
    requested_type: The type of input requested. If not specified, the type will
      be inferred to be the output type of the current invocation.
    context: Anything that provides context for the request.
  Raises:
    InputRequest: The input request exception.
    InputRequestOutsideInvocation: If the request was made outside an invocation.
  """
  builder: Optional[Builder] = Builder.get_current()
  if not builder:
    raise InputRequestOutsideInvocation(context, requested_type)
  requested_type = requested_type or builder.invokable.get_output_type()
  if not requested_type:
    raise RequestedTypeUndetermined(
      'Requested type must be specified when output type is undetermined.')
  raise InputRequest(
    references.commit(builder.invokable),
    references.commit(for_resource),
    requested_type,
    context)


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
  # Exception raised during call.
  raised: Optional[references.Ref[ExceptionResource]]
  # Whether the exception was raised locally or propagated from a child.
  raised_here: bool
  # Subinvocations associated with this invocation.
  children: List[references.Ref['Invocation']]

  def is_complete(self) -> bool:
    """Returns whether this invocation is complete."""
    return self.raised is not None or self.output is not None


# A function that may override some exceptions that occur during invocation.
ExceptionOverride = Callable[[references.Ref[ExceptionResource]],
                             Optional[interfaces.ResourceBase]]


@resource_registry.register
@dataclasses.dataclass
class Invocation(Generic[I_contra, O_co], resources.Resource):
  """An invocation."""
  request: references.Ref[Request[I_contra, O_co]]
  response: references.Ref[Response[I_contra, O_co]]

  def successful(self) -> bool:
    """Returns true if the invocation completed successfully."""
    if not self.response:
      return False
    return self.response().output is not None

  def get_output(self) -> O_co:
    """Returns the output or raise assertion error."""
    output = self.response().output
    assert output
    return output()

  def get_raised(self) -> ExceptionResource:
    """Returns the raised exception or raise assertion error."""
    raised = self.response().raised
    assert raised
    return raised()

  def get_raised_here(self) -> bool:
    """Whether the exception was originally raised here or in a child."""
    response = self.response()
    assert response.raised, 'No exception was raised.'
    return response.raised_here

  def get_children(self) -> Iterable['Invocation']:
    """Yields the child invocations."""
    children = self.response().children
    for child in children:
      yield child()

  def get_child(self, index: int) -> 'Invocation':
    """Returns the child invocation corresponding to the index."""
    children = self.response().children
    return children[index]()

  def rewind(self, num_calls=1) -> 'Invocation[I_contra, O_co]':
    """Rewinds the invocation by the specified number of calls."""
    invocation = self.deep_copy_resource()
    with invocation.response.modify() as response:
      response.output = None
      for _ in range(num_calls):
        if response.children:
          response.children.pop(-1)
    return invocation

  def replay(
      self,
      exception_override: ExceptionOverride=lambda x: None,
      strict: bool=True) -> (
        'Invocation[I_contra, O_co]'):
    """Replay the invocation, retrying exceptions or overiding them."""
    return self.request().invokable().invoke(
      self.request().input,
      replay_from=self,
      exception_override=exception_override,
      strict=strict)


class ReplayError(InvocationError):
  """An error during replay."""


@contexts.register
class ReplayContext(Generic[I_contra, O_co], contexts.Context):
  """A replay of an invocation."""

  def __init__(
      self,
      subinvocations: Iterable[Invocation[I_contra, O_co]],
      exception_override: ExceptionOverride=lambda x: None,
      strict: bool = True):
    """Create a new replay context.

    Args:
      subinvocations: The subinvocations to replay.
      exception_override: A function that may override exceptions that occur
        during replay. If an exception is not overriden, the invokable will
        be retried.
      strict: If true, an error will be raised when attempting to replay a
        subinvocation, but the provided subinvocation does not match the
        invokable and input. If false, a non-matching sub-invocation will
        be ignored and the corresponding invokable will be retried on its
        actual input.
    """
    self._exception_override = exception_override
    self._available_children = list(subinvocations)
    self._strict = strict

  @classmethod
  def call_or_replay(
      cls, invokable: 'InvokableBase[I_contra, O_co]', arg: I_contra):
    """If there is a replay active, try to use it."""
    context: Optional[ReplayContext[I_contra, O_co]] = (
      ReplayContext.get_current())
    call = invokable.call

    if (isinstance(arg, interfaces.NoneResource) and
        len(inspect.signature(invokable.call).parameters) == 0):
      # Allow invokables that take no call args if they accept NoneResources.
      call = lambda _: invokable.call()  # type: ignore

    if context:
      replayed_output, child_ctx = context._consume_replay(invokable, arg)
      if replayed_output is not None:
        return replayed_output
      else:
        with child_ctx:
          return call(arg)
    return call(arg)

  def _consume_replay(
      self,
      invokable: 'InvokableBase[I_contra, O_co]',
      input: I_contra) -> Tuple[Optional[O_co], 'ReplayContext[I_contra, O_co]']:
    """Replay the invocation if possible and return a child context."""
    for i, child in enumerate(self._available_children):
      if (child.request().invokable == references.commit(invokable) and
          child.request().input == references.commit(input)):
        break
      elif self._strict:
        raise ReplayError(
          f'Expected invocation {invokable}({input}) but got '
          f'{child.request().invokable()}({child.request().input()}).\n'
          f'Ensure that calls to subinvokable are deterministic '
          f'or use strict=False.')
    else:
      # No matching replay found.
      return None, ReplayContext([], self._exception_override)

    # Consume child invocation
    self._available_children.pop(i)
    response = child.response()

    # Replay successful executions.
    if response.output:
      invokable.set_from(response.invokable())
      return response.output(), ReplayContext(
        list(child.get_children()),
        self._exception_override,
        self._strict)

    # Check for exception override
    if response.raised and response.raised_here:
      # Only override exceptions raised in the current frame.
      override = self._exception_override(response.raised)
      invokable.set_from(response.invokable())
      if override is not None:
        # Typecheck the override.
        output_type = invokable.get_output_type()
        if output_type and not isinstance(override, output_type):
          raise InvokableTypeError(
            f'Exception override {override} is not of required type '
            f'{invokable.get_input_type()}.')
        # Set invokable from response.
        return (
          cast(O_co, override),
          ReplayContext(child.get_children(),
                        self._exception_override,
                        self._strict))
      # No override found.
      raise response.raised()

    # Trigger reexecution of the invocation.
    return None, ReplayContext(
      child.get_children(), self._exception_override, self._strict)


@contexts.register
class Builder(Generic[I_contra, O_co], contexts.Context):
  """A builder for invocations."""

  def __init__(
      self,
      invokable: 'InvokableBase[I_contra, O_co]',
      input: references.Ref[I_contra]):
    """Initializes the builder."""
    self.children: List[references.Ref[Invocation]] = []

    self.invokable = invokable
    self.input_ref = input
    self._request = Request(
      references.commit(invokable), input)
    self._invocation: Optional[Invocation] = None
    self._exceptions_raised_by_children: List[Exception] = []

  def record_child(self, invocation: Invocation):
    """Records a subinvocation."""
    self.children.append(references.commit(invocation))

  def record_child_exception(self, exception: Exception):
    """Records a subinvocation."""
    self._exceptions_raised_by_children.append(exception)

  def _is_child_exception(self, exception: Exception) -> bool:
    """Records a subinvocation."""
    return any(exc is exception for exc in self._exceptions_raised_by_children)

  @property
  def invocation(self) -> Invocation[I_contra, O_co]:
    assert self._invocation, (
      'The "call" function must be called before accessing the invocation.')
    return self._invocation

  def call(self) -> O_co:
    """Call the invokable and set self.invocation."""
    parent: Optional[Builder] = Builder.get_current()
    with self:
      output: Optional[references.Ref[O_co]] = None
      exception: Optional[references.Ref[ExceptionResource]] = None
      python_exc: Optional[Exception] = None
      try:
        input_resource = self.input_ref()
        output_resource = ReplayContext.call_or_replay(
          self.invokable, input_resource)
        if references.commit(input_resource) != self.input_ref:
          raise InputChanged(
            f'Input changed during invocation of {self.invokable} on input '
            f'{input_resource}. Only the invokable may change.')
        if output_resource is None:
          output_resource = interfaces.NoneResource()
        if not isinstance(output_resource, interfaces.ResourceBase):
          raise InvokableTypeError(
            f'Invokable {self.invokable} returned {output_resource} '
            f'which is not a resource.')
        output = cast(references.Ref[O_co], references.commit(output_resource))
      except ExceptionResource as e:
        python_exc = e
        exception = references.commit(e)
        raise
      except Exception as e:
        python_exc = e
        exception = references.commit(WrappedException(traceback.format_exc()))
        raise
      finally:
        raised_here = False
        if python_exc:
          if parent:
            parent.record_child_exception(python_exc)
          raised_here = not self._is_child_exception(python_exc)
        response = Response(
          references.commit(self.invokable), output,
          exception, raised_here, self.children)
        self._invocation = Invocation(
          references.commit(self._request),
          references.commit(response))
        if parent:
          parent.record_child(self._invocation)
      return cast(O_co, output_resource)


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
          f'Untyped invokables must be called with a single resource argument, '
          f'got args={args} and kwargs={kwargs}.')
    else:
      arg = args[0]

    if input_type and not isinstance(arg, input_type):
      raise InvokableTypeError(
        f'Input type {type(arg).__qualname__} does not match '
        f'{input_type.__qualname__}.')

    parent: Optional[Builder] = Builder.get_current()

    # Execution not tracked, so just call the invokable.
    if not parent:
      return ReplayContext.call_or_replay(self, arg)

    builder = Builder(self, references.commit(arg))
    output = builder.call()
    output_type = self.get_output_type()
    if output_type and not isinstance(output, output_type):
      raise InvokableTypeError(
        f'Output type {type(output)} does not match {output_type}.')
    return output

  def invoke(
      self,
      arg: Optional[references.Ref[I_contra]]=None,
      replay_from: Optional[Invocation[I_contra, O_co]]=None,
      exception_override: ExceptionOverride=lambda _: None,
      raise_on_invocation_error:bool=True,
      strict: bool=True) -> Invocation[I_contra, O_co]:
    """Invoke the invokable, tracking invocation metadata.

    Args:
      arg: The input resource.
      replay_from: An optional invocation to replay form.
      exception_override: If replaying, an optional override for replayed
        exceptions.
      raise_on_invocation_error: Whether invocation errors should be reraised.
      strict: Whether replay should fail if the replayed invocation
        does not match the current invocation.
    Returns:
      The invocation generated.
    """
    if arg is None:
      arg = cast(references.Ref[I_contra],
                 references.commit(interfaces.NoneResource()))
    if not isinstance(arg, references.Ref):
      raise InvokableTypeError('Input must be a reference.')

    exit_stack = contextlib.ExitStack()
    # Execute in a top-level context to ensure that there are no parents.
    exit_stack.enter_context(Builder.top_level())
    if replay_from:
      exit_stack.enter_context(ReplayContext.top_level())
      exit_stack.enter_context(ReplayContext(
        [replay_from], exception_override, strict))

    with exit_stack:
      builder = Builder(self, arg)
      try:
        builder.call()
      except InvocationError:
        if raise_on_invocation_error:
          raise
      except Exception:
        pass  # Do nothing
      invocation = builder.invocation
    return invocation


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


@resource_registry.register
@dataclasses.dataclass
class RequestInput(Invokable):
  """An invokable that raises an InputRequest."""
  requested_type: Type[interfaces.ResourceBase]
  context: Optional[interfaces.FieldValue] = None

  def call(self, resource: interfaces.ResourceBase) -> interfaces.ResourceBase:
    _request_input(resource, self.requested_type, self.context)
    assert False

def request_input(
    requested_type: Type[C],
    for_resource: Optional[interfaces.ResourceBase]=None,
    context: Optional[interfaces.FieldValue]=None) -> C:
  """Request an input from an external system / user.

  Args:
    requested_type: The type of input to request.
    for_resource: The resource to request input for. If unset, defaults to None.
    context: An optional context to provide to the input request, e.g., instructions
      to a user.
  Returns:
    The requested input. Note that this function will not return a value
    during normal execution, but will raise an InputRequest exception,
    which can be used to resume the execution with an injected value.
  Raises:
    InputRequest: Raised to halt execution in order to await input.
  """
  if for_resource is None:
    for_resource = interfaces.NoneResource()
  return RequestInput(requested_type, context)(for_resource)


class InvocationGenerator(
    collections.Generator,
    Generic[I_contra, O_co]):
  """A generator that yields InputRequests from an invocation."""

  def __init__(
      self,
      invokable: InvokableBase[I_contra, O_co],
      input: references.Ref[I_contra]):
    """Initializes an interactive invocation."""
    self._invocation: Optional[Invocation[I_contra, O_co]] = None
    self._invokable = invokable
    self._input = input
    self._request_input: Optional[interfaces.ResourceBase] = None

  @property
  def complete(self) -> bool:
    """Whether the invocation is complete."""
    if not self._invocation:
      return False
    if self._invocation.successful():
      return True
    return not isinstance(self._invocation.get_raised(), InputRequest)

  @property
  def invocation(self) -> Invocation[I_contra, O_co]:
    """The invocation."""
    if not self._invocation:
      raise InvocationError(
        'Invocation not started, please call next.')
    return self._invocation

  @property
  def input_request(self) -> InputRequest:
    """The current input request."""
    if self.complete:
      raise InvocationError('Invocation is complete.')
    input_request = self.invocation.get_raised()
    assert isinstance(input_request, InputRequest)
    return input_request

  def context(self) -> interfaces.FieldValue:
    """Returns the context for the input request."""
    return self.input_request.context

  def requested_type(self) -> Type[interfaces.ResourceBase]:
    """Returns the requested type of the input."""
    return self.input_request.requested_type

  def for_resource(self) -> interfaces.ResourceBase:
    """For which resource is the input requested."""
    return self.input_request.input()

  def set_input(self, resource: interfaces.ResourceBase):
    """Set input for the next call to __next__.

    This allows using the generator in iterator-style, e.g.,

      for input_request in invocation_generator:
        input_request.set_input(...)

    Which can be more convenient than using send():

      input_request = next(invocation_generator)
      while True:
        try:
          input_request.send(...)
        except StopIteration:
          break

    Args:
      resource: The resource to set as input.
    """
    self._request_input = resource.deep_copy_resource()

  def __next__(self) -> InputRequest:
    """Continues the invocation until the next input request or completion."""
    if self.complete:
      raise StopIteration()

    if not self._invocation:
      self._invocation = self._invokable.invoke(self._input)
      if self.complete:
        raise StopIteration()
      return self.input_request
    else:
      if self._request_input is None:
        if not issubclass(interfaces.NoneResource,
                          self.input_request.requested_type):
          raise InvocationError(
            'Invocation requests non-None input. Please use \'send(...)\' '
            'instead or set the input using \'set_input(...)\'.')
        self._request_input = interfaces.NoneResource()
      self._invocation = self.input_request.continue_invocation(
        self._invocation, self._request_input)
      self._request_input = None
      if self.complete:
        raise StopIteration()
      return self.input_request

  def send(self, value) -> InputRequest:
    if not self._invocation:
      if value is not None:
        raise TypeError('Can\'t send non-None value to a just-started generator.')
      return self.__next__()
    if self.complete:
      raise StopIteration()
    if not isinstance(value, self.input_request.requested_type):
      raise InvokableTypeError(
        f'Input type {type(value)} does not match requested type: '
        f'{self.input_request.requested_type}.')
    self._invocation = self.input_request.continue_invocation(
      self.invocation, value)
    if self.complete:
      raise StopIteration()
    return self.input_request

  def throw(self, *args, **kwargs):
    super().throw(*args, **kwargs)

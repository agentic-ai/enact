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

"""Functionality for invokable resources."""

import contextlib
import dataclasses
import inspect
import traceback
from typing import (
  Any, Callable, Generic, Iterable, List, Mapping, Optional, Tuple, Type,
  TypeVar, cast)

from enact import contexts
from enact import interfaces
from enact import references
from enact import resources
from enact import resource_registry


AnyT = TypeVar('AnyT')
ResourceT = TypeVar('ResourceT', bound=interfaces.ResourceBase)
ExceptionT = TypeVar('ExceptionT', bound='ExceptionResource')


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
    yield resource_registry.to_field_value(self.args)

  @classmethod
  def from_fields(cls: Type[ResourceT],
                  field_dict: Mapping[str, interfaces.FieldValue]) -> ResourceT:
    """Constructs the resource from a value dictionary."""
    return cls(*resource_registry.from_field_value(field_dict['args']))

  def set_from(self: ResourceT, other: Any):
    """Sets the fields of this resource from another resource."""
    super().set_from(other)  # Raise error


@resource_registry.register
class WrappedException(ExceptionResource):
  """A python exception wrapped as a resource."""

# Input value type.
I_contra = TypeVar('I_contra', contravariant=True)
# Output value type.
O_co = TypeVar('O_co', covariant=True)


@resource_registry.register
class InputRequest(ExceptionResource):
  """An exception indicating that external input is required."""

  def __init__(
      self,
      invokable: references.Ref[Callable],
      for_value: references.Ref,
      requested_output: Type,
      context: Any):
    """Initialize a new input request.

    Args:
      invokable: The parent invokable that issued the request.
      for_input: A reference to the value input is requested for.
      requested_output: The type of output that is requested.
      context: Any additional context required to resolve the input request.
    """
    if not references.Store.get_current():
      raise contexts.NoActiveContext(
        'InputRequest must be created within a Store context.')
    super().__init__(
      invokable,
      for_value,
      requested_output,
      context)

  @property
  def invokable(self) -> references.Ref[Callable]:
    """Returns the invokable that requested the input."""
    return self.args[0]

  @property
  def for_value(self) -> references.Ref:
    """Returns a reference to the resource for which input is requested."""
    return self.args[1]

  @property
  def requested_type(self) -> Type:
    """Returns the type of input requested."""
    return self.args[2]

  @property
  def context(self) -> Any:
    return self.args[3]

  def continue_invocation(
      self,
      invocation: 'Invocation[I_contra, O_co]',
      value: Any,
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

  async def continue_invocation_async(
      self,
      invocation: 'Invocation[I_contra, O_co]',
      value: Any,
      strict: bool=True):
    ref = references.commit(self)
    def _exception_override(exception_ref: references.Ref[ExceptionResource]):
      if exception_ref == ref:
        return value
      return None
    return await invocation.replay_async(
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
    for_input: Any,
    requested_type: Optional[Type],
    context: Any=None):
  """Requests an input from a user or external system.

  Args:
    for_resource: The resource for which input is requested.
    requested_type: The type of input requested. If not specified, the type will
      be inferred to be the output type of the current invocation.
    context: Anything that provides context for the request.
  Raises:
    InputRequest: The input request exception.
    InputRequestOutsideInvocation: If the request was made outside an
      invocation.
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
    references.commit(for_input),
    requested_type,
    context)


@resource_registry.register
@dataclasses.dataclass
class Request(Generic[I_contra, O_co], resources.Resource):
  """An invocation request."""
  invokable: references.Ref['_InvokableBase[I_contra, O_co]']
  input: references.Ref[I_contra]


@resource_registry.register
@dataclasses.dataclass
class Response(Generic[I_contra, O_co], resources.Resource):
  """An invocation response."""
  invokable: references.Ref[Callable]
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
ExceptionOverride = Callable[[references.Ref[ExceptionResource]], Any]



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

  def get_input(self) -> Any:
    """Returns the input."""
    return self.request().input()

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

  def clear_output(self):
    """Clear the output of the invocation."""
    with self.response.modify() as response:
      response.output = None

  def rewind(self, num_calls=1) -> 'Invocation[I_contra, O_co]':
    """Rewinds the invocation by the specified number of calls."""
    invocation = self.deepcopy_resource()
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
    invokable = resource_registry.wrap(self.request().invokable())
    if isinstance(invokable, AsyncInvokableBase):
      raise InvocationError(
        'Cannot replay async invocations synchronously. '
        'Use the "replay_async" coroutine instead.')
    assert isinstance(invokable, InvokableBase)
    return invokable.invoke(
      self.request().input,
      replay_from=self,
      exception_override=exception_override,
      strict=strict)

  async def replay_async(
      self,
      exception_override: ExceptionOverride=lambda x: None,
      strict: bool=True) -> (
        'Invocation[I_contra, O_co]'):
    """Replay the invocation, retrying exceptions or overiding them."""
    invokable = resource_registry.wrap(self.request().invokable())
    if isinstance(invokable, InvokableBase):
      raise InvocationError(
        'Cannot replay synchronous invocations asynchronously. '
        'Use the "replay" function instead.')
    assert isinstance(invokable, AsyncInvokableBase)
    return await invokable.invoke(
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
      subinvocations: Iterable[references.Ref[Invocation]],
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
    super().__init__()
    self._exception_override = exception_override
    self._available_children = list(subinvocations)
    if not all(isinstance(x, references.Ref) for x in self._available_children):
      assert False
    self._strict = strict

  @classmethod
  def call_or_replay(
      cls,
      invokable: 'InvokableBase[I_contra, O_co]',
      arg: I_contra) -> O_co:
    """If there is a replay active, try to use it."""
    context: Optional[ReplayContext[I_contra, O_co]] = (
      ReplayContext.get_current())
    call = invokable.call

    if arg is None and len(inspect.signature(invokable.call).parameters) == 0:
      # Allow invokables that take no call args if they accept NoneResources.
      # pylint: disable=unnecessary-lambda-assignment
      call = lambda _: invokable.call()  # type: ignore

    if context:
      # pylint: disable=protected-access
      replayed_output, child_ctx = context._consume_replay(invokable, arg)
      if replayed_output is not None:
        return replayed_output
      else:
        with child_ctx:
          return call(arg)
    return call(arg)

  @classmethod
  async def async_call_or_replay(
      cls,
      invokable: 'AsyncInvokableBase[I_contra, O_co]',
      arg: I_contra) -> O_co:
    """If there is a replay active, try to use it."""
    context: Optional[ReplayContext[I_contra, O_co]] = (
      ReplayContext.get_current())
    call = invokable.call
    if (arg is None and
        len(inspect.signature(invokable.call).parameters) == 0):
      # Allow invokables that take no call args if they accept NoneResources.
      # pylint: disable=unnecessary-lambda-assignment
      call = lambda _: invokable.call()  # type: ignore

    if context:
      # pylint: disable=protected-access
      replayed_output, child_ctx = context._consume_replay(invokable, arg)
      if replayed_output is not None:
        return replayed_output
      else:
        with child_ctx:
          result = await call(arg)
          return result
    result = await call(arg)
    return result

  def _consume_replay(
      self,
      invokable: '_InvokableBase[I_contra, O_co]',
      input_resource: I_contra) -> Tuple[Optional[O_co],
                                'ReplayContext[I_contra, O_co]']:
    """Replay the invocation if possible and return a child context."""
    request = references.commit(Request(
      references.commit(invokable),
      references.commit(input_resource)))
    for i, child in enumerate(self._available_children):
      if child().request == request:
        break
      elif self._strict:
        raise ReplayError(
          f'Expected invocation {invokable}({input_resource}) but got '
          f'{child().request().invokable()}({child().request().input()}).\n'
          f'Ensure that calls to subinvokables are deterministic '
          f'or use strict=False.')
    else:
      # No matching replay found.
      return None, ReplayContext([], self._exception_override)

    # Consume child invocation
    self._available_children.pop(i)
    replay_response = child().response()
    replay_children = replay_response.children

    # Replay successful executions.
    if replay_response.output:
      invokable.set_from(resource_registry.wrap(replay_response.invokable()))
      Builder.register_replayed_subinvocations(replay_children)
      return replay_response.output(), ReplayContext(
        replay_children,
        self._exception_override,
        self._strict)

    # Check for exception override
    if replay_response.raised and replay_response.raised_here:
      # We only override exceptions at the point of the stack where they were
      # originally raised.
      override = self._exception_override(replay_response.raised)
      invokable.set_from(resource_registry.wrap(replay_response.invokable()))
      if override is not None:
        # Typecheck the override.
        output_type = invokable.get_output_type()
        if output_type and not isinstance(override, output_type):
          raise InvokableTypeError(
            f'Exception override {override} is not of required type '
            f'{invokable.get_input_type()}.')
        Builder.register_replayed_subinvocations(replay_children)
        # Set invokable from response.
        return (
          cast(O_co, override),
          ReplayContext(replay_children,
                        self._exception_override,
                        self._strict))

    # Trigger reexecution of the invocation.
    return None, ReplayContext(
      replay_children,
      self._exception_override, self._strict)


class IncompleteSubinvocationError(InvocationError):
  """A subinvocation hasn't completed during the call."""


@contexts.register
class Builder(Generic[I_contra, O_co], contexts.Context):
  """A builder for invocations."""

  def __init__(
      self,
      invokable: '_InvokableBase[I_contra, O_co]',
      input_resource: references.Ref[I_contra]):
    """Initializes the builder."""
    super().__init__()
    self.invokable = invokable
    self.input_ref = input_resource

    self._children: Optional[List[Builder]] = None
    self._replayed_subinvocations: Optional[
      List[references.Ref[Invocation]]] = None
    self._request = Request(
      references.commit(invokable), input_resource)

    self._invocation: Optional[Invocation] = None
    self._parent: Optional[Builder] = self.get_current()
    if self._parent:
      self._parent.register_child(self)
    self.exception_raised: Optional[Exception] = None

  @property
  def completed(self) -> bool:
    """Returns true if the invocation is complete."""
    return self._invocation is not None

  def register_child(self, child: 'Builder'):
    """Registers a subinvocation."""
    if self._children is None:
      self._children = []
    self._children.append(child)

  @classmethod
  def register_replayed_subinvocations(
      cls, subinvocations: Iterable[references.Ref[Invocation]]):
    """Registers a list of subinvocations that were replayed."""
    builder = cls.get_current()
    if not builder:
      return
    # pylint: disable=protected-access
    builder._replayed_subinvocations = list(subinvocations)

  def _get_subinvocations(self) -> List[references.Ref[Invocation]]:
    """Returns the list of subinvocations."""
    assert (
      self._replayed_subinvocations is None or self._children is None), (
      'Cannot have both replayed and non-replayed subinvocations.')
    if self._replayed_subinvocations is not None:
      return self._replayed_subinvocations
    children = self._children or []
    for i, child in enumerate(children or []):
      if not child.completed:
        raise IncompleteSubinvocationError(
          f'Subinvocation {i} did not complete during invocation of parent:'
          f' {child.invokable} invoked on'
          f' {child.input_ref()}')
    return [references.commit(c.invocation) for c in children]

  def _is_child_exception(self, exception: Exception) -> bool:
    """Checks if the exception was originally raised by an immediate child."""
    children = self._children or []
    assert not self._replayed_subinvocations, (
      'Subinvocations were replayed, but an exception was raised.')
    return any(s.exception_raised is exception for s in children)

  @property
  def invocation(self) -> Invocation[I_contra, O_co]:
    assert self._invocation, (
      'The "call" function must be called before accessing the invocation.')
    return self._invocation

  def _check_call_valid(
      self,
      input_value: Any):
    """Checks that the call did not do invalid things."""
    if references.commit(input_value) != self.input_ref:
      raise InputChanged(
        f'Input changed during invocation of {self.invokable} on input '
        f'{input_value}. Only the invokable may change.')

  def _process_output(
      self,
      output: Any) -> references.Ref[O_co]:
    """Process the output and check for errors."""
    return cast(references.Ref[O_co], references.commit(output))

  def _wrap_exception(self, exception: Exception) -> ExceptionResource:
    """Wraps an exception if necessary."""
    if not isinstance(exception, ExceptionResource):
      return WrappedException(traceback.format_exc())
    return exception

  def _create_invocation(
      self,
      output_ref: Optional[references.Ref[O_co]],
      exception: Optional[Exception]):
    """Process a call result and set the invocation object."""
    exception_ref: Optional[references.Ref[ExceptionResource]] = None
    raised_here = False
    if exception:
      exception_ref = references.commit(
        self._wrap_exception(exception))
      self.exception_raised = exception
      raised_here = not self._is_child_exception(exception)
    subinvocations = self._get_subinvocations()
    response: Response = Response(
      references.commit(self.invokable), output_ref,
      exception_ref, raised_here, children=subinvocations)
    self._invocation = Invocation(
      references.commit(self._request),
      references.commit(response))

  def call(self) -> O_co:
    """Call the invokable and set self.invocation."""
    with self:
      exception: Optional[Exception] = None
      output_ref: Optional[references.Ref[O_co]] = None
      try:
        input_value = self.input_ref()
        invokable = self.invokable
        assert isinstance(invokable, InvokableBase)
        output_value = ReplayContext.call_or_replay(
          invokable, input_value)
        self._check_call_valid(input_value)
        output_ref = self._process_output(output_value)
      except Exception as e:
        exception = e
        raise
      finally:
        self._create_invocation(output_ref, exception)
      return cast(O_co, output_value)

  async def async_call(self) -> O_co:
    """Call the async invokable and set self.invocation."""
    with self:
      exception: Optional[Exception] = None
      output_ref: Optional[references.Ref[O_co]] = None
      try:
        input_value = self.input_ref()
        invokable = self.invokable
        assert isinstance(invokable, AsyncInvokableBase)
        output_value = await ReplayContext.async_call_or_replay(
          invokable, input_value)
        self._check_call_valid(input_value)
        output_ref = self._process_output(output_value)
      except Exception as e:
        exception = e
        raise
      finally:
        self._create_invocation(output_ref, exception)
      return cast(O_co, output_value)

class _InvokableBase(Generic[I_contra, O_co], interfaces.ResourceBase):
  """Base class for sync / async invokable resources."""
  _input_type: Optional[Type[I_contra]] = None
  _output_type: Optional[Type[O_co]] = None

  @classmethod
  def get_input_type(cls) -> Optional[Type[I_contra]]:
    """Returns the type of the input if known."""
    return cls._input_type

  def __call__(self, *args, **kwargs):
    """Subclasses implement this directly or as async."""
    raise NotImplementedError()

  def call(self, *args, **kwargs):
    """Subclasses implement this directly or as async."""
    raise NotImplementedError()

  @classmethod
  def _check_input_type(cls, value: Any):
    """Check the input type."""
    input_type: Optional[Type] = cls._input_type
    # pylint: disable=isinstance-second-argument-not-valid-type
    if input_type and not isinstance(value, input_type):
      raise InvokableTypeError(
        f'Input must be of type {cls._input_type}, but got {type(value)}.')

  @classmethod
  def _check_output_type(cls, value: Any):
    """Check the output type."""
    output_type: Optional[Type] = cls._output_type
    # pylint: disable=isinstance-second-argument-not-valid-type
    if output_type is not None and not isinstance(value, output_type):
      raise InvokableTypeError(
        f'Output must be of type {cls._output_type}, but got {type(value)}.')

  @classmethod
  def get_output_type(cls) -> Optional[Type[O_co]]:
    """Returns the type of the output if known."""
    return cls._output_type

  @staticmethod
  def _process_invoke_arg(
      arg: Optional[references.Ref[I_contra]]) -> (
        references.Ref[I_contra]):
    """Processes an invocation argument."""
    if arg is None:
      arg = cast(references.Ref[I_contra],
                 references.commit(None))
    if not isinstance(arg, references.Ref):
      raise InvokableTypeError('Input must be a reference.')
    return arg

  @staticmethod
  def _invoke_exit_stack(
      replay_from: Optional[Invocation[I_contra, O_co]]=None,
      exception_override: ExceptionOverride=lambda _: None,
      strict: bool=False) -> contextlib.ExitStack:
    """Creates an exit stack for invoking an invokable."""
    exit_stack = contextlib.ExitStack()
    # Execute in a top-level context to ensure that there are no parents.
    exit_stack.enter_context(Builder.top_level())
    if replay_from:
      exit_stack.enter_context(ReplayContext.top_level())
      exit_stack.enter_context(ReplayContext(
        [references.commit(replay_from)],
        exception_override, strict))
    return exit_stack


class InvokableBase(_InvokableBase[I_contra, O_co]):
  """Base class for invokable resources."""

  def call(self, value: I_contra) -> O_co:
    """Executes the invokable."""
    raise NotImplementedError()

  def __call__(self, arg=None) -> O_co:
    """Executes the invokable, tracking invocation metadata."""
    parent: Optional[Builder] = Builder.get_current()

    self._check_input_type(arg)
    # Execution not tracked, so just call or replay the invokable.
    if not parent:
      output = ReplayContext.call_or_replay(self, arg)
    else:
      builder = Builder(self, references.commit(arg))
      output = builder.call()
    self._check_output_type(output)
    return output

  def invoke(
      self,
      arg: Optional[references.Ref[I_contra]]=None,
      replay_from: Optional[Invocation[I_contra, O_co]]=None,
      exception_override: ExceptionOverride=lambda _: None,
      raise_on_errors: Tuple[Type[Exception], ...]=(
        InvocationError, interfaces.FrameworkError),
      strict: bool=True,
      commit: bool=True) -> Invocation[I_contra, O_co]:
    """Invoke the invokable, tracking invocation metadata.

    Args:
      arg: The input resource.
      replay_from: An optional invocation to replay form.
      exception_override: If replaying, an optional override for replayed
        exceptions.
      raise_on_errors: Which errors should raise beyond the invocation.
      strict: Whether replay should fail if the replayed invocation
        does not match the current invocation.
      commit: Whether to commit the new invocation object.
    Returns:
      The invocation generated.
    """
    arg = self._process_invoke_arg(arg)

    with self._invoke_exit_stack(replay_from, exception_override, strict):
      builder = Builder(self, arg)
      try:
        builder.call()
      except raise_on_errors:
        raise
      except Exception:  # pylint: disable=broad-exception-caught
        if not builder.completed:
          raise
      invocation = builder.invocation
    if commit:
      references.commit(invocation)
    return invocation


class AsyncInvokableBase(_InvokableBase[I_contra, O_co]):
  """Base class for invokable resources."""

  async def call(self, value: I_contra) -> O_co:  # pylint: disable=invalid-overridden-method
    """Executes the async invokable."""
    raise NotImplementedError()

  async def __call__(self, arg=None) -> O_co:  # pylint: disable=invalid-overridden-method
    """Executes the async invokable, tracking invocation metadata."""
    parent: Optional[Builder] = Builder.get_current()

    self._check_input_type(arg)

    # Execution not tracked, so just call or replay the invokable.
    if not parent:
      output = await ReplayContext.async_call_or_replay(self, arg)
    else:
      builder = Builder(self, references.commit(arg))
      output = await builder.async_call()

    self._check_output_type(output)
    return output

  async def invoke(
      self,
      arg: Optional[references.Ref[I_contra]]=None,
      replay_from: Optional[Invocation[I_contra, O_co]]=None,
      exception_override: ExceptionOverride=lambda _: None,
      raise_on_errors: Tuple[Type[Exception], ...]=(
        InvocationError, interfaces.FrameworkError),
      strict: bool=True,
      commit: bool=True) -> Invocation[I_contra, O_co]:
    """Invoke the invokable, tracking invocation metadata.

    Args:
      arg: The input resource.
      replay_from: An optional invocation to replay form.
      exception_override: If replaying, an optional override for replayed
        exceptions.
      raise_on_errors: Which errors should raise beyond the invocation.
      strict: Whether replay should fail if the replayed invocation
        does not match the current invocation.
      commit: Whether to commit the new invocation object.
    Returns:
      The invocation generated.
    """
    arg = self._process_invoke_arg(arg)

    with self._invoke_exit_stack(replay_from, exception_override, strict):
      builder = Builder(self, arg)
      try:
        await builder.async_call()
      except raise_on_errors:
        raise
      except Exception:  # pylint: disable=broad-exception-caught
        if not builder.completed:
          raise
      invocation = builder.invocation
    if commit:
      references.commit(invocation)
    return invocation


@dataclasses.dataclass
class Invokable(InvokableBase[I_contra, O_co], resources.Resource):
  """Base class for dataclass-based invokable resources."""


@dataclasses.dataclass
class AsyncInvokable(AsyncInvokableBase[I_contra, O_co], resources.Resource):
  """Base class for dataclass-based async invokable resources."""


I = TypeVar('I', bound=_InvokableBase)


def typed_invokable(
    input_type: Optional[Type],
    output_type: Optional[Type],
    register=True) -> Callable[
      [Type[I]], Type[I]]:
  """A decorator for creating typed invokables."""
  def _decorator(cls: Type[I]) -> Type[I]:
    """Decorates a class as a typed invokable."""
    if not issubclass(cls, _InvokableBase):
      raise TypeError('Invokable must be a subclass of InvokableBase.')
    # pylint: disable=protected-access
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
  requested_type: Type
  context: Any = None

  def call(self, input_to_user: Any) -> Any:
    _request_input(input_to_user, self.requested_type, self.context)
    assert False


def request_input(
    requested_type: Type[AnyT],
    for_value: Any=None,
    context: Optional[interfaces.FieldValue]=None) -> AnyT:
  """Request an input from an external system / user.

  Args:
    requested_type: The type of input to request.
    for_value: The value to request input for. Defaults to None.
    context: An optional context to provide to the input request, e.g.,
      instructions to a user.
  Returns:
    The requested input. Note that this function will not return a value
    during normal execution, but will raise an InputRequest exception,
    which can be used to resume the execution with an injected value.
  Raises:
    InputRequest: Raised to halt execution in order to await input.
  """
  return RequestInput(requested_type, context)(for_value)

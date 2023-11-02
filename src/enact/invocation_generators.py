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

"""Invocation generators for handling input requests."""

from typing import Any, Callable, Dict, Generic, Optional, Sequence, TypeVar, cast

from enact import function_wrappers, invocations, resource_registry
from enact import references

# Input value type.
I_contra = TypeVar('I_contra', contravariant=True)
# Output value type.
O_co = TypeVar('O_co', covariant=True)


class InvocationGenerator(Generic[I_contra, O_co]):
  """A generator that yields InputRequests from an invocation."""

  @classmethod
  def from_callable(
      cls,
      fun: Callable,
      args: Sequence=(),
      kwargs: Optional[Dict[str, Any]]=None) -> 'InvocationGenerator':
    """Create an InvocationGenerator from a callable.

    Args:
      fun: The callable to generate an invocation from.
      args: The arguments to the callable.
      kwargs: The keyword arguments to the callable.
    """
    kwargs = kwargs or {}
    if isinstance(fun, invocations.AsyncInvokableBase):
      raise TypeError(
        'AsyncInvokables are not currently supported by InvocationGenerator.')
    if isinstance(fun, invocations.InvokableBase):
      if not len(args) >= 1: raise ValueError(
        'Cannot specify more than one argument for an invokable.')
      if kwargs:
        raise ValueError('May not use keyword arguments with an invokable.')
      arg = args[0] if args else None
      return InvocationGenerator(
        invokable=fun,
        input_ref=references.commit(arg))
    if not function_wrappers.get_wrapper_type(fun):
      raise ValueError(
        'Cannot use invocation generator with an unregistered function.')
    invocation = function_wrappers.invoke(fun, args, kwargs)
    return cls.from_invocation(invocation)

  @classmethod
  def from_invocation(
      cls,
      invocation: invocations.Invocation[I_contra, O_co]) -> (
        'InvocationGenerator[I_contra, O_co]'):
    return InvocationGenerator(from_invocation=invocation)

  def __init__(
      self,
      invokable: Optional[invocations.InvokableBase[I_contra, O_co]]=None,
      input_ref: Optional[references.Ref[I_contra]]=None,
      from_invocation: Optional[invocations.Invocation[I_contra, O_co]]=None):
    """Initializes an interactive invocation."""
    if invokable and not isinstance(invokable, invocations.InvokableBase):
      if isinstance(invokable, invocations.AsyncInvokableBase):
        raise TypeError(
          'AsyncInvokables cannot be used with InvocationGenerator.')
      raise TypeError(
        'Invokable must be a child of Syncinvocations.InvokableBase.')
    if from_invocation is not None and invokable is not None:
      raise ValueError(
        'Cannot specify both an invokable and an invocation.')
    if from_invocation is not None and input_ref is not None:
      raise ValueError(
        'Cannot specify both an input and an invocation.')
    self._invocation: Optional[invocations.Invocation[I_contra, O_co]] = None
    self._from_invocation = from_invocation
    self._invokable = invokable
    self._input = input_ref
    self._request_input: Any = None

  @property
  def complete(self) -> bool:
    """Whether the invocation is complete."""
    if not self._invocation:
      return False
    if self._invocation.successful():
      return True
    if not self._invocation.response().raised:
      return False
    return not isinstance(self._invocation.get_raised(),
                          invocations.InputRequest)

  @property
  def invocation(self) -> invocations.Invocation[I_contra, O_co]:
    """The invocation."""
    if not self._invocation:
      raise invocations.InvocationError(
        'Invocation not started, please call next.')
    return self._invocation

  @property
  def input_request(self) -> invocations.InputRequest:
    """The current input request."""
    if self.complete:
      raise invocations.InvocationError('Invocation is complete.')
    input_request = self.invocation.get_raised()
    assert isinstance(input_request, invocations.InputRequest)
    return input_request

  def set_input(self, value: Any):
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
    self._request_input = resource_registry.deepcopy(value)

  def __iter__(self) -> 'InvocationGenerator[I_contra, O_co]':
    """Returns the generator."""
    return self

  def __next__(self) -> invocations.InputRequest:
    """Continues the invocation until the next input request or completion."""
    if self.complete:
      raise StopIteration()

    if not self._invocation:
      if self._from_invocation:
        self._invocation = self._from_invocation.replay()
      else:
        assert self._invokable
        if not self._input:
          if self._invokable.get_input_type() != type(None):
            raise invocations.InvocationError(
              'Invokable has non-None input type. '
              'Please provide an explicit input reference on generator '
              'construction')
          self._input = cast(references.Ref[I_contra],
                             references.commit(None))
        self._invocation = self._invokable.invoke(self._input)
      if self.complete:
        raise StopIteration()
      return self.input_request
    else:
      if self._request_input is None:
        if not self.input_request.requested_type is type(None):
          raise invocations.InvocationError(
            'Invocation requests non-None input. Please use \'send(...)\' '
            'instead or set the input using \'set_input(...)\'.')
      self._invocation = self.input_request.continue_invocation(
        self._invocation, self._request_input)
      self._request_input = None
      if self.complete:
        raise StopIteration()
      return self.input_request

  def send(self, value) -> invocations.InputRequest:
    if not self._invocation:
      if value is not None:
        raise TypeError(
          'Can\'t send non-None value to a just-started generator.')
      return next(self)
    if self.complete:
      raise StopIteration()
    if not isinstance(value, self.input_request.requested_type):
      raise invocations.InvokableTypeError(
        f'Input type {type(value)} does not match requested type: '
        f'{self.input_request.requested_type}.')
    self._invocation = self.input_request.continue_invocation(
      self.invocation, value)
    if self.complete:
      raise StopIteration()
    return self.input_request

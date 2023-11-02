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

"""Wrappers for python callables."""

import inspect
from types import MethodType
from typing import (
  Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar,
  Union)
import wrapt  # type: ignore
import dataclasses

from enact import interfaces
from enact import invocations
from enact import references
from enact import resource_registry
from enact import resources


C = TypeVar('C', bound=Callable)


@resource_registry.register
@dataclasses.dataclass
class CallArgs(resources.Resource):
  """Represents call arguments to a python function.

  This class is used for wrapping python functions.
  """
  args: List
  kwargs: Dict[str, Any]

  def to_python_args(self) -> Tuple[List, Dict[str, Any]]:
    """Transform into python call args."""
    return self.args, self.kwargs

  @classmethod
  def from_python_args(cls, *args, **kwargs) -> 'CallArgs':
    """Get CallArgs from python call args."""
    return cls(list(args), dict(kwargs))



def _create_wrapt_function_wrapper(
    enact_wrapper_type: Type[resource_registry.FunctionWrapper]):
  """Create a wrapt function wrapper for the given enact wrapper."""
  class WraptBoundFunctionWrapper(wrapt.BoundFunctionWrapper):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    @property
    def _enact_function_wrapper_type(self) -> Type[
        resource_registry.FunctionWrapper]:
      return enact_wrapper_type

  class WraptFunctionWrapper(wrapt.FunctionWrapper):
    __bound_function_wrapper__ = WraptBoundFunctionWrapper

    @property
    def _enact_function_wrapper_type(self) -> Type[
        resource_registry.FunctionWrapper]:
      return enact_wrapper_type

  return WraptFunctionWrapper


def _wrapt_wrapper(
    enact_function_wrapper_type: Type[resource_registry.FunctionWrapper]):
  """A decorator that routes a function into an Invokable.

  This decorator will replace a callable with a wrapt wrapper
  that routes calls into an enact Invokable of the type indicated
  in enact_wrapper_type.

  Args:
    enact_wrapper_type: The type of the enact
  """
  def _wrapper_fun(wrapped, instance, args, kwargs):
    call_args = CallArgs.from_python_args(*args, **kwargs)
    wrapper: Union[resource_registry.FunctionWrapper,
                   resource_registry.MethodWrapper]
    if instance is not None:
      assert not isinstance(instance, type), (
        'Wrapping class methods is currently not supported.')
      wrapper = enact_function_wrapper_type.method_wrapper()(
        instance)  # type: ignore
      # If this function happens to be assigned as a member of the instance,
      # without being a proper 'member function', then we get a non-None
      # instance on an unbound function.
      wrapt_wrapped_func = getattr(wrapped, '__func__', wrapped)
    else:
      wrapper = enact_function_wrapper_type()
      wrapt_wrapped_func = wrapped
    assert wrapt_wrapped_func == wrapper.wrapper_function(), (
      f'Unexpected wrapped function passed in by wrapt: '
      f'{wrapt_wrapped_func} != {wrapper.wrapper_function()}')
    return wrapper(call_args)  # type: ignore
  return wrapt.decorator(
    wrapper=_wrapper_fun,
    proxy=_create_wrapt_function_wrapper(enact_function_wrapper_type))


class FunctionWrapper(invocations.Invokable,
                      resource_registry.FunctionWrapper):
  """Invokable base class for synchronous functions."""

  def call(self, call_args: CallArgs):
    args, kwargs = call_args.to_python_args()
    return self.wrapped_function()(*args, **kwargs)


@dataclasses.dataclass
class MethodWrapper(invocations.Invokable,
                    resource_registry.MethodWrapper):
  """Invokable base class for synchronous methods."""
  instance: Any

  def call(self, call_args: CallArgs):
    bound = resource_registry.MethodType(self.wrapped_function(),
                                         self.get_instance())
    args, kwargs = call_args.to_python_args()
    return bound(*args, **kwargs)  # pylint: disable=not-callable


class AsyncFunctionWrapper(invocations.AsyncInvokable,
                           resource_registry.FunctionWrapper):
  """Invokable base class for synchronous functions."""

  async def call(self, call_args: CallArgs):
    args, kwargs = call_args.to_python_args()
    return await self.wrapped_function()(*args, **kwargs)


@dataclasses.dataclass
class AsyncMethodWrapper(invocations.AsyncInvokable,
                         resource_registry.MethodWrapper):
  """Invokable base class for synchronous methods."""
  instance: Any

  async def call(self, call_args: CallArgs):
    bound = MethodType(self.wrapped_function(),
                       self.get_instance())
    args, kwargs = call_args.to_python_args()
    return await bound(*args, **kwargs)  # pylint: disable=not-callable




def _create_async_function_wrapper(wrapped: Callable) -> (
    Type[AsyncFunctionWrapper]):
  """Create an enact async function wrapper for the given callable."""
  class _MethodWrapper(AsyncMethodWrapper):
    """The generated method wrapper."""
    _wrapper_function: Optional[Callable] = None

    @classmethod
    def wrapped_function(cls) -> Callable:
      """The associated function wrapper."""
      return wrapped

    @classmethod
    def wrapper_function(cls) -> Callable:
      """The associated function wrapper."""
      assert cls._wrapper_function
      return cls._wrapper_function

  class _FunctionWrapper(AsyncFunctionWrapper):
    """The generated function wrapper."""
    _wrapper_function: Optional[Callable] = None

    @classmethod
    def wrapped_function(cls) -> Callable:
      """The function (not method) that is wrapped by this."""
      return wrapped

    @classmethod
    def wrapper_function(cls) -> Callable:
      """The associated function wrapper."""
      assert cls._wrapper_function
      return cls._wrapper_function

    @classmethod
    def method_wrapper(cls) -> Type[_MethodWrapper]:
      """The associated method wrapper."""
      return _MethodWrapper

  return _FunctionWrapper


def _create_sync_function_wrapper(fun: Callable) -> (
    Type[FunctionWrapper]):
  """Create an enact function wrapper for the given callable."""
  class _MethodWrapper(MethodWrapper):
    """The generated method wrapper"""
    _wrapper_function: Optional[Callable] = None

    @classmethod
    def wrapped_function(cls) -> Callable:
      """The associated wrapped function."""
      return fun

    @classmethod
    def wrapper_function(cls) -> Callable:
      """The associated wrapper function."""
      assert cls._wrapper_function
      return cls._wrapper_function

  class _FunctionWrapper(FunctionWrapper):
    """The generated function wrapper."""
    _wrapper_function: Optional[Callable] = None

    @classmethod
    def wrapped_function(cls) -> Callable:
      """The associated wrapped function."""
      return fun

    @classmethod
    def wrapper_function(cls) -> Callable:
      """The associated wrapper function."""
      assert cls._wrapper_function
      # This unfortunately binds the function to cls, yielding a
      # BoundFunctionWrapper. We don't want that here, so we undo
      # the binding and return the parent, which is a function wrapper.
      # pylint: disable=protected-access
      return cls._wrapper_function._self_parent  # type: ignore

    @classmethod
    def method_wrapper(cls) -> Type[_MethodWrapper]:
      """The associated method wrapper."""
      return _MethodWrapper

  return _FunctionWrapper


def _create_function_wrapper(fun: Callable) -> (
    Type[resource_registry.FunctionWrapper]):
  """Create a sync or async function wrapper for the callable."""
  function_wrapper_type: Union[
    Type[FunctionWrapper],
    Type[AsyncFunctionWrapper]]
  if inspect.iscoroutinefunction(fun):
    function_wrapper_type = _create_async_function_wrapper(fun)
  else:
    function_wrapper_type = _create_sync_function_wrapper(fun)
  # Override names and qualnames to match the wrapped function.
  for generated_type in (function_wrapper_type,
                         function_wrapper_type.method_wrapper()):
    generated_type.__module__ = fun.__module__
    generated_type.__name__ = fun.__name__
    generated_type.__qualname__ = fun.__qualname__
  function_wrapper_type.method_wrapper().__name__ += '__method_wrapper'
  return function_wrapper_type


def register(fun: C) -> C:
  """Decorator for registering a wrapped function."""
  if inspect.ismethod(fun):
    raise ValueError(
      'Bound methods (e.g., A().foo) cannot be registered. '
      'Please register the unbound function instead (e.g., A.foo)')
  wrapper_type = _create_function_wrapper(fun)
  wrapper_fun = _wrapt_wrapper(wrapper_type)(fun)

  # pylint: disable=protected-access
  wrapper_type._wrapper_function = wrapper_fun  # type: ignore
  wrapper_type.method_wrapper()._wrapper_function = wrapper_fun  # type: ignore

  resource_registry.register(wrapper_type)
  resource_registry.register(wrapper_type.method_wrapper())
  return wrapper_fun

@resource_registry.register
@dataclasses.dataclass
class Call(invocations.Invokable[CallArgs, Any]):
  """An invokable representing a function call to an unregistered function."""
  function_name: str

  def __post_init__(self):
    """"""
    self._callable: Optional[Callable] = None

  @classmethod
  def create(cls, function_name: str, fun: Callable) -> 'Call':
    """Create a new call, setting the callable."""
    call = Call(function_name)
    call._callable = fun
    return call

  def call(self, resource: CallArgs) -> Any:
    """Call the wrapped function."""
    if not self._callable:
      raise invocations.InvocationError(
        'Cannot repeat a call of an unregistered function. '
        'Please make sure your top-level function is registered.')
    args, kwargs = resource.to_python_args()
    return self._callable(*args, **kwargs)


def get_wrapper_type(fun: Callable) -> Optional[
    resource_registry.FunctionWrapper]:
  """Return the CallableInvokable of this function, if it has one."""
  return getattr(fun, '_enact_function_wrapper_type', None)


def invoke(
    fun: Callable,
    args: Sequence=(),
    kwargs: Optional[Mapping[str, Any]]=None,
    replay_from: Optional[invocations.Invocation]=None,
    exception_override: invocations.ExceptionOverride=lambda _: None,
    raise_on_errors: Tuple[Type[Exception], ...]=(
      invocations.InvocationError, interfaces.FrameworkError),
    wrap_exceptions: bool=False,
    strict: bool=True,
    commit: bool=True):
  """Invoke an invokable, wrapped function or python callable.

  Args:
    fun: A callable to invoke.
    args: The args to pass to callable.
    kwargs: The kwargs to pass to callable.
    replay_from: The invocation to replay from.
    exception_override: A function that takes an exception and returns either
      None or a resource to override the exception with.
    raise_on_errors: Which errors should raise beyond the invocation.
    wrap_exceptions: Whether to wrap non-resource exceptions in NativeException.
      If False, non-resource exceptions raised during invocation will be
      re-raised.
    strict: If replaying, whether to replay strictly (expecting the same calls).
    commit: Whether to commit the invocation after invoking.
  """
  kwargs = kwargs or {}
  try:
    wrapped = resource_registry.wrap(fun)  # Try wrapping the callable.
  except interfaces.FieldTypeError:
    wrapped = None
  call_args: Optional[references.Ref] = None
  if wrapped:
    if not callable(wrapped):
      raise TypeError(
        f'The wrapper {wrapped} of {fun} is not callable.')
    fun = wrapped
  if isinstance(fun, invocations.InvokableBase):
    invokable = fun
    if fun.get_input_type() == CallArgs:
      call_args = references.commit(CallArgs.from_python_args(
        *args, **kwargs))
    elif isinstance(fun, (resource_registry.FunctionWrapper,
                          resource_registry.MethodWrapper)):
      call_args = references.commit(CallArgs.from_python_args(
        *args, **kwargs))
    else:
      if len(args) > 1 or kwargs:
        raise ValueError(
          'Invokables can only be invoked with up to one arg and no kwargs.')
      if args:
        call_args = references.commit(args[0])
      else:
        call_args = references.commit(None)
  else:
    call_args = references.commit(
      CallArgs.from_python_args(*args, **kwargs))
    invokable = Call.create(f'{fun.__module__}.{fun.__qualname__}', fun)

  invocation = invokable.invoke(
    call_args,
    replay_from=replay_from,
    exception_override=exception_override,
    raise_on_errors=raise_on_errors,
    wrap_exceptions=wrap_exceptions,
    strict=strict,
    # Skip commit, since we may throw away the outermost invocation:
    commit=False)

  # Remove the extraneous Call invocation if the called object
  # was a registered python function.
  if not isinstance(fun, invocations.InvokableBase):
    invokable_wrapper = get_wrapper_type(fun)
    if invokable_wrapper:
      children = list(invocation.get_children())
      assert len(children) == 1, (
        f'Expected one child containing the CallableInvokable associated '
        f'with {fun}. Got {len(children)} children instead.')
      invocation = children[0]
  elif commit:
    references.commit(invocation)

  return invocation

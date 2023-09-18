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

import abc
import inspect
from typing import (
  Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar,
  Union, cast)
import wrapt  # type: ignore
import dataclasses

from enact import interfaces
from enact import invocations
from enact import references
from enact import resource_registry
from enact import resources


@resource_registry.register
@dataclasses.dataclass
class CallArgs(resources.Resource):
  """Wrapper for call arguments."""
  args: List[interfaces.FieldValue]
  kwargs: Dict[str, interfaces.FieldValue]

  def to_python_args(self) -> Tuple[List, Dict[str, Any]]:
    """Transform into python call args."""
    return (resource_registry.from_field_value(self.args),
            resource_registry.from_field_value(self.kwargs))

  @classmethod
  def from_python_args(cls, *args, **kwargs) -> 'CallArgs':
    """Get CallArgs from python call args."""
    return cls(
      cast(List[interfaces.FieldValue],
           resource_registry.to_field_value(list(args))),
      cast(Dict[str, interfaces.FieldValue],
           resource_registry.to_field_value(kwargs)))


invocations.typed_invokable(CallArgs, None)
@dataclasses.dataclass(frozen=True)
class CallableWrapper(
    resources.ImmutableResource,
    invocations.InvokableBase):
  """An immutable wrapper for a non-member callable."""

  def call(self, arg: CallArgs) -> Any:
    args, kwargs = arg.to_python_args()
    return self.wrapped()(*args, **kwargs)

  @classmethod
  @abc.abstractmethod
  def wrapped(cls) -> Callable:
    """Return the wrapped callable."""
    raise NotImplementedError()

  def __str__(self):
    """Return a string representation."""
    return self.wrapped().__name__


@dataclasses.dataclass
class MemberCallableWrapper(invocations.Invokable[CallArgs, Any]):
  """Invokable wrapper for a python callable."""
  instance: Any

  @classmethod
  @abc.abstractmethod
  def wrapped(cls) -> Callable:
    """Return the wrapped callable."""
    raise NotImplementedError()

  def call(self, resource: CallArgs):
    """Call the wrapped function."""
    args, kwargs = resource.to_python_args()
    wrapped_callable = self.wrapped()
    if self.instance:
      args = [self.instance, *args]
    result = wrapped_callable(*args, **kwargs)
    return result

  def __str__(self):
    """Return a string representation."""
    if self.instance:
      return f'{self.instance}.{self.wrapped().__name__}'
    else:
      return self.wrapped().__name__


C = TypeVar('C', bound=Callable)


class FunctionWrapper(wrapt.FunctionWrapper):
  """Wrapper for a python callable."""

  def __init__(
      self,
      wrapped: Callable,
      wrapper: Callable,
      wrapper_type_or_instance: Union[
        Type[MemberCallableWrapper], CallableWrapper]):
    super().__init__(wrapped, wrapper)
    self._self_enact_callable_wrapper = wrapper_type_or_instance


def _is_instance_or_class_method(fun: Callable) -> bool:
  """Tries to establish whether this is an instance or class method."""
  params = inspect.signature(fun).parameters
  if len(params) == 0:
    return False
  first_param_name, *_ = params
  if first_param_name in ('self', 'cls'):
    return True
  return False


def register(fun: C) -> C:
  """Decorator for registering a wrapped function."""
  if _is_instance_or_class_method(fun):
    class _MemberWrapper(MemberCallableWrapper):
      """Member or class wrapper."""
      @classmethod
      def wrapped(cls):
        return fun
    _MemberWrapper.__module__ = fun.__module__
    _MemberWrapper.__name__ = fun.__name__
    _MemberWrapper.__qualname__ = fun.__qualname__
    resource_registry.register(_MemberWrapper)

    def _wrapper_fun(unused_wrapped, instance, args, kwargs):
      call_args = CallArgs.from_python_args(*args, **kwargs)
      wrapper = _MemberWrapper(instance)
      return wrapper(call_args)
    function_wrapper = FunctionWrapper(
      fun, _wrapper_fun, _MemberWrapper)
  else:
    class _Wrapper(CallableWrapper):
      """Member or class wrapper."""
      @classmethod
      def wrapped(cls):
        return fun
    _Wrapper.__module__ = fun.__module__
    _Wrapper.__name__ = fun.__name__
    _Wrapper.__qualname__ = fun.__qualname__
    invocations.typed_invokable(CallArgs, None)(_Wrapper)

    wrapper = _Wrapper()
    def _wrapper_fun(unused_wrapped, instance, args, kwargs):
      assert not instance
      call_args = CallArgs.from_python_args(*args, **kwargs)
      return wrapper(call_args)
    function_wrapper = FunctionWrapper(
      fun, _wrapper_fun, wrapper)
    resource_registry.register_instance_wrapper(
      function_wrapper, wrapper)

  return function_wrapper


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

  def call(self, resource: CallArgs) -> interfaces.ResourceBase:
    """Call the wrapped function."""
    if not self._callable:
      raise invocations.InvocationError(
        'Cannot repeat a call of an unregistered function. '
        'Please make sure your top-level function is registered.')
    args, kwargs = resource.to_python_args()
    result = self._callable(*args, **kwargs)
    return resource_registry.wrap(result)


def get_wrapper_instance_or_type(fun: Callable) -> Optional[
  Union[Type[MemberCallableWrapper], CallableWrapper]]:
  """See if this is a wrapped function and if so, return the CallableWrapper."""
  invokable_wrapper = getattr(fun, '_self_enact_callable_wrapper', None)
  if not invokable_wrapper:
    # For a bound function, our FunctionWrapper is wrapped again in a
    # BoundFunctionWrapper, which is stored at '_self_parent'.
    parent = getattr(fun, '_self_parent', None)
    if parent:
      invokable_wrapper = getattr(parent, '_self_enact_callable_wrapper', None)
  return invokable_wrapper


def invoke(
    fun: Callable,
    args: Sequence=(),
    kwargs: Optional[Mapping[str, Any]]=None,
    replay_from: Optional[invocations.Invocation]=None,
    exception_override: invocations.ExceptionOverride=lambda _: None,
    raise_on_errors: Tuple[Type[Exception], ...]=(
      invocations.InvocationError, interfaces.FrameworkError),
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
    else:
      if len(args) > 1 or kwargs:
        raise ValueError(
          'Invokables can only be invoked with up to one arg and no kwargs.')
      if args:
        call_args = references.commit(args[0])
      else:
        call_args = references.commit(None)
  else:
    call_args = references.commit(CallArgs.from_python_args(*args, **kwargs))

    invokable_wrapper = get_wrapper_instance_or_type(fun)
    if invokable_wrapper:
      # This helps avoid having a useless Call invokable in the invocation if
      # the type is wrapped.
      instance = getattr(fun, '__self__', None)
      invokable = invokable_wrapper(instance)  # type: ignore
    else:
      invokable = Call.create(f'{fun.__module__}.{fun.__qualname__}', fun)

  return invokable.invoke(
    call_args,
    replay_from=replay_from,
    exception_override=exception_override,
    raise_on_errors=raise_on_errors,
    strict=strict,
    commit=commit)

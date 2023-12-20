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

"""Thread-safe context management."""

import contextlib
import contextvars
from typing import Callable, Dict, Optional, Type, TypeVar, cast


_context_vars: Dict[Type['Context'],
                    contextvars.ContextVar[Optional['Context']]] = {}


class ContextError(Exception):
  """Error raised when there is a problem with the context."""


class NoActiveContext(ContextError):
  """Raised when there is no active context."""

ContextT = TypeVar('ContextT', bound='Context')
ContextSuperT = TypeVar('ContextSuperT', bound='Context')


class Context:
  """A thread-aware context superclass."""

  def __init__(self: ContextT):
    """Creates a new context."""
    self._token: Optional[contextvars.Token[Optional[ContextT]]] = None

  @classmethod
  def _get_context_var(cls: Type[ContextT]) -> (
      contextvars.ContextVar[Optional[ContextT]]):
    """Returns the context var for this type."""
    try:
      return cast(
        contextvars.ContextVar[Optional[ContextT]], _context_vars[cls])
    except KeyError as key_error:
      raise ContextError(
        f'Context {cls} not registered. A context class must be registered '
        f'with the "@register" decorator.') from key_error

  @classmethod
  def permissive_initialization(cls: Type[ContextT]) -> bool:
    """Returns whether the class has permissive initialization.

    Contexts use contextvars for tracking whether an execution is in-context or
    out-of-context. Contextvars have special handling in threads and async
    executions, e.g., they are thread-local and hence starting a new thread will
    erase the context stack. The Context class can detect when this happens,
    since it will store information in the main thread during registration. This
    variable controls how it handles such situations:

    If permissive_initialization is set to True, the context will automatically
    initialize to an empty stack when it detects that it is being run in an
    uninitialized context. If set to False, then the class will trigger an error
    and ask the developer to specify whether they want to copy context or start
    a fresh calling context. This may be done using the 'with_current_contexts'
    or with_new_contexts function decorators.

    Overriding this function to return False is useful in situation where
    accidentally operating in a fresh context is a problem, e.g., when
    implementing contexts that deal with encryption.
    """
    return True

  @classmethod
  @contextlib.contextmanager
  def top_level(cls: Type[ContextT]):
    """Returns a context manager to execute code in a top-level context."""
    context_var = cls._get_context_var()
    token = context_var.set(None)
    try:
      yield
    finally:
      context_var.reset(token)


  @classmethod
  def get_current(cls: Type[ContextT]) -> Optional[ContextT]:
    """Returns the current context of this type or None."""
    context_var = cls._get_context_var()
    try:
      current_context = context_var.get()
    except LookupError as lookup_error:
      if cls.permissive_initialization():
        context_var.set(None)
        return None
      raise ContextError(
        f'Context {cls} not initialized. If running inside a thread, make sure '
        f'to annotate the thread function with either the '
        f'"@with_current_contexts" or "@with_new_contexts" decorator.'
        ) from lookup_error
    if current_context is not None and not isinstance(current_context, cls):
      raise LookupError(
        f'Current context {current_context} is not of type {cls}.')
    return current_context

  @classmethod
  def current(cls: Type[ContextT]) -> ContextT:
    """Returns the current context of this type or raises an error."""
    context = cls.get_current()
    if context is None:
      raise NoActiveContext(f'No context of type {cls.__qualname__} is active.')
    return context

  def __enter__(self: ContextT) -> ContextT:
    """Enters the context."""
    context_var = self._get_context_var()
    self.get_current()  # Raise an error if the context is not initialized.
    self.enter()
    self._token = context_var.set(self)
    return self

  def __exit__(self: ContextT, exc_type, exc_value, traceback):
    """Exits the context."""
    context_var = self._get_context_var()
    assert self._token
    context_var.reset(self._token)
    self.exit()
    self._token = None

  def enter(self: ContextT):
    """Overridable on context entry."""

  def exit(self: ContextT):
    """Overridable on context exit."""

def register(cls: Type[ContextT]) -> Type[ContextT]:
  """Registers a context class."""
  assert cls not in _context_vars, (
    f'Context class already registered: {cls}')
  ctx_var: contextvars.ContextVar[Optional[Context]] = (
    contextvars.ContextVar(cls.__qualname__))
  ctx_var.set(None)
  _context_vars[cls] = ctx_var
  return cls


def register_to_superclass(superclass: Type[ContextSuperT]) -> Callable[
    [Type[ContextT]], Type[ContextT]]:
  def _register(cls: Type[ContextT]) -> Type[ContextT]:
    """Registers a context class."""
    assert cls not in _context_vars, (
      f'Context class already registered: {cls}')
    assert superclass in _context_vars, (
      f'Superclass {superclass} not registered.')
    assert issubclass(cls, superclass), (
      f'Context class {cls} must be a subclass of {superclass}.')
    _context_vars[cls] = _context_vars[superclass]
    # MyPy can't handle intersection types, which is what would be required
    # here. We cast to Type[ContextT] to make it happy.
    return cast(Type[ContextT], cls)
  return _register


R = TypeVar('R')


def _with_contexts(
    ctx: contextvars.Context,
    fun: Callable[..., R], *args, **kwargs) -> R:
  """Temporarily set _context_vars from ctx and run function."""
  # We don't use ctx.run here, since we don't want to modify how unmanaged
  # ContextVar objects are propagated.
  tokens: Dict[contextvars.ContextVar, contextvars.Token] = {}
  for context_var in _context_vars.values():
    assert context_var in ctx
    tokens[context_var] = context_var.set(ctx[context_var])
  try:
    return fun(*args, **kwargs)
  finally:
    for context_var, token in tokens.items():
      context_var.reset(token)


def with_current_contexts(fun: Callable) -> Callable:
  """Decorator that runs the function in all current contexts."""
  ctx = contextvars.copy_context()
  return lambda *args, **kwargs: _with_contexts(ctx, fun, *args, **kwargs)


def with_new_contexts(fun: Callable) -> Callable:
  """Decorator that runs the function with new contexts."""
  ctx = contextvars.Context()
  def _set_to_none():
    for context_var in _context_vars.values():
      context_var.set(None)
  ctx.run(_set_to_none)  # Set the managed contextvars to None.
  return lambda *args, **kwargs: _with_contexts(ctx, fun, *args, **kwargs)

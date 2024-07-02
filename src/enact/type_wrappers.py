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

"""Default resource wrappers for non-field types."""

import dataclasses
import types as python_types
import sys
from typing import Type


from enact import resources
from enact import registration
from enact import types


@registration.register
@dataclasses.dataclass
class TupleWrapper(resources.TypeWrapper[tuple]):
  """Wrapper for tuples."""
  value: list

  @classmethod
  def is_immutable(cls) -> bool:
    return True

  @classmethod
  def wrapped_type(cls) -> Type[tuple]:
    return tuple

  @classmethod
  def wrap(cls, value: tuple) -> 'TupleWrapper':
    """Wrap a tuple value directly."""
    assert isinstance(value, tuple), (
      f'Cannot wrap value of type {type(value)} with wrapper {cls}.')
    return cls(list(value))

  def unwrap(self) -> tuple:
    """Unwrap the tuple."""
    return tuple(self.value)


@registration.register
@dataclasses.dataclass
class SetWrapper(resources.TypeWrapper[set]):
  """Wrapper for sets."""
  value: list

  @classmethod
  def wrapped_type(cls) -> Type[set]:
    return set

  @classmethod
  def wrap(cls, value: set) -> 'SetWrapper':
    """Wrap a set value directly."""
    assert isinstance(value, set), (
      f'Cannot wrap value of type {type(value)} with wrapper {cls}.')
    return cls(list(value))

  def unwrap(self) -> set:
    """Unwrap the set."""
    return set(self.value)


@registration.register
@dataclasses.dataclass
class TypeDescriptorWrapper(resources.TypeWrapper[types.TypeDescriptor]):
  """Wrapper for TypeDescriptors."""
  json: types.Json

  @classmethod
  def wrapped_type(cls) -> Type[types.TypeDescriptor]:
    return types.TypeDescriptor

  @classmethod
  def wrap(cls, value: types.TypeDescriptor) -> 'TypeDescriptorWrapper':
    """Wrap a type descriptor."""
    assert isinstance(value, types.TypeDescriptor), (
      f'Cannot wrap value of type {type(value)} with wrapper {cls}.')
    return cls(value.to_json())

  def unwrap(self) -> types.TypeDescriptor:
    """Unwrap the type descriptor."""
    return types.TypeDescriptor.from_json(self.json)


@registration.register
@dataclasses.dataclass
class ModuleWrapper(resources.TypeWrapper[python_types.ModuleType]):
  """Wrapper for python modules."""
  # TODO: Figure out a way to track type dependencies of a module wrapper.
  name: str

  @classmethod
  def wrapped_type(cls) -> Type[python_types.ModuleType]:
    """Returns the type of the wrapped resource."""
    # We could use any module here instead of sys, we just want to access
    # # <class 'module'>.
    return type(sys)

  @classmethod
  def wrap(cls, value: python_types.ModuleType) -> 'ModuleWrapper':
    """Wrap a python module."""
    assert isinstance(value, python_types.ModuleType), (
      f'Cannot wrap value of type {type(value)} with wrapper {cls}.')
    return cls(value.__name__)

  def unwrap(self) -> python_types.ModuleType:
    """Unwrap the module."""
    if self.name not in sys.modules:
      raise ValueError(
        f'Module {self.name} not found in sys.modules. Please make sure it '
        f'is imported before attempting to check out its reference.')
    return sys.modules[self.name]

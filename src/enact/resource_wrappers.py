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
import io
from typing import Type

import numpy as np
import PIL.Image

from enact import resources
from enact import registration


@registration.register
@dataclasses.dataclass
class TupleWrapper(resources.ResourceWrapper[tuple]):
  """Wrapper for tuples."""
  value: list

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
class SetWrapper(resources.ResourceWrapper[set]):
  """Wrapper for tuples."""
  value: list

  @classmethod
  def wrapped_type(cls) -> Type[set]:
    return set

  @classmethod
  def wrap(cls, value: set) -> 'SetWrapper':
    """Wrap a tuple value directly."""
    assert isinstance(value, set), (
      f'Cannot wrap value of type {type(value)} with wrapper {cls}.')
    return cls(list(value))

  def unwrap(self) -> set:
    """Unwrap the tuple."""
    return set(self.value)


@registration.register
@dataclasses.dataclass
class NPArrayWrapper(resources.ResourceWrapper):
  """A resource wrapper for numpy arrays."""
  value: bytes

  @classmethod
  def wrapped_type(cls) -> Type[np.ndarray]:
    """Returns the type of the wrapped resource."""
    return np.ndarray

  @classmethod
  def wrap(cls, value: np.ndarray) -> 'NPArrayWrapper':
    """Returns a wrapper for the resource."""
    bytes_io = io.BytesIO()
    np.save(bytes_io, value)
    return NPArrayWrapper(bytes_io.getvalue())

  def unwrap(self) -> np.ndarray:
    """Returns the wrapped resource."""
    bytes_io = io.BytesIO(self.value)
    return np.load(bytes_io)


@registration.register
@dataclasses.dataclass
class PILImageWrapper(resources.ResourceWrapper):
  """An resource wrapper for PIL images."""
  value: bytes

  @classmethod
  def wrapped_type(cls) -> 'Type[PIL.Image.Image]':
    return PIL.Image.Image

  @classmethod
  def wrap(cls, value: PIL.Image.Image) -> 'PILImageWrapper':
    """Returns a wrapper for the resource."""
    bytes_io = io.BytesIO()
    value.save(bytes_io, format='png')
    return PILImageWrapper(bytes_io.getvalue())

  def unwrap(self) -> PIL.Image.Image:
    """Returns the wrapped resource."""
    bytes_io = io.BytesIO(self.value)
    return PIL.Image.open(bytes_io)

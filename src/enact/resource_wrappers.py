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
from typing import Type, TypeVar

import numpy as np
import PIL.Image

from enact import resources
from enact import registration


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
class NPArrayWrapper(resources.TypeWrapper):
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

NPFloatWrapperT = TypeVar('NPFloatWrapperT', bound='NPFloatWrapper')

@dataclasses.dataclass
class NPFloatWrapper(resources.TypeWrapper):
  """Base class for resource wrappers for numpy float scalars."""
  value: float

  @classmethod
  def wrap(cls: Type[NPFloatWrapperT], value: np.float32) -> NPFloatWrapperT:
    """Returns a wrapper for the resource."""
    return cls(value=float(value))

  def unwrap(self) -> np.ndarray:
    """Returns the wrapped resource."""
    return self.wrapped_type()(self.value)


@registration.register
class NPFloat16Wrapper(NPFloatWrapper):
  """Resource wrapper for numpy float16 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.float16]:
    return np.float16


@registration.register
class NPFloat32Wrapper(NPFloatWrapper):
  """Resource wrapper for numpy float32 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.float32]:
    return np.float32


@registration.register
class NPFloat64Wrapper(NPFloatWrapper):
  """Resource wrapper for numpy float64 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.float64]:
    return np.float64


@registration.register
@dataclasses.dataclass
class PILImageWrapper(resources.TypeWrapper):
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

  @classmethod
  def set_wrapped_value(cls, target: PIL.Image.Image, src: PIL.Image.Image):
    """Set a wrapped value target to correspond to source."""
    target.resize(src.size)
    target.paste(src, (0, 0))

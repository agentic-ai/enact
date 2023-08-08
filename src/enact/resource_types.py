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

"""Basic resource types."""

import io
from typing import Iterable, Mapping, Type, TypeVar

import numpy as np
import PIL.Image

from enact import interfaces
from enact import resource_registry


C = TypeVar('C', bound='_ResourceMixin')


class _ResourceMixin(interfaces.ResourceBase):
  """A mix-in for annotating simple types as resources."""

  @classmethod
  def field_value_type(cls) -> Type:
    """Return the field value type."""
    raise NotImplementedError()

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    return ('value',)

  def field_values(self) -> Iterable[interfaces.FieldValue]:
    """Return a list of field values, aligned with field_names."""
    return (self.field_value_type()(self),)

  @classmethod
  def from_fields(cls: Type[C],
                  field_dict: Mapping[str, interfaces.FieldValue]) -> C:
    """Constructs the resource from a field dictionary."""
    value = field_dict['value']
    return cls(value)  # type: ignore


@resource_registry.register
class Int(int, _ResourceMixin):

  @classmethod
  def field_value_type(cls) -> Type:
    """Return the field value type."""
    return int


@resource_registry.register
class Float(float, _ResourceMixin):

  @classmethod
  def field_value_type(cls) -> Type:
    """Return the field value type."""
    return float



@resource_registry.register
class Str(str, _ResourceMixin):

  @classmethod
  def field_value_type(cls) -> Type:
    """Return the field value type."""
    return str


@resource_registry.register
class Bytes(bytes, _ResourceMixin):

  @classmethod
  def field_value_type(cls) -> Type:
    """Return the field value type."""
    return bytes


@resource_registry.register
class List(list, _ResourceMixin):

  @classmethod
  def field_value_type(cls) -> Type:
    """Return the field value type."""
    return list


N = TypeVar('N', bound='NPArray')


@resource_registry.register
class NPArray(interfaces.ResourceBase):
  """A resource wrapper for numpy arrays."""

  def __init__(self, value: np.ndarray):
    """Initialize the resource from an array."""
    self.value = value

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    return ('value',)

  def field_values(self) -> Iterable[interfaces.FieldValue]:
    """Return a list of field values, aligned with field_names."""
    bytes_io = io.BytesIO()
    np.save(bytes_io, self.value)
    return (bytes_io.getvalue(),)

  @classmethod
  def from_fields(cls: Type[N],
                  field_dict: Mapping[str, interfaces.FieldValue]) -> N:
    """Constructs the resource from a field dictionary."""
    value = field_dict['value']
    assert isinstance(value, bytes)
    bytes_io = io.BytesIO(value)
    array = np.load(bytes_io)
    return cls(array)  # type: ignore

  def __str__(self):
    """Return a string representation of the resource."""
    return str(self.value)


I = TypeVar('I', bound='Image')


@resource_registry.register
class Image(interfaces.ResourceBase):
  """An image resource."""

  def __init__(self, value: PIL.Image.Image):
    """Initialize the resource."""
    self.value = value

  @property
  def data(self) -> bytes:
    """Return the image data."""
    bytes_io = io.BytesIO()
    self.value.save(bytes_io, format='png')
    data = bytes_io.getvalue()
    return data

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    return ('data',)

  def field_values(self) -> Iterable[interfaces.FieldValue]:
    """Return a list of field values, aligned with field_names."""
    return (self.data,)

  @classmethod
  def from_fields(cls: Type[I],
                  field_dict: Mapping[str, interfaces.FieldValue]) -> I:
    """Constructs the resource from a field dictionary."""
    value = field_dict['data']
    assert isinstance(value, bytes)
    return cls(PIL.Image.open(io.BytesIO(value)))

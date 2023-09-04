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

"""Dataclass-based resources."""

import abc
import dataclasses
from typing import Iterable, Mapping, Type, TypeVar

from enact import digests
from enact import interfaces


FieldValue = interfaces.FieldValue

C = TypeVar('C', bound='Resource')


@dataclasses.dataclass
class Resource(interfaces.ResourceBase):
  """Base class for dataclass-based resources.

  Subclasses must be registered with the @enact.register decorator in order
  to allow deserialization from references.
  """

  @classmethod
  def type_descr(cls) -> Mapping[str, interfaces.Json]:
    """Returns a unique identifier for the type."""
    descr = super().type_descr()
    assert isinstance(descr, dict)
    descr['digest'] = digests.type_digest(cls)
    return descr

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    return (f.name for f in dataclasses.fields(cls))

  def field_values(self) -> Iterable[FieldValue]:
    """Return a list of field values, aligned with field_names."""
    return (getattr(self, f) for f in self.field_names())

  @classmethod
  def from_fields(cls: Type[C],
                  field_values: Mapping[str, FieldValue]) -> C:
    """Constructs the resource from a value dictionary."""
    return cls(**field_values)

  def set_from(self: C, other: C):
    """Sets the fields of this resource from another resource.

    Implementation of set_from is required to support replays of invokable
    resources that change their internal state during execution.
    """
    if not type(self) is type(other):  # pylint: disable=unidiomatic-typecheck
      raise TypeError(f'Cannot set_from {type(other)} into {type(self)}.')
    copy = other.deep_copy_resource()
    for field in dataclasses.fields(self):
      setattr(self, field.name, getattr(copy, field.name))


WrappedT = TypeVar('WrappedT')
WrapperT = TypeVar('WrapperT', bound='ResourceWrapper')


@dataclasses.dataclass
class ResourceWrapper(interfaces.ResourceWrapperBase[WrappedT], Resource):
  """Base class for dataclass-based resource wrappers."""

  @classmethod
  @abc.abstractmethod
  def wrapped_type(cls) -> Type[WrappedT]:
    """Returns the type of the wrapped value."""
    raise NotImplementedError()

  @classmethod
  @abc.abstractmethod
  def wrap(cls: Type[WrapperT], value: WrappedT) -> WrapperT:
    """Wrap a value."""
    raise NotImplementedError()

  @abc.abstractmethod
  def unwrap(self) -> WrappedT:
    """Wrap a value."""
    raise NotImplementedError()

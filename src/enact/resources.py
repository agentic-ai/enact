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

from enact import interfaces
from enact import resource_registry


FieldValue = interfaces.FieldValue

C = TypeVar('C', bound='_Resource')


class _Resource(interfaces.ResourceBase):
  """Base class for Resource and FrozenResource."""

  @classmethod
  def field_names(cls) -> Iterable[str]:
    """Returns the names of the fields of the resource."""
    return (f.name for f in dataclasses.fields(cls))  # type: ignore

  def field_values(self) -> Iterable[FieldValue]:
    """Return a list of field values, aligned with field_names."""
    return (resource_registry.to_field_value(
      getattr(self, f)) for f in self.field_names())

  @classmethod
  def from_fields(cls: Type[C],
                  field_values: Mapping[str, FieldValue]) -> C:
    """Constructs the resource from a value dictionary."""
    field_values = {k: resource_registry.from_field_value(v)
                    for k, v in field_values.items()}
    return cls(**field_values)

ResourceT = TypeVar('ResourceT', bound='Resource')

@dataclasses.dataclass
class Resource(_Resource):
  """Base class for dataclass-based resources.

  Subclasses must be registered with the @enact.register decorator in order
  to allow deserialization from references.
  """
  def set_from(self, other: interfaces.ResourceBase):
    """Sets the fields of this resource from another resource.

    Implementation of set_from is required to support replays of invokable
    resources that change their internal state during execution.
    """
    if not type(self) == type(other):  # pylint: disable=unidiomatic-typecheck
      raise TypeError(f'Cannot set_from {type(other)} into {type(self)}.')
    copy = other.deepcopy_resource()
    for field in dataclasses.fields(self):
      self_field = getattr(self, field.name)
      target = getattr(other, field.name)
      if self_field is target:
        continue
      if type(self_field) == type(target):  # pylint: disable=unidiomatic-typecheck
        if isinstance(self_field, interfaces.ResourceBase):
          # In cases where we have compatible field values, we can set them
          # directly without creating a new instance. Note that this may lead to
          # unexpected behavior in cases where a new object is explicitly
          # allocated (and later checked, e.g., via ID). See
          # https://github.com/agentic-ai/enact/issues/54
          self_field.set_from(target)
          continue
        else:
          wrapper_type = resource_registry.Registry.get().get_type_wrapper(
            type(self_field))
          if wrapper_type and not wrapper_type.is_immutable():
            wrapper_type.set_wrapped_value(self_field, target)
            continue
      setattr(self, field.name, getattr(copy, field.name))


@dataclasses.dataclass(frozen=True)
class ImmutableResource(_Resource):
  """Base class for immutable dataclass-based resources.

  Subclasses must be registered with the @enact.register decorator in order
  to allow deserialization from references.
  """
  def set_from(self, other: interfaces.ResourceBase):
    """Sets the fields of this resource from another resource.

    Implementation of set_from is required to support replays of invokable
    resources that change their internal state during execution.
    """
    if self != other:
      raise TypeError(f'Cannot call set_from on immutable resource {self}.')


WrappedT = TypeVar('WrappedT')
WrapperT = TypeVar('WrapperT', bound='TypeWrapper')


@dataclasses.dataclass
class TypeWrapper(interfaces.TypeWrapperBase[WrappedT], Resource):
  """Base class for dataclass-based type wrappers."""

  @classmethod
  @abc.abstractmethod
  def wrapped_type(cls) -> Type[WrappedT]:
    """Returns the type wrapped by this TypeWrapper."""
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

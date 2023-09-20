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

"""A unified registration function."""

import inspect
from typing import Callable, Optional, Tuple, Type, TypeVar, Union, cast

from enact import function_wrappers
from enact import interfaces
from enact import invocations
from enact import resource_registry


Registerable = Union[
  Type[interfaces.ResourceBase],
  Callable]

RegisterableT = TypeVar('RegisterableT', bound=Registerable)


def _to_python_type(t) -> Optional[Type]:
  """Try to extract a python type from a type annotation."""
  if t is None:
    return type(None)
  if t is inspect.Signature.empty:
    return None
  if isinstance(t, type):
    return t
  if hasattr(t, '__origin__'):  # Handle generics
    return t.__origin__
  return None


def _infer_types(invokable: Type[invocations._InvokableBase]) -> Tuple[
    Optional[Type], Optional[Type]]:
  """Infer types from the invokable."""
  signature = inspect.signature(invokable.call)
  output_type = _to_python_type(signature.return_annotation)
  params = signature.parameters
  if 'self' not in params:
    raise TypeError(
      'Invokable "call" function must have a "self" parameter.')
  if len(params) > 2:  # We count self here.
    raise TypeError(
      'Invokable "call" function must have at most one non-self parameter.')
  if len(params) == 1:
    input_type: Optional[Type] = type(None)
  else:
    (_, param) = params.values()
    if param.kind not in (param.POSITIONAL_ONLY,
                          param.POSITIONAL_OR_KEYWORD):
      raise TypeError(
        'Invokable "call" function must have a single '
        'non-self positional parameter.')
    input_type = _to_python_type(param.annotation)
  return input_type, output_type


def register(registerable: RegisterableT) -> RegisterableT:
  """Register a resource class, wrapper or python callable."""
  if (isinstance(registerable, type) and
      issubclass(registerable, invocations._InvokableBase)):  # pylint: disable=protected-access
    input_type, output_type = _infer_types(registerable)
    return cast(RegisterableT, invocations.typed_invokable(
      input_type, output_type)(registerable))
  elif (isinstance(registerable, type) and
      issubclass(registerable, interfaces.ResourceBase)):
    return cast(RegisterableT, resource_registry.register(registerable))
  elif callable(registerable):
    if isinstance(registerable, type):
      raise TypeError(
        f'Cannot register callable type {registerable} as a function. '
        f'Please inherit from Invokable to register callable types.')
    return cast(RegisterableT, function_wrappers.register(registerable))
  else:
    raise TypeError(f'Cannot register {registerable}')

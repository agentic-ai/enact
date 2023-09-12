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

from typing import Callable, Type, TypeVar, Union, cast

from enact import function_wrappers, interfaces, resource_registry


Registerable = Union[Type[interfaces.ResourceBase], Callable]


RegisterableT = TypeVar('RegisterableT', bound=Registerable)


def register(registerable: RegisterableT) -> RegisterableT:
  """Register a resource class, wrapper or python callable."""
  if (isinstance(registerable, type) and
      issubclass(registerable, interfaces.ResourceBase)):
    return cast(RegisterableT, resource_registry.register(registerable))
  elif callable(registerable):
    return cast(RegisterableT, function_wrappers.register(registerable))
  else:
    raise TypeError(f'Cannot register {registerable}')

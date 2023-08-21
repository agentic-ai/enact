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

"""FastAPI bindings for enact invokables."""

import dataclasses
import inspect
from typing import List, Optional, Type

import enact


import fastapi

def get(
  app, route: str, invokable: enact.Invokable,
  store: enact.Store):
  """Exposes an envokable as a new get API endpoint.

  Args:
    app: The FastAPI app instance to modify.
    route: The route for the new API.
    invokable: The Invokable instance that the API is wrapping.
    store: The store instance that will hold the results of the invocation.
  """
  _add_fast_api(app, route, invokable, store, ['GET'])

def post(
  app, route: str, invokable: enact.Invokable,
  store: enact.Store):
  """Exposes an envokable as a new post API endpoint.

  Args:
    app: The FastAPI app instance to modify.
    route: The route for the new API.
    invokable: The Invokable instance that the API is wrapping.
    store: The store instance that will hold the results of the invocation.
  """
  _add_fast_api(app, route, invokable, store, ['POST'])

def _check_resource_type(t: Optional[Type[enact.ResourceBase]], in_out: str):
  """Check that input or output type is set and not overly general."""
  if t is None:
    raise ValueError(f'{in_out} type must be set on invokable, use the'
                     f'@enact.typed_invokable decorator')
  if t in (enact.Resource, enact.ResourceBase):
    raise ValueError(
      f'{in_out} type is overly general: {t}. FastAPI requires specific '
      f'types to be set for the request.')
  if issubclass(t, enact.NoneResource):
    return
  if not issubclass(t, enact.Resource):
    raise ValueError(
      f'{in_out} is not a subclass of enact.Resource: {t}. Only '
      f'dataclass based resource are supported by FastAPI.')
  dataclass_fields = set(f.name for f in dataclasses.fields(t))
  resource_fields = set(t.field_names())
  # Make sure that dataclass fields coincide with resource fields.
  if dataclass_fields != resource_fields:
    raise ValueError(
      f'{in_out} type {t} has dataclass fields {dataclass_fields} that do not '
      f'coincide with resource fields {resource_fields}. Using FastAPI '
      f'that all resource fields are declared as dataclass fields.')

def _check_invokable(invokable: enact.Invokable, methods: List[str]):
  """Check input and output type."""
  _check_resource_type(invokable.get_input_type(), 'Input')
  _check_resource_type(invokable.get_output_type(), 'Output')
  if 'GET' in methods and invokable.get_input_type() != enact.NoneResource:
    raise ValueError(
      'GET is only supported for invokables with input type NoneResource.')

def _add_fast_api(
  app, route: str, invokable: enact.Invokable,
  store: enact.Store, methods: List[str]):
  """Internal implementation of wrapping an Invokable via FastAPI"""
  def get_output(invocation: enact.Invocation):
    if not invocation.successful():
      raise fastapi.HTTPException(
        status_code=500, detail=str(invocation.get_raised()))
    return invocation.get_output()

  def journaled_call() -> enact.Resource:
    with store:
      return get_output(invokable.invoke())

  def journaled_call_input(data: enact.Resource) -> enact.Resource:
    if data is None:
      raise fastapi.HTTPException(status_code=400, detail='No data provided')
    with store:
      return get_output(invokable.invoke(enact.commit(data)))

  _check_invokable(invokable, methods)

  param = inspect.Parameter(
    'data', inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None,
    annotation=invokable.get_input_type())

  journaled_call_input.__signature__ = inspect.Signature(  # type: ignore
    parameters=[param], return_annotation=invokable.get_output_type())

  wrapper_fn = (journaled_call if
                invokable.get_input_type() == enact.NoneResource
                else journaled_call_input)

  app.router.add_api_route(
    route, wrapper_fn, name=type(invokable).__name__, # type: ignore[arg-type]
    description=invokable.__doc__, response_model=invokable.get_output_type(),
    methods=methods)

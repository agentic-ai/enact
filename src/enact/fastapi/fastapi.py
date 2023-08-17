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

import fastapi
from typing import List

import enact
from enact import invocations
from enact import references
from enact import resources

def get(
  app: fastapi.FastAPI, route: str, invokable: invocations.Invokable,
  store: references.Store):
  """Exposes an envokable as a new get API endpoint.

  Args:
    app: The FastAPI app instance to modify.
    route: The route for the new API.
    invokable: The Invokable instance that the API is wrapping.
    store: The store instance that will hold the results of the invocation.
  """
  _add_fast_api(app, route, invokable, store, ['GET'])

def post(
  app: fastapi.FastAPI, route: str, invokable: invocations.Invokable,
  store: references.Store):
  """Exposes an envokable as a new post API endpoint.

  Args:
    app: The FastAPI app instance to modify.
    route: The route for the new API.
    invokable: The Invokable instance that the API is wrapping.
    store: The store instance that will hold the results of the invocation.
  """
  _add_fast_api(app, route, invokable, store, ['POST'])

def _add_fast_api(
  app: fastapi.FastAPI, route: str, invokable: invocations.Invokable,
  store: references.Store, methods: List[str]):
  """Internal implementation of wrapping an Invokable via FastAPI"""
  def journaled_call() -> resources.Resource:
    with store:
      return invokable.invoke().get_output()

  def journaled_call_input(data: resources.Resource) -> resources.Resource:
    with store:
      return invokable.invoke(enact.commit(data)).get_output()

  wrapper_fn = (journaled_call if
                invokable.get_input_type() == enact.NoneResource
                else journaled_call_input)

  app.router.add_api_route(
    route, wrapper_fn, name=type(invokable).__name__, # type: ignore[arg-type]
    description=invokable.__doc__, response_model=invokable.get_output_type(),
    methods=methods)

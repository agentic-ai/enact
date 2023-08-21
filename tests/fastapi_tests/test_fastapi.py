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

"""Tests for the fastapi module."""

import dataclasses
from typing import Iterable
import unittest

import enact
import enact.fastapi
import fastapi
import fastapi.testclient



@enact.register
@dataclasses.dataclass
class MyRequest(enact.Resource):
  x: int
  y: float

@enact.register
@dataclasses.dataclass
class MyResponse(enact.Resource):
  x: int
  y: float

@enact.typed_invokable(input_type=MyRequest,
                       output_type=MyResponse)
class MyInputInvokable(enact.Invokable):
  def call(self, value: MyRequest):
    return MyResponse(value.x + 1, value.y + 1.0)


@enact.typed_invokable(input_type=enact.NoneResource,
                       output_type=MyResponse)
class MyInvokable(enact.Invokable):
  def call(self):
    return MyResponse(1, 2.0)


@enact.register
class NonDataclassResource(enact.ResourceBase):
  pass


@enact.typed_invokable(input_type=NonDataclassResource,
                       output_type=MyResponse)
class BadInputTypeInvokable(enact.Invokable):
  def call(self, unused_val):
    return MyResponse(1, 2.0)


@enact.register
class NonDataclassFieldsResource(enact.Resource):
  @classmethod
  def field_names(cls) -> Iterable[str]:
    return ['a']


@enact.typed_invokable(input_type=NonDataclassFieldsResource,
                       output_type=MyResponse)
class BadInputTypeInvokableExtraFields(enact.Invokable):
  def call(self, unused_val):
    return MyResponse(1, 2.0)


@enact.register
class UntypedInvokable(enact.Invokable):
  def call(self, unused_val):
    return MyResponse(1, 2.0)


class TestFastAPI(unittest.TestCase):
  """Tests for FastAPI."""

  def test_post_request(self):
    """Tests that a FastAPI server can serve a post request."""
    app = fastapi.FastAPI()
    store = enact.Store()
    my_invokable = MyInputInvokable()
    enact.fastapi.post(app, '/my_invokable/', my_invokable, store)
    result = fastapi.testclient.TestClient(app).post(
      '/my_invokable/', json={'x': 1, 'y': 2.0})
    self.assertEqual(result.json(), {'x': 2, 'y': 3.0})

  def test_get_request(self):
    """Tests that a FastAPI server can serve a get request."""
    app = fastapi.FastAPI()
    store = enact.Store()
    my_invokable = MyInvokable()
    enact.fastapi.get(app, '/my_invokable/', my_invokable, store)
    result = fastapi.testclient.TestClient(app).get('/my_invokable')
    self.assertEqual(result.json(), {'x': 1, 'y': 2.0})

  def test_bad_types(self):
    """Tests that using invalid input/output types raises."""
    app = fastapi.FastAPI()
    store = enact.Store()
    for i, invokable_cls in enumerate([
        BadInputTypeInvokable,
        BadInputTypeInvokableExtraFields,
        UntypedInvokable]):
      with self.subTest(i):
        my_invokable = invokable_cls()
        with self.assertRaises(ValueError):
          enact.fastapi.post(app, '/my_invokable/', my_invokable, store)
        with self.assertRaises(ValueError):
          enact.fastapi.get(app, '/my_invokable/', my_invokable, store)

  def test_get_with_non_none_input(self):
    app = fastapi.FastAPI()
    store = enact.Store()
    with self.assertRaises(ValueError):
      enact.fastapi.get(app, '/my_invokable/', MyInputInvokable(), store)

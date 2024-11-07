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

"""Common functionality for notebooks."""

import dataclasses
import io
import os
from typing import List, Tuple, Type, TypeVar, Union

import enact
import PIL.Image
import numpy as np
import replicate  # type: ignore
import requests  # type: ignore

import api_keys


OPENAI_API_KEY = api_keys.OPENAI_API_KEY

REPLICATE_API_KEY = api_keys.OPENAI_API_KEY

_TIMEOUT = 20

CHAT_GPT_URL = 'https://api.openai.com/v1/chat/completions'
REPLICATE_URL = 'https://api.replicate.com/v1/predictions'

KANDINSKY_REPLICATE_VERSION = (
  'ai-forever/kandinsky-2.2:'
  'ea1addaab376f4dc227f5368bbd8eff901820fd1cc14ed8cad63b29249e9d463')


SDXL_REPLICATE_VERSION = (
  'stability-ai/sdxl:'
  '8beff3369e81422112d93b89ca01426147de542cd4684c244b673b105188fe5f')


def kandinsky(prompt: str, **kwargs) -> PIL.Image.Image:
  """Call the kandinsky text-to-image model using replicate's API."""
  return replicate_text2image(prompt, KANDINSKY_REPLICATE_VERSION, **kwargs)


def sdxl(
    prompt: str,
    refine: str='expert_ensemble_refiner',
    **kwargs) -> PIL.Image.Image:
  """Call the SDXL text-to-image model using replicate's API."""
  return replicate_text2image(
    prompt, SDXL_REPLICATE_VERSION,
    refine=refine,
    **kwargs)


def replicate_text2image(
    prompt: str,
    model_version: str,
    **kwargs) -> PIL.Image.Image:
  """Call the kandinsky text-to-image model using replicate's API."""
  os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_KEY.get()
  output = replicate.run(
    model_version,
    input={'prompt': prompt, **kwargs})
  return PIL.Image.open(
      io.BytesIO(requests.get(output[0], timeout=_TIMEOUT).content))


def chat_gpt(messages: List[Tuple[str, str]],
             model: str='gpt-3.5-turbo',
             **kwargs) -> str:
  """Calls ChatGPT given a conversation history.
  Args:
    messages: A list of pairs (role, message), where role can be one of
      'system', 'assistant' or 'user', and message is the text of the message.
    model: The name of the model to use.
  Returns:
    The response from ChatGPT.
  """
  json_dict = {
    'model': model,
    'messages': [
      {'role': role, 'content': content}
      for role, content in messages],
    **kwargs}
  r = requests.post(
    url=CHAT_GPT_URL,
    json=json_dict,
    headers={'Authorization': 'Bearer ' + OPENAI_API_KEY.get(),
             'Content-Type': 'application/json'},
    timeout=_TIMEOUT)
  return r.json()[
    'choices'][0]['message']['content']


@enact.register
@dataclasses.dataclass
class NPArrayWrapper(enact.TypeWrapper):
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
NPIntWrapperT = TypeVar('NPIntWrapperT', bound='NPIntWrapper')

NPFloatType = Union[np.float16, np.float32, np.float64]
NPIntType = Union[np.int8, np.int16, np.int32, np.int64]


@dataclasses.dataclass
class NPFloatWrapper(enact.TypeWrapper):
  """Base class for resource wrappers for numpy float scalars."""
  value: float

  @classmethod
  def wrap(cls: Type[NPFloatWrapperT], value: np.float32) -> NPFloatWrapperT:
    """Returns a wrapper for the resource."""
    return cls(value=float(value))

  def unwrap(self) -> np.ndarray:
    """Returns the wrapped resource."""
    return self.wrapped_type()(self.value)


@enact.register
class NPFloat16Wrapper(NPFloatWrapper):
  """Resource wrapper for numpy float16 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.float16]:
    return np.float16


@enact.register
class NPFloat32Wrapper(NPFloatWrapper):
  """Resource wrapper for numpy float32 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.float32]:
    return np.float32


@enact.register
class NPFloat64Wrapper(NPFloatWrapper):
  """Resource wrapper for numpy float64 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.float64]:
    return np.float64


@dataclasses.dataclass
class NPIntWrapper(enact.TypeWrapper):
  """Base class for resource wrappers for numpy int scalars."""
  value: int

  @classmethod
  def wrap(cls: Type[NPIntWrapperT], value: NPIntType) -> NPIntWrapperT:
    """Returns a wrapper for the resource."""
    return cls(value=int(value))

  def unwrap(self) -> np.ndarray:
    """Returns the wrapped resource."""
    return self.wrapped_type()(self.value)

@enact.register
class NPInt8Wrapper(NPIntWrapper):
  """Resource wrapper for numpy int8 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.int8]:
    return np.int8

@enact.register
class NPInt16Wrapper(NPIntWrapper):
  """Resource wrapper for numpy int16 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.int16]:
    return np.int16


@enact.register
class NPInt32Wrapper(NPIntWrapper):
  """Resource wrapper for numpy int32 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.int32]:
    return np.int32


@enact.register
class NPInt64Wrapper(NPIntWrapper):
  """Resource wrapper for numpy int64 scalars."""
  @classmethod
  def wrapped_type(cls) -> Type[np.int64]:
    return np.int64


@enact.register
@dataclasses.dataclass
class PILImageWrapper(enact.TypeWrapper):
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

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

import functools
import io
import os
from typing import List, NamedTuple, Tuple

import PIL.Image
import replicate  # type: ignore
import requests  # type: ignore

class APIKeyResolver(NamedTuple):
  """Looks for an API key."""
  name: str
  env_var: str
  file_path: str

  @functools.lru_cache
  def get(self) -> str:
    if os.environ.get(self.env_var):
      return os.environ[self.env_var]
    try:
      with open(os.path.expanduser(self.file_path)) as f:
        return f.read().strip()
    except FileNotFoundError:
      raise APIKeyNotFound(
          f'Plese provide {self.name} API key in environment variable '
          f'{self.env_var} or in file {self.file_path}.')

OPENAI_API_KEY = APIKeyResolver(
  name='OpenAI',
  env_var='OPENAI_API_KEY',
  file_path='~/openai.key')

REPLICATE_API_KEY = APIKeyResolver(
  name='Replicate',
  env_var='REPLICATE_API_KEY',
  file_path='~/replicate.key')


CHAT_GPT_URL = 'https://api.openai.com/v1/chat/completions'
REPLICATE_URL = 'https://api.replicate.com/v1/predictions'

KANDINSKY_REPLICATE_VERSION = (
  'ai-forever/kandinsky-2.2:'
  'ea1addaab376f4dc227f5368bbd8eff901820fd1cc14ed8cad63b29249e9d463')


SDXL_REPLICATE_VERSION = (
  'stability-ai/sdxl:'
  '8beff3369e81422112d93b89ca01426147de542cd4684c244b673b105188fe5f')


class APIKeyNotFound(Exception):
  """The API key was not found."""


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
      io.BytesIO(requests.get(output[0]).content))




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
             'Content-Type': 'application/json'})
  try:
    return r.json()[
      'choices'][0]['message']['content']
  except IndexError:
    raise ValueError('Unexpected response: %s' % r.json())
  except KeyError:
    raise ValueError('Unexpected response: %s' % r.json())

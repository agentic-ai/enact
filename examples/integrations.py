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

import functools
import os
from typing import List, Tuple
import requests  # type: ignore

CHAT_GPT_URL = 'https://api.openai.com/v1/chat/completions'
OPENAI_KEY_PATH = os.path.expanduser('~/openai.key')
OPENAI_ENV_VAR = 'OPENAI_KEY'


class APIKeyNotFound(Exception):
  """The API key was not found."""



@functools.lru_cache
def openai_api_key() -> str:
  """Returns the OpenAI API key."""
  if OPENAI_ENV_VAR in os.environ:
    return os.environ[OPENAI_ENV_VAR]
  try:
    with open(OPENAI_KEY_PATH) as f:
      return f.read().strip()
  except FileNotFoundError:
    raise APIKeyNotFound(
        f'Plese provide OpenAI API key in environment variable '
        f'{OPENAI_KEY_PATH} or in file {OPENAI_KEY_PATH}.')


def chat_gpt(messages: List[Tuple[str, str]],
             model: str='gpt-3.5-turbo') -> str:
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
      for role, content in messages]}
  r = requests.post(
    url=CHAT_GPT_URL,
    json=json_dict,
    headers={'Authorization': 'Bearer ' + openai_api_key(),
             'Content-Type': 'application/json'})
  try:
    return r.json()[
      'choices'][0]['message']['content']
  except IndexError:
    raise ValueError('Unexpected response: %s' % r.json())
  except KeyError:
    raise ValueError('Unexpected response: %s' % r.json())

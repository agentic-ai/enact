from typing import NamedTuple

import functools
import os


class APIKeyNotFound(Exception):
  """The API key was not found."""


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
      with open(os.path.expanduser(self.file_path), encoding='utf-8') as f:
        return f.read().strip()
    except FileNotFoundError:
      # pylint: disable=raise-missing-from
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

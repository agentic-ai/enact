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
"""Random resource generator for fuzz testing."""

import dataclasses
import random
from typing import Callable, List, Mapping, Sequence

import enact

def rand_str() -> str:
  return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz',
                                k=random.randint(0, 10)))

def rand_resource() -> enact.ResourceBase:
  """Returns a random resource."""
  return R(rand_value(), rand_value())

def rand_sequence() -> Sequence[enact.FieldValue]:
  """Return a random sequence."""
  return [rand_value() for _ in range(random.randint(0, 3))]

def rand_map() -> Mapping[str, enact.FieldValue]:
  """Return a random mapping."""
  return {rand_str(): rand_value() for _ in range(random.randint(0, 3))}

rand_generators: List[Callable[[], enact.FieldValue]] = [
  lambda: None,
  lambda: random.choice([True, False]),
  lambda: random.randint(-1000, 1000),
  lambda: random.uniform(-1000, 1000),
  rand_str,
  lambda: rand_str().encode('utf-8'),
  rand_resource,
  rand_sequence,
  rand_map,
  lambda: R,  # type-valued field.
]

def rand_value() -> enact.FieldValue:
  """Returns a random value for a field."""
  return random.choice(rand_generators)()

@enact.register
@dataclasses.dataclass
class R(enact.Resource):
  a: enact.FieldValue
  b: enact.FieldValue

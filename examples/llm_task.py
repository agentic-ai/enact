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

import dataclasses
from typing import List, Optional

import enact

import chat
import integrations


@enact.register
@dataclasses.dataclass
class TaskExample(enact.Resource):
  input: str
  output: str

@enact.register
@dataclasses.dataclass
class Correction(TaskExample):
  correction: str


@enact.typed_invokable(enact.Str, enact.Str)
class TaskInput(enact.RequestInput):
  def __init__(self, *args, **kwargs):
    super().__init__(
      enact.Str,
      'Please provide a task prompt')


@dataclasses.dataclass
class ProcessedOutput(enact.Resource):
  """A postprocessed output."""
  output: Optional[enact.Str]
  correction: Optional[enact.Str]


@enact.typed_invokable(enact.Str, enact.Str)
@dataclasses.dataclass
class Task(enact.Invokable):
  """A text-based task with a prompt and some input/output examples."""
  task_prompt: Optional[str]
  examples: List[TaskExample] = dataclasses.field(default_factory=list)
  corrections: List[Correction] = dataclasses.field(default_factory=list)
  post_processor: Optional[
    enact.Ref[enact.Invokable[enact.Str, ProcessedOutput]]] = None
  max_retries: int = 3
  chat_agent: chat.ChatAgent = dataclasses.field(
    default_factory=chat.GPTAgent)

  def add_example(self, input: str, output: str):
    """Add an example to the task."""
    self.examples.append(TaskExample(input=input, output=output))

  def add_correction(self, input: str, output: str, correction: str):
    """Add an example to the task."""
    self.corrections.append(Correction(
      input=input, output=output, correction=correction))

  def attempt(self, input: enact.Str) -> enact.Str:
    if not self.task_prompt:
      task_input = TaskInput()
      self.task_prompt = str(task_input(input))
    conv = chat.Conversation()
    prompt: List[str] = []
    conv.add_system(
      f'Your job is to accomplish the following task. Adhere closely '
      f'to the provided task description and examples (if provided).\n'
      f'TASK: {self.task_prompt}')
    if self.examples:
      conv.add_system('The following is a list of examples.')
    for ex in self.examples:
      conv.add_system('INPUT:')
      conv.add_system(ex.input)
      conv.add_system('OUTPUT:')
      conv.add_system(ex.output)
    if self.corrections:
      conv.add_system(
        'The following is a a list of corrections of previous outputs '
        'from this task, please use these to improve your response.')
    for c in self.corrections:
      conv.add_system('INPUT:')
      conv.add_system(c.input)
      conv.add_system('CRITIQUED OUTPUT:')
      conv.add_system(c.output)
      conv.add_system('CRITIQUE/CORRECTION:')
      conv.add_system(c.correction)

    conv.add_system('START TASK:')
    conv.add_system('INPUT:')
    conv.add_system(str(input))
    conv.add_system('OUTPUT:')
    return enact.Str(self.chat_agent(conv).content)

  def call(self, input: enact.Str) -> enact.Str:
    """Call the task."""
    for _ in range(self.max_retries):
      output = self.attempt(input)
      if self.post_processor:
        processed_output = self.post_processor()(output)
        if not processed_output.correction:
          return enact.Str(processed_output.output)
        self.add_correction(input=str(input),
                            output=str(output),
                            correction=processed_output.correction)
      else:
        return output
    raise MaxRetriesExceeded()


@enact.register
class MaxRetriesExceeded(enact.ExceptionResource):
  pass
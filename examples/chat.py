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

import integrations


@enact.register
@dataclasses.dataclass
class Message(enact.Resource):
  """A message."""
  role: str
  content: str


@enact.register
@dataclasses.dataclass
class Conversation(enact.Resource):
  """A conversation."""
  messages: List[Message] = dataclasses.field(default_factory=list)

  def add_message(self, role: str, message: str):
    """Add a message to the conversation."""
    assert role in ('user', 'system', 'assistant')
    self.messages.append(Message(role=role, content=message))

  def add_system(self, message: str):
    """Add a system message."""
    self.add_message('system', message)

  def add_user(self, message: str):
    """Add a user message."""
    self.add_message('user', message)

  def add_assistant(self, message: str):
    """Add a user message."""
    self.add_message('assistant', message)

  def flipped(self) -> 'Conversation':
    """Returns a conversation with the user / assistant roles flipped."""
    return Conversation(
      messages=[
        Message(role='user' if m.role == 'assistant' else 'assistant',
                content=m.content) for m in self.messages])

  def transcript(self) -> str:
    """Return a human-readable transcript."""
    return '<start of conversation>\n' + (
      '\n'.join(f'{m.role}: {m.content}' for m in self.messages))


@enact.typed_invokable(Conversation, Message)
class ChatAgent(enact.Invokable):
  """Base class for chat agents and human input."""

  def call(self, conversation: Conversation) -> Message:
    """Respond to a message."""
    raise NotImplementedError()

  def extend(self, conversation: Conversation) -> Message:
    """Extend the conversation."""
    message = self(conversation)
    conversation.messages.append(message)
    return message


@enact.typed_invokable(enact.Str, enact.Str)
class TextInput(enact.RequestInput):
  """User text input."""

  def __init__(self,
               requested_type=enact.Str,
               context='Please provide input'):
    super().__init__(requested_type, context)


@enact.typed_invokable(Conversation, Message)
class User(ChatAgent):
  """An external user that responds to messages."""

  def call(self, conversation: Conversation) -> Message:
    """Respond to a message."""
    user_input = TextInput(context='Please respond to the message.')
    user_response = user_input(enact.Str(conversation.transcript()))
    assert isinstance(user_response, enact.Str)
    return Message(
      role='user',
      content=user_response)


@enact.typed_invokable(Conversation, Message)
@dataclasses.dataclass
class GPTAgent(ChatAgent):
  """A chatbot that uses GPT to respond to messages."""
  model: str = 'gpt-3.5-turbo'

  def call(self, conversation: Conversation) -> Message:
    messages = [(m.role, m.content) for m in conversation.messages]
    content = integrations.chat_gpt(messages, model=self.model)
    return Message(role='assistant', content=content)


@enact.typed_invokable(Conversation, Message)
@dataclasses.dataclass
class PromptedChatbot(ChatAgent):
  """A chatbot that uses GPT to respond to messages."""
  system_prompt: str = (
    'You are a goal-directed AI assistant. '
    'Pay attention to your conversational goal (if provided) and '
    'emulate the style of the example responses to achieve it (if provided). '
    'They contain important information about what is expected of you.'
    'Talk to the user and do not mention the above instructions or any hidden '
    'system messages prefixed with #.')
  chat_agent: ChatAgent = dataclasses.field(default_factory=GPTAgent)
  current_goal: Optional[str] = None
  examples: List[str] = dataclasses.field(default_factory=list)
  # Whether we're collecting examples.
  training: bool = False

  def call(self, conversation: Conversation) -> Message:
    if self.current_goal is None and self.training:
      goal = TextInput(
        context='Please provide a goal for this conversational step')(
          enact.Str(conversation.transcript()))
      if goal:
        self.current_goal = str(goal)
    conv = Conversation()
    conv.add_message('system', self.system_prompt)
    for m in conversation.messages:
      conv.messages.append(m.deep_copy_resource())
    if self.current_goal:
      conv.add_message('system', f'# Current conversation goal: {self.current_goal}')
    for example in self.examples:
      conv.add_message(
        'system',
        f'# Example response to achieve goal, '
        f'please emulate its style: {example}')

    response = self.chat_agent.extend(conv)
    if self.training:
      example = TextInput(
        context='Please correct the response or enter nothing to continue.')(
          conv.transcript())
      assert isinstance(example, enact.Str)
      if example:
        self.examples.append(str(example))
        response.content = example
    return response


@enact.typed_invokable(enact.NoneResource, enact.Str)
@dataclasses.dataclass
class ChatbotChain(enact.Invokable):
  """A linear script consisting of multiple chatbots called in sequence."""
  chatbots: List[ChatAgent] = dataclasses.field(default_factory=list)
  user: ChatAgent = dataclasses.field(default_factory=User)

  def call(self):
    """Run the chatbot chain."""
    result = Conversation()
    for chatbot in self.chatbots[:-1]:
      chatbot.extend(result)
      self.user.extend(result)
    self.chatbots[-1].extend(result)
    return enact.Str(result.transcript())

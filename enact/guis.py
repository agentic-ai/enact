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

"""Generic UI components."""

import abc
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

import gradio as gr  # type: ignore
from gradio import events as gradio_events
import PIL.Image

from enact import contexts
from enact import interfaces
from enact import invocations
from enact import references
from enact import pretty_print
from enact import resource_types
from enact import serialization


W = TypeVar('W', bound='ResourceWidget')
C = TypeVar('C', bound=gr.components.Component)


class ResourceWidget(abc.ABC):
  """A UI widget that represents a resource."""

  def __init__(self):
    """Initializes the component."""
    self.components: List[gr.components.Component] = []

  @classmethod
  @abc.abstractmethod
  def supports_type(cls, resource_type: Type[interfaces.ResourceBase]) -> bool:
    """Returns whether the widget supports the resource type."""

  @classmethod
  def create(cls: Type[W],
             resource_type: Type[interfaces.ResourceBase],
             **kwargs) -> W:
    """Create a widget for a resource type or raises an error if not supported."""
    if not cls.supports_type(resource_type):
      raise TypeError(f'{cls} does not support {resource_type}')
    return cls(**kwargs)

  def add(self, c: C) -> C:
    """Adds a component to the widget."""
    self.components.append(c)
    return c

  @abc.abstractmethod
  def set(self,
          resource: Optional[interfaces.ResourceBase]=None,
          **kwargs) -> List[Dict[str, Any]]:
    """Set content and properties of UI elements to resource."""

  @abc.abstractmethod
  def get(self, *component_values) -> Optional[interfaces.ResourceBase]:
    """Returns the current contents as a resource."""

  def set_from_event(
      self,
      event: gradio_events.EventListenerMethod,
      fun: Callable[..., Optional[interfaces.ResourceBase]],
      inputs: List[gr.components.Component],
      **kwargs) -> gradio_events.Dependency:
    """Set from an event."""
    def _set(*args):
      resource = fun(*args)
      updates = self.set(resource, **kwargs)
      if len(updates) == 1:
        # Gradio needs unpacking for singleton lists.
        return updates[0]
      else:
        return updates
    return event(
      contexts.with_current_contexts(_set),
      inputs=inputs,
      outputs=self.components)


RW = TypeVar('RW', bound='RefWidget')

class RefWidget(ResourceWidget):
  """A UI widget that represents a reference."""

  def __init__(self, **kwargs):
    super().__init__()
    with gr.Group():
      self.digest_box = self.add(gr.Textbox(show_label=False, **kwargs))
      with gr.Accordion(label='Referenced resource', open=True):
        kwargs['interactive'] = False
        self.ref_details = self.add(
          gr.Markdown(**kwargs, value='```\nNo reference selected\n```'))
    self.change = self.digest_box.change

    def changed_ref_id(ref_id: str) -> Dict:
      """Performs necessary updates when ref id changes."""
      try:
        ref: references.Ref = references.Ref.from_id(ref_id)
      except json.JSONDecodeError:
        return self.ref_details.update(value='Invalid reference id.')
      try:
        resource = ref.get()
      except references.NotFound:
        return self.ref_details.update(value='Reference not found.')
      return self.ref_details.update(
        value=f'```\n{pretty_print.pformat(resource)}\n```')

    self.change(contexts.with_current_contexts(changed_ref_id),
                inputs=[self.digest_box],
                outputs=[self.ref_details])

  @classmethod
  def supports_type(cls, resource_type: Type[interfaces.ResourceBase]) -> bool:
    """Returns whether the widget supports the resource type."""
    return issubclass(resource_type, references.Ref)

  def set(self,
          resource: Optional[interfaces.ResourceBase]=None,
          **kwargs) -> List[Dict[str, Any]]:
    """Set content and properties of UI elements."""
    if not resource:
      return [
        self.digest_box.update(value='', **kwargs),
        self.ref_details.update(**kwargs)]
    assert isinstance(resource, references.Ref)
    return [
      self.digest_box.update(value=resource.id, **kwargs),
      self.ref_details.update(**kwargs)]

  def get(self, *component_values) -> Optional[interfaces.ResourceBase]:
    """Returns the current contents as a resource."""
    ref_box_value = component_values[0]
    if not ref_box_value:
      return None
    return references.Ref.from_id(ref_box_value)


class ImageWidget(ResourceWidget):
  """Supports image-type resources."""

  def __init__(self, **kwargs):
    super().__init__()
    self._image = self.add(gr.Image(**kwargs))

  @classmethod
  def supports_type(cls, resource_type: Type[interfaces.ResourceBase]) -> bool:
    """Supports all resources."""
    return issubclass(resource_type, resource_types.Image)

  def set(self,
          resource: Optional[interfaces.ResourceBase]=None,
          **kwargs) -> List[Dict[str, Any]]:
    """Set content and properties of UI elements to resource."""
    assert isinstance(resource, resource_types.Image)
    return [self._image.update(value=resource.value, **kwargs)]

  def get(self, *component_values) -> Optional[interfaces.ResourceBase]:
    """Returns the current contents as a resource."""
    if component_values[0] is None:
      return None
    return resource_types.Image(
      PIL.Image.fromarray(component_values[0]))


class JsonWidget(ResourceWidget):
  """Default widget to read / display a resource using JSON field input."""

  def __init__(self, resource_type: Type[interfaces.ResourceBase], **kwargs):
    """Initializes the component."""
    super().__init__()
    self._type = resource_type
    self._serializer = serialization.JsonSerializer()
    self._boxes: Dict[str, gr.Textbox] = {}
    with gr.Group():
      for field_name in self._type.field_names():
        self._boxes[field_name] = self.add(gr.Textbox(
          label=field_name, **kwargs))

  @classmethod
  def supports_type(cls, resource_type: Type[interfaces.ResourceBase]) -> bool:
    """Supports all resources."""
    return True

  @classmethod
  def create(
      cls,
      resource_type: Type[interfaces.ResourceBase],
      **kwargs) -> 'JsonWidget':
    """Creates a new instance of the widget."""
    return cls(resource_type, **kwargs)

  def set(self,
          resource: Optional[interfaces.ResourceBase]=None,
          **kwargs) -> List[Dict[str, Any]]:
    """Set content and properties of UI elements."""
    updates: List[Dict[str, Any]] = []
    if resource:
      resource_dict = resource.to_resource_dict()
    else:
      resource_dict = None

    for field_name in self._type.field_names():
      update_dict: Dict[str, Any] = {}
      if resource_dict:
        update_dict['value'] = json.dumps(
          self._serializer.to_json(resource_dict[field_name]),
          indent=2)
      update_dict.update(kwargs)
      updates.append(self._boxes[field_name].update(**update_dict))
    return updates

  def get(self, *component_values) -> Optional[interfaces.ResourceBase]:
    """Returns the current contents as a resource."""
    resource_dict = interfaces.ResourceDict(self._type)
    for field_name, value in zip(self._type.field_names(), component_values):
      json_val = json.loads(value)
      resource_dict[field_name] = self._serializer.from_json(json_val)
    return self._type.from_resource_dict(resource_dict)


class GUI:
  """A gradio GUI component for an invokable."""

  def __init__(self, invokable: references.Ref[invocations.InvokableBase]):
    """Create a new UI for the component."""
    self._invokable = invokable

    input_type = self._invokable.get().get_input_type()
    if not input_type:
      raise ValueError('Input type must be specified.')
    self._input_type: Type[interfaces.ResourceBase] = input_type

    output_type = self._invokable.get().get_output_type()
    if not output_type:
      raise ValueError('Output type must be specified.')
    self._output_type: Type[interfaces.ResourceBase] = output_type

    self._blocks: Optional[gr.Blocks] = None
    self._input_widget: Optional[ResourceWidget] = None
    self._output_widget: Optional[ResourceWidget] = None
    self._invocation_widget: Optional[RefWidget] = None

    self._widget_types: List[Type[ResourceWidget]] = []
    self.register(JsonWidget)
    self.register(RefWidget)
    self.register(ImageWidget)

  @property
  def input_widget(self) -> ResourceWidget:
    assert self._input_widget, 'Blocks not generated yet.'
    return self._input_widget

  @property
  def output_widget(self) -> ResourceWidget:
    assert self._output_widget, 'Blocks not generated yet.'
    return self._output_widget

  @property
  def invocation_widget(self) -> RefWidget:
    assert self._invocation_widget, 'Blocks not generated yet.'
    return self._invocation_widget

  def register(self, resource_widget: Type[ResourceWidget]):
    self._widget_types.append(resource_widget)

  @property
  def blocks(self) -> gr.Blocks:
    if not self._blocks:
      self._blocks = self._create_blocks()
    return self._blocks

  def _create_widget_by_resource_type(
      self, resource_type: Type[interfaces.ResourceBase], **kwargs) -> (
        ResourceWidget):
    """Create a widget for a resource type."""
    for widget_type in self._widget_types[::-1]:
      if widget_type.supports_type(resource_type):
        return widget_type.create(resource_type, **kwargs)
    raise TypeError(f'No widget for type {resource_type} found.')

  def _get_input_widget(self, resource_type: Type[interfaces.ResourceBase]):
    """Returns the widget for the input resource type."""
    return self._create_widget_by_resource_type(resource_type)

  def _get_output_widget(self, resource_type: Type[interfaces.ResourceBase]):
    """Returns the widget for the input resource type."""
    return self._create_widget_by_resource_type(resource_type, interactive=False)

  def _invoke(self, *args) -> Optional[references.Ref[invocations.Invocation]]:
    """Invoke the object."""
    input_resource = self.input_widget.get(*args[:len(self.input_widget.components)])
    if not input_resource:
      return None
    last_invocation = self.invocation_widget.get(*args[len(
      self.input_widget.components):])
    invokable = self._invokable
    if last_invocation:
      assert isinstance(last_invocation, references.Ref)
      invokable = last_invocation.get().response.get().invokable
    invocation = invokable.get().invoke(references.commit(input_resource))
    return references.commit(invocation)

  def _title(self, ref: references.Ref) -> str:
    return f'### *{type(ref.get()).__name__}* `{ref.digest[:6]}`'

  def _create_blocks(self) -> gr.Blocks:
    """Return the gradio Blocks object representing the UI."""
    with gr.Blocks() as blocks:
      with gr.Group():
        title = gr.Markdown(value=self._title(self._invokable))
        with gr.Accordion(label='Resource details', open=False) as details:
          invokable_details = gr.Markdown(
            value=f'```{pretty_print.pformat(self._invokable.get())}```')

      with gr.Group():
        gr.Markdown(value='Input:')
        self._input_widget = self._create_widget_by_resource_type(
          self._input_type)

      # Row is a workaround https://github.com/gradio-app/gradio/issues/4505.
      output_group = gr.Row(visible=False)
      with output_group, gr.Group():
        gr.Markdown(value='Output:')
        self._output_widget = self._create_widget_by_resource_type(
          self._output_type, interactive=False)

      # Row is a workaround for https://github.com/gradio-app/gradio/issues/4505.
      exception_group = gr.Row(visible=False)
      with exception_group, gr.Group():
        gr.Markdown(value='Exception:')
        exception_area = gr.TextArea(value='', interactive=False, show_label=False)

      with gr.Group():
        with gr.Accordion(label='Invocation details', open=False):
          self._invocation_widget = RefWidget()

      run_button = gr.Button('Run')
      def invocation(*args):
        """Grab the current invocation."""
        ref = self.invocation_widget.get(*args)
        if not ref:
          return None
        assert isinstance(ref, references.Ref)
        invocation = ref.get()
        assert isinstance(invocation, invocations.Invocation)
        return invocation

      def invocation_input(*args):
        """Return the input of the current invocation."""
        return invocation(*args).request.get().input.get()

      def invocation_output(*args):
        """Return the output of the current invocation."""
        output = invocation(*args).response.get().output
        if not output:
          return None
        return output.get()

      def handle_invocation_exception(*args):
        """Enable / disable exception and output field."""
        exception = invocation(*args).response.get().raised
        if not exception:
          return (exception_group.update(visible=False),
                  exception_area.update(value=''),
                  output_group.update(visible=True))
        else:
          return (exception_group.update(visible=True),
                  exception_area.update(value=str(exception.get())),
                  output_group.update(visible=False))

      def update_title(*args):
        response = invocation(*args).response
        assert response
        return (
          title.update(value=self._title(response.get().invokable)),
          invokable_details.update(
            value=f'```{pretty_print.pformat(response.get().invokable)}```'))

      # Run the invokable on click.
      self.invocation_widget.set_from_event(
        run_button.click,
        contexts.with_current_contexts(self._invoke),
        inputs=self._input_widget.components + self._invocation_widget.components)

      # Set the input if the invocation changes.
      self._input_widget.set_from_event(
        self.invocation_widget.change,
        contexts.with_current_contexts(invocation_input),
        inputs=self.invocation_widget.components)

      # Set the output if the invocation changes.
      self._output_widget.set_from_event(
        self.invocation_widget.change,
        contexts.with_current_contexts(invocation_output),
        inputs=self.invocation_widget.components)

      # Handle exception.
      self._invocation_widget.change(
        contexts.with_current_contexts(handle_invocation_exception),
        inputs=self._invocation_widget.components,
        outputs=[exception_group, exception_area, output_group])

      # Update the invokable title if the the invokable was updated.
      self._invocation_widget.change(
        contexts.with_current_contexts(update_title),
        inputs=self._invocation_widget.components,
        outputs=[title, invokable_details])

      return blocks

  def launch(self, *args, **kwargs) -> Tuple[Any, str, str]:
      """Launch the gradio UI, passing arguments to Blocks.launch."""
      return self.blocks.launch(*args, **kwargs)

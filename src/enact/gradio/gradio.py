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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, cast

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

  def consume(self, component_values: List) -> Optional[interfaces.ResourceBase]:
    """Returns the contents as a resource and removes args from input list."""
    return self.get(*self.consume_args(component_values))

  def consume_args(self, component_values: List) -> List:
    """Return and consume the component arts."""
    component_args = component_values[:len(self.components)]
    del component_values[:len(self.components)]
    return component_args

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
        resource = ref()
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
    if resource:
      return [self._image.update(value=resource.value, **kwargs)]
    else:
      return [self._image.update(value=None, **kwargs)]

  def get(self, *component_values) -> Optional[interfaces.ResourceBase]:
    """Returns the current contents as a resource."""
    if component_values[0] is None:
      return None
    return resource_types.Image(
      PIL.Image.fromarray(component_values[0]))


class StrWidget(ResourceWidget):
  """Supports string-type resources."""

  def __init__(self, **kwargs):
    super().__init__()
    self._box = self.add(gr.Textbox(**kwargs, show_label=False))

  @classmethod
  def supports_type(cls, resource_type: Type[interfaces.ResourceBase]) -> bool:
    """Supports all resources."""
    return issubclass(resource_type, resource_types.Str)

  def set(self,
          resource: Optional[interfaces.ResourceBase]=None,
          **kwargs) -> List[Dict[str, Any]]:
    """Set content and properties of UI elements to resource."""
    if resource is None:
      return [self._box.update(value='', **kwargs)]
    assert isinstance(resource, resource_types.Str)
    return [self._box.update(value=str(resource), **kwargs)]

  def get(self, *component_values) -> Optional[interfaces.ResourceBase]:
    """Returns the current contents as a resource."""
    if component_values[0] is None:
      return None
    return resource_types.Str(component_values[0])


class JsonFieldWidget(ResourceWidget):
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
      **kwargs) -> 'JsonFieldWidget':
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
      else:
        update_dict['value'] = ''
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

  def __init__(
      self,
      invokable: references.Ref[invocations.InvokableBase],
      input_required_inputs: Optional[List[Type[interfaces.ResourceBase]]]=None,
      input_required_outputs: Optional[List[Type[interfaces.ResourceBase]]]=None):
    """Create a new UI for the component.

    Args:
      invokable: The invokable resource to create a UI for.
      input_required_inputs: Input types of invokables that may raise an
        input required exception.
      input_rquired_outputs: Output types of invokables that may raise an
        input required exception.
    """
    self._invokable = invokable
    self._input_required_inputs = input_required_inputs or [
      resource_types.Str
    ]
    self._input_required_outputs = input_required_outputs or [
      resource_types.Str
    ]

    input_type = self._invokable().get_input_type()
    if not input_type:
      raise ValueError('Input type must be specified.')
    self._input_type: Type[interfaces.ResourceBase] = input_type

    output_type = self._invokable().get_output_type()
    if not output_type:
      raise ValueError('Output type must be specified.')
    self._output_type: Type[interfaces.ResourceBase] = output_type

    self._blocks: Optional[gr.Blocks] = None
    self._input_widget: Optional[ResourceWidget] = None
    self._output_widget: Optional[ResourceWidget] = None
    self._invocation_widget: Optional[RefWidget] = None
    self._input_required_input_widgets: Dict[
      Type[interfaces.ResourceBase], ResourceWidget] = {}
    self._input_required_output_widgets: Dict[
      Type[interfaces.ResourceBase], ResourceWidget] = {}

    self._widget_types: List[Type[ResourceWidget]] = []
    self.register(JsonFieldWidget)
    self.register(RefWidget)
    self.register(StrWidget)
    self.register(ImageWidget)

  @property
  def input_widget(self) -> ResourceWidget:
    """Input widget of GUI."""
    assert self._input_widget, 'Blocks not generated yet.'
    return self._input_widget

  @property
  def output_widget(self) -> ResourceWidget:
    """Output widget of GUI."""
    assert self._output_widget, 'Blocks not generated yet.'
    return self._output_widget

  @property
  def invocation_widget(self) -> RefWidget:
    """Invocation widget of GUI."""
    assert self._invocation_widget, 'Blocks not generated yet.'
    return self._invocation_widget

  def register(self, resource_widget: Type[ResourceWidget]):
    """Register a resource widget."""
    self._widget_types.append(resource_widget)

  @property
  def blocks(self) -> gr.Blocks:
    """Blocks object representing the UI."""
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
      invokable = last_invocation().response.get().invokable
    invocation = invokable().invoke(references.commit(input_resource))
    return references.commit(invocation)

  def _continue(self, *args) -> Optional[references.Ref[invocations.Invocation]]:
    """Invoke the object."""
    component_values = list(args)
    invocation_ref = self.invocation_widget.consume(component_values)
    assert isinstance(invocation_ref, references.Ref)
    invocation = invocation_ref()
    assert isinstance(invocation, invocations.Invocation)
    raised = invocation.get_raised()
    assert isinstance(raised, invocations.InputRequest)
    requested_type = raised.requested_type

    user_input: Optional[interfaces.ResourceBase]
    for handled_type, widget in self._input_required_input_widgets.items():
      if handled_type == requested_type:
        user_input = widget.consume(component_values)
      else:
        widget.consume_args(component_values)

    if user_input is None:
      return None
    continued = raised.continue_invocation(invocation, user_input)
    return references.commit(continued)

  def _title(self, ref: references.Ref) -> str:
    return f'### *{type(ref()).__name__}* `{ref.digest[:6]}`'

  def _create_blocks(self) -> gr.Blocks:
    """Return the gradio Blocks object representing the UI."""
    with gr.Blocks() as blocks:
      with gr.Group():
        title = gr.Markdown(value=self._title(self._invokable))
        with gr.Accordion(label='Resource details', open=False):
          invokable_details = gr.Markdown(
            value=f'```{pretty_print.pformat(self._invokable())}```')

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
        input_required_header_md = gr.Markdown(visible=False)
        for resource_type in self._input_required_outputs:
          # Create widgets to display info to the user when querying for input.
          self._input_required_output_widgets[resource_type] = (
            self._create_widget_by_resource_type(
              resource_type, interactive=False, visible=False))
        input_required_input_md = gr.Markdown(visible=False)
        for resource_type in self._input_required_inputs:
          # Create widgets to sample input from the user for InputRequired exceptions.
          self._input_required_input_widgets[resource_type] = (
            self._create_widget_by_resource_type(
              resource_type, interactive=True, visible=False))
        continue_button = gr.Button(value='Continue', visible=False)

      with gr.Group():
        with gr.Accordion(label='Invocation details', open=False):
          self._invocation_widget = RefWidget()

      run_button = gr.Button('Run')
      def get_invocation(*args):
        """Grab the current invocation."""
        ref = self.invocation_widget.get(*args)
        if not ref:
          return None
        assert isinstance(ref, references.Ref)
        invocation = ref()
        assert isinstance(invocation, invocations.Invocation)
        return invocation

      def invocation_input(*args):
        """Return the input of the current invocation."""
        return get_invocation(*args).request().input()

      def invocation_output(*args):
        """Return the output of the current invocation."""
        output = get_invocation(*args).response().output
        if not output:
          return None
        return output()

      def flatten(l):
        """Flatten a list of lists."""
        return [item for sublist in l for item in sublist]

      def component_update(widget: ResourceWidget, **kwargs):
        """Return a list of updates for the widget components."""
        return [c.update(**kwargs) for c in widget.components]

      def update_input_required_widgets(
          updates_list: List[Dict],
          input_widgets: bool,
          input_type: Optional[Type[interfaces.ResourceBase]]=None,
          input_resource: Optional[interfaces.ResourceBase]=None,
          **kwargs) -> bool:
        """Updates widgets that handle InputRequired exceptions."""
        found = False
        if input_widgets:
          input_required_widgets = self._input_required_input_widgets
        else:
          input_required_widgets = self._input_required_output_widgets

        for resource_type, widget in input_required_widgets.items():
          if input_type and input_type == resource_type:
            found = True
            kwargs['visible'] = True
            updates_list += widget.set(input_resource, **kwargs)
          else:
            kwargs['visible'] = False
            updates_list += component_update(widget, **kwargs)
        return found

      def handle_invocation_exception(*args):
        """Enable / disable exception and output field."""
        invocation = get_invocation(*args)
        exception_ref = invocation.response().raised
        if not exception_ref:
          updates = [
            exception_group.update(visible=False),
            exception_area.update(value=''),
            output_group.update(visible=True),
            input_required_header_md.update(visible=False),
            input_required_input_md.update(visible=False),
            continue_button.update(visible=False)]
          update_input_required_widgets(
            updates, input_widgets=False, visible=False)
          update_input_required_widgets(
            updates, input_widgets=True, visible=False)
          return updates

        exception = exception_ref()
        if not isinstance(exception, invocations.InputRequest):
          updates = [
            exception_group.update(visible=True),
            exception_area.update(value=str(exception)),
            output_group.update(visible=False),
            input_required_header_md.update(visible=False),
            input_required_input_md.update(visible=False),
            continue_button.update(visible=False)]
          update_input_required_widgets(
            updates, input_widgets=False, visible=False)
          update_input_required_widgets(
            updates, input_widgets=True, visible=False)
          return updates

        # Try and display the exception causing input to the user.
        output_to_user = cast(interfaces.ResourceBase, exception.input())
        requested_input_from_user_type = exception.requested_type
        updates = []
        found_output_to_user_widget = update_input_required_widgets(
          updates, input_widgets=False,
          input_resource=output_to_user,
          input_type=type(output_to_user))
        found_input_from_user_widget = update_input_required_widgets(
          updates, input_widgets=True,
          input_type=requested_input_from_user_type)

        # If we found a way to display the output, use it, otherwise just dump
        # the exception as text.
        md_value = ''
        if not found_input_from_user_widget:
          md_value = (
            f'User input requested, but no widget found for '
            f'{exception.requested_type}')
        if exception.context:
          md_value += f'\n\n*Context:*\n\n{pretty_print.pformat(exception.context)}'
        if not md_value:
          md_value = 'Provide input below.'

        updates_prefix = [
          exception_group.update(visible=not found_output_to_user_widget),
          exception_area.update(value='' if found_output_to_user_widget else str(exception)),
          output_group.update(visible=False),
          input_required_header_md.update(
            value='### User input interrupt',
            visible=True),
          input_required_input_md.update(value=md_value, visible=True),
          continue_button.update(visible=found_input_from_user_widget)]
        return updates_prefix + updates

      def update_title(*args):
        response = get_invocation(*args).response
        assert response
        return (
          title.update(value=self._title(response().invokable)),
          invokable_details.update(
            value=f'```{pretty_print.pformat(response().invokable)}```'))

      # Run the invokable on clicking Run.
      self.invocation_widget.set_from_event(
        run_button.click,
        contexts.with_current_contexts(self._invoke),
        inputs=self._input_widget.components + self._invocation_widget.components)

      # Continue the invokation on click Continue.
      self.invocation_widget.set_from_event(
        continue_button.click,
        contexts.with_current_contexts(self._continue),
        inputs=self._invocation_widget.components + flatten(
          [w.components for w in self._input_required_input_widgets.values()]
        ))

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
        outputs=[
          exception_group, exception_area, output_group,
          input_required_header_md, input_required_input_md,
          continue_button] +
          flatten([w.components
                  for w in self._input_required_output_widgets.values()]) +
          flatten([w.components
                  for w in self._input_required_input_widgets.values()]))

      # Update the invokable title if the the invokable was updated.
      self._invocation_widget.change(
        contexts.with_current_contexts(update_title),
        inputs=self._invocation_widget.components,
        outputs=[title, invokable_details])

      return blocks

  def launch(self, *args, use_queue: bool=True, **kwargs) -> (
      Tuple[Any, str, Optional[str]]):
    """Launch the gradio UI, passing arguments to Blocks.launch.

    Args:
      use_queue: Specify whether queue should be used. The queue is required
        for executions exceeding 60 seconds.

    Returns:
      Tuple containing the FastAPI app object running the demo, local URL, and
      optional public URL if called with share=True.
    """
    if use_queue:
      return self.blocks.queue().launch(*args, **kwargs)
    return self.blocks.launch(*args, **kwargs)

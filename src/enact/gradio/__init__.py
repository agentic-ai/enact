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

"""Top-level gradio package definitions."""

try:
  import gradio  # type: ignore
  import PIL.Image  # type: ignore
except ModuleNotFoundError as e:
  raise ModuleNotFoundError(
      'Please install gradio and Pillow to enable enact.gradio') from e

from enact.gradio.gradio import *

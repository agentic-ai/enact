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

"""Version information for Enact.

This file should not be moved out of the root directory.
"""

from importlib import metadata
import logging
import os


# The name of the pip distribution.
DIST_NAME = 'enact'
# The root directory containing all python files associated with this
# distribution.
DIST_PYTHON_DIR = os.path.abspath(os.path.dirname(__file__))


try:
  __version__ = metadata.version(DIST_NAME)
except metadata.PackageNotFoundError:
  __version__ = '0.0.0'
  logging.warning(
    'Could not find installed version for %s, using %s',
    DIST_NAME, __version__)

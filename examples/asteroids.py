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
"""A python-only implementation of an asteroids style control task."""

from typing import Callable, Optional, Tuple
import numpy as np
from matplotlib import pyplot as plt  # type: ignore


def normalize_angle(theta: np.ndarray) -> np.ndarray:
  return (theta + np.pi) % (2 * np.pi) - np.pi


class Action:
  """Represents the actions available to an agent:

  * Torque (i.e., steering force)
  * Thrust (i.e., forward acceleration)
  """
  TORQUE_INDEX = 0
  THRUST_INDEX = 1
  # Total dimensionality of the action space array.
  ARRAY_SIZE = 2

  def __init__(self, array=None, batch_shape=()):
    """Create a new action, either from an array or zero-initialized."""
    self.array = array if array is not None else np.zeros(batch_shape + (2,))

  @property
  def torque(self) -> np.ndarray:
    """The torque (i.e., steering force) of the action."""
    return self.array[..., Action.TORQUE_INDEX]

  @torque.setter
  def torque(self, value: np.ndarray):
    """Sets the torque (i.e., steering force) of the action."""
    self.array[..., Action.TORQUE_INDEX] = value

  @property
  def thrust(self) -> np.ndarray:
    """The thrust (i.e., forward acceleration) of the action."""
    return self.array[..., Action.THRUST_INDEX]

  @thrust.setter
  def thrust(self, value: np.ndarray):
    """Sets the thrust (i.e., forward acceleration) of the action."""
    self.array[..., Action.THRUST_INDEX] = value

  def __str__(self) -> str:
    """A string representation of the action."""
    return str(self.array)

  def __repr__(self) -> str:
    """A string representation of the action."""
    return f'<Action: {str(self)}>'

class State:
  """Represents a game state.

  The state is represented internally as a numpy array of length 10.
  State components can be accessed with setters and getters:
  * Agent position
  * Agent orientation [-pi, pi]
  * Agent velocity
  * Agent angular speed
  * Goal position
  * Has been at goal

  A state object may represent an arbitrary batch of states.
  """
  # Indices and slices into the underlying arrays.
  POSITION_SLICE = slice(0, 2)
  ROTATION_INDEX = 2
  VELOCITY_SLICE = slice(3, 5)
  ANGULAR_VELOCITY_INDEX = 5
  GOAL_POSITION_SLICE = slice(6, 8)
  HAS_BEEN_AT_GOAL_INDEX = 8
  # Total dimensionality of the state space array.
  ARRAY_SIZE = 9

  # Game configuration.
  BOARD_SIZE = 25
  REACHED_EPSILON = 1
  TORQUE_FORCE = 0.5
  THRUST_FORCE = 0.5
  MAX_SPEED = 2
  FRICTION_COEFFICIENT = 0.99

  def __init__(
      self,
      array: Optional[np.ndarray]=None,
      batch_shape: Tuple[int, ...]=()):
    """Wraps an existing array in a state or initializes"""
    if array is None:
      self.array = np.zeros(batch_shape + (State.ARRAY_SIZE,),
                            dtype=np.float32)
      # Initialize randomly
      self.initialize_randomly()
    else:
      self.array = array

  def initialize_randomly(self):
    """Initializes the state randomly."""
    batch_shape = self.array.shape[:-1]
    self.position = np.random.uniform(0, State.BOARD_SIZE, batch_shape + (2,))
    self.rotation = np.random.uniform(-np.pi, np.pi, batch_shape)
    self.goal_position = np.random.uniform(
      0, State.BOARD_SIZE, batch_shape + (2,))
    self.success = np.zeros_like(self.rotation)

  @property
  def position(self) -> np.ndarray:
    """Return the position of the agent."""
    return self.array[..., State.POSITION_SLICE]

  @position.setter
  def position(self, value: np.ndarray):
    """Set the position of the agent."""
    self.array[..., State.POSITION_SLICE] = value

  @property
  def rotation(self) -> np.ndarray:
    """Return the rotation of the agent. Positive is counter-clockwise."""
    return self.array[..., State.ROTATION_INDEX]

  @rotation.setter
  def rotation(self, value: np.ndarray):
    """Set the rotation of the agent (normalized to [-pi, pi])."""
    self.array[..., State.ROTATION_INDEX] = normalize_angle(value)

  @property
  def velocity(self) -> np.ndarray:
    """Two dimensional velocity vector of the agent."""
    return self.array[..., State.VELOCITY_SLICE]

  @velocity.setter
  def velocity(self, value: np.ndarray):
    """Set the velocity."""
    self.array[..., State.VELOCITY_SLICE] = value

  @property
  def angular_velocity(self) -> np.ndarray:
    """The angular velocity of the agent."""
    return self.array[..., State.ANGULAR_VELOCITY_INDEX]

  @angular_velocity.setter
  def angular_velocity(self, value: np.ndarray):
    """Set the angular velocity of the agent."""
    self.array[..., State.ANGULAR_VELOCITY_INDEX] = value

  @property
  def goal_position(self) -> np.ndarray:
    """The position of the goal."""
    return self.array[..., State.GOAL_POSITION_SLICE]

  @goal_position.setter
  def goal_position(self, value: np.ndarray):
    """Set the position of the goal."""
    self.array[..., State.GOAL_POSITION_SLICE] = value

  @property
  def has_been_at_goal(self) -> np.ndarray:
    """Whether the agent has been at the goal."""
    return self.array[..., State.HAS_BEEN_AT_GOAL_INDEX]

  @has_been_at_goal.setter
  def has_been_at_goal(self, value: np.ndarray):
    """Set whether the agent has been at the goal."""
    self.array[..., State.HAS_BEEN_AT_GOAL_INDEX] = value

  def forward(self, offset: float=0) -> np.ndarray:
    """The unit vector pointing forward relative to the agent."""
    return np.stack([np.cos(self.rotation + offset),
                     np.sin(self.rotation + offset)], axis=-1)

  def right(self, offset: float=0) -> np.ndarray:
    """The unit vector pointing right relative to the agent."""
    return self.forward(offset - np.pi / 2)

  def at_goal(self) -> np.ndarray:
    """Whether the agent is currently at the goal."""
    return np.linalg.norm(
      self.position - self.goal_position, axis=-1) < State.REACHED_EPSILON

  def __str__(self):
    """Returns a string representation of the state."""
    return str(self.array)

  def __repr__(self):
    """Returns a string representation of the state."""
    return f'<State: {str(self)}>'

  def dynamics(self, a: Action, dt: float=1e-1) -> 'State':
    """Computes the new state resulting from applying an action."""
    next_state = State(np.copy(self.array))
    # Update velocity by applying acceleration forces.
    thrust = np.clip(a.thrust, 0, 1)
    torque = np.clip(a.torque, -1, 1)
    next_state.velocity += (
        dt * State.THRUST_FORCE *
        self.forward() * np.expand_dims(thrust, axis=-1))
    next_state.angular_velocity += dt * State.TORQUE_FORCE * torque

    # Clamp max speed and apply friction.
    speed = np.clip(
        np.expand_dims(np.linalg.norm(next_state.velocity, axis=-1), -1),
        1e-10, np.infty)

    at_maxed_out_speed = (next_state.velocity / speed) * State.MAX_SPEED
    next_state.velocity = np.where(
        speed > State.MAX_SPEED, at_maxed_out_speed, next_state.velocity)

    next_state.velocity *= State.FRICTION_COEFFICIENT
    next_state.angular_velocity *= State.FRICTION_COEFFICIENT

    # Update position by applying velocity.
    next_state.position += dt * next_state.velocity
    next_state.rotation += dt * next_state.angular_velocity

    next_state.has_been_at_goal += next_state.at_goal()
    next_state.has_been_at_goal = np.clip(next_state.has_been_at_goal, 0, 1)

    return next_state


def create_trajectory(
    policy: Callable[[State], Action],
    batch_shape: Optional[Tuple[int, ...]]=None,
    steps: int=200,
    initial_state: Optional[State]=None,
    dynamics: Callable[[State, Action, float],
                       State]=lambda s, a, dt: s.dynamics(a, dt),
    dt: float=1e-1) -> Tuple[State, Action]:
  """Creates a batch of randomly initialized trajectories.

  Args:
    policy: A callable that transforms states into actions.
    batch_shape: The shape of the batch dimensions.
    steps: The number of steps in the trajectory (time dimension).
    initial_state: If not None, the State from which the trajectory
      starts.
    dynamics: An optional dynamics function for computing the next state.
    dt: The timestep of the simulation.

  Returns:
    A pair of state and action arrays of shape (batch_shape, steps, array_dim).
  """
  assert batch_shape is not None or initial_state is not None
  if initial_state is None and batch_shape is not None:
    s = State(batch_shape=batch_shape)
  elif initial_state is not None:
    s = initial_state
  states = []
  actions = []
  for _ in range(steps):
    a = policy(s)
    states.append(s.array)
    actions.append(a.array)
    s = dynamics(s, a, dt)

  result_states = np.stack(states, axis=-2)
  result_actions = np.stack(actions, axis=-2)

  return State(result_states), Action(result_actions)


def measure_policy(
    policy: Callable[[State], Action], steps: int=200, runs: int=50) -> float:
  """Computes the average success rate of a policy."""
  states, _ = create_trajectory(policy, (runs,), steps)
  last_state = State(states.array[..., -1, :])
  return np.mean(last_state.has_been_at_goal)


def plot_trajectory(
    trajectory: Tuple[State, Action],
    axis: Optional[plt.Axes]=None,
    index: Tuple[int, ...]=(0,),
    marker: Optional[str]=None) -> plt.Axes:
  """Plot one of a batch of trajectories.

  Args:
    trajectory: A pair of batched State and Action objects where dimension
      -2 is the time dimension.
    axis: An optional axis object to plot on.
    index: The index of the trajectory to plot.
    marker: An optional plot marker for locations.

  Returns:
    The axis object used to plot the trajectory.
  """
  if axis is None:
    _, axis = plt.subplots(1, 1)

  states, _ = trajectory
  x_pos = states.array[index][..., 0]
  y_pos = states.array[index][..., 1]
  axis.set_aspect(1)
  axis.plot(x_pos, y_pos, marker=marker)
  axis.set_xlim([-10, State.BOARD_SIZE + 10])
  axis.set_ylim([-10, State.BOARD_SIZE + 10])

  axis.add_artist(plt.Circle(
      states.goal_position[index][0],
      1, color='orange'))
  axis.add_artist(plt.Circle(
    states.position[index][0], 0.5))

  start_pos = states.position[index][0]
  fwd = states.forward()[index][0]
  axis.add_artist(plt.Line2D([start_pos[0], start_pos[0] + 2 * fwd[0]],
                             [start_pos[1], start_pos[1] + 2 * fwd[1]],
                             color='purple'))
  axis.set_aspect('equal')
  return axis

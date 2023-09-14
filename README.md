# enact - A Framework for Generative Software.

Enact is a python framework for building generative software, specifically
software that integrates with machine learning models or APIs that generate
distributions of outputs.

Generative AI is driving changes in the way software is built. The unique
challenges of implementing, maintaining and improving generative systems
indicate a need to rethink the software stack from a first-principles
perspective. See [why-enact](#why-enact) for a more in-depth discussion.

## Key features

Enact was created to help developers build generative software. Our focus is on
offering core framework functionality required for developing, inspecting and
improving software that samples from generative subsystems such as LLMs and
diffusion models. To this end, enact boosts your existing python programs by
giving you the ability to
* recursively track generative executions,
* persist execution data and system parameters in versioned storage,
* explore the space of possible system executions via rewind and replay,
* turn your existing code into asynchronous flows that can halt and
  sample input from users, and
* orchestrate higher-order generative flows in which AI systems write and run
  software.

Enact is complementary to libraries that focus on unified integrations with
generative AI systems. Instead of focusing on particular algorithm schemas for
orchestrating generative flows (e.g., tree of thought), we aim to provide a
framework that simplifies the implementation of such schemas.

## Installation and overview

Enact is available as a [pypi package](https://pypi.org/project/enact/) and can
be installed via:

```bash
pip install enact
```
### Invocations
Enact recursively tracks inputs and outputs of functions registered with the
framework. A record of an execution is called an `Invocation`, and can be
persisted to a store. This is useful for tracking the behavior generative
software.

```python
import enact
import random

@enact.register
def roll_die(sides: int) -> int:
  """Roll a die."""
  return random.randint(1, sides)

@enact.register
def roll_sum(num_rolls: int) -> int:
  """Roll dice."""
  return sum(roll_die(6) for _ in range(num_rolls))

with enact.InMemoryStore() as store:
  invocation = enact.invoke(roll_sum, (2,))
  print(enact.invocation_summary(invocation))
```
_Output:_
```
->roll_sum(2) = 7
  ->roll_die(6) = 2
  ->roll_die(6) = 5
```

### Rewind/Replay
Invocations can be rewound and replayed:

```python
# Rewind the last roll and replay it.
with store:
  first_roll = invocation.rewind(1)    # Rewind by one roll.
  print('Partial invocation: ')
  print(enact.invocation_summary(first_roll))
  reroll_second = first_roll.replay()  # Replay the second die roll.
  print('\nReplayed invocation: ')
  print(enact.invocation_summary(reroll_second))
```
_Output:_
```python
Partial invocation:
->roll_sum(2) incomplete
  ->roll_die(6) = 2

Replayed invocation:
->roll_sum(2) = 8
  ->roll_die(6) = 2
  ->roll_die(6) = 6
```

### Human-in-the-loop

In enact, humans are treated as just another generative distribution. Their
input can be sampled using `enact.request_input`.

```python
@enact.register
def human_rolls_die():
  """Query the user for a die roll."""
  return enact.request_input(requested_type=int, for_value='Please roll a die.')

@enact.register
def roll_dice_user_flow():
  """Request the number of die to roll and optionally sample die rolls."""
  num_rolls = enact.request_input(requested_type=int, for_value='Total number of rolls?')
  return sum(human_rolls_die() for _ in range(num_rolls))

request_responses = {
  'Total number of rolls?': 3,  # Roll 3 dice.
  'Please roll a die.': 6,      # Humans always roll 6.
}

with store:
  invocation_gen = enact.InvocationGenerator.from_callable(
    roll_dice_user_flow)
  # Process all user input requests in order.
  for input_request in invocation_gen:
    invocation_gen.set_input(request_responses[input_request.for_value()])
  print(enact.invocation_summary(invocation_gen.invocation))
```
_Output:_
```
->roll_dice_user_flow() = 18
  ->RequestInput(requested_type=<class 'int'>, context=None)(Total number of rolls?) = 3
  ->human_rolls_die() = 6
    ->RequestInput(requested_type=<class 'int'>, context=None)(Please roll a die.) = 6
  ->human_rolls_die() = 6
    ->RequestInput(requested_type=<class 'int'>, context=None)(Please roll a die.) = 6
  ->human_rolls_die() = 6
    ->RequestInput(requested_type=<class 'int'>, context=None)(Please roll a die.) = 6
```

### Making your types enact compatible

Enact has built-in support for basic python datatypes and a few other common
types common to generative software, such as numpy arrays and pillow images.

If you need support for additional types, you can either define them as
`Resource` dataclasses in the framework or define a wrapper.

#### Defining a resource

```python
import dataclasses

@enact.register
@dataclasses.dataclass
class MyResource(enact.Resource):
  x: int
  y: list = dataclasses.field(default_factory=list)

with store:
  # Commit your resource to the store, obtain a reference.
  ref = enact.commit(MyResource(x=1, y=[2, 3]))
  # Check out your reference.
  print(ref.checkout())  # Equivalent to "print(ref())".
```
_Output:_
```python
MyResource(x=1, y=[2, 3])
```

#### Defining a wrapper
For existing types, it can be more convenient to wrap them rather than to
redefine them as enact `Resource` objects.

This can be accomplished by registering a `ResourceWrapper` subclass.

```python
class Die:
  """Non-enact python type."""
  def __init__(self, sides):
    self.sides = sides

  @enact.register
  def roll(self):
    return random.randint(1, self.sides)

@enact.register
@dataclasses.dataclass
class DieWrapper(enact.ResourceWrapper[Die]):
  """Wrapper for Die."""
  sides: int

  @classmethod
  def wrapped_type(cls):
    return Die

  @classmethod
  def wrap(cls, value: Die):
    """Translate to enact wrapper."""
    return DieWrapper(value.sides)

  def unwrap(self) -> Die:
    """Translate to basic python type."""
    return Die(self.sides)

with store:
  die = Die(sides=6)
  invocation = enact.invoke(die.roll)
  print(enact.invocation_summary(invocation))
```
_Output:_
```
-><__main__.Die object at 0x7f3464398340>.roll() = 4
```


## Documentation

Full documentation is work in progress.  See the
[quickstart](examples/quickstart.ipynb) and [enact
concepts](examples/enact_concepts.ipynb) for more information.
And take a look at other [examples](examples/).

## Why enact - A manifesto

Generative AI models are transforming the software development process.

Traditional software relies primarily on functional buildings blocks in which
inputs and system state directly determine outputs. In contrast, software
that samples one or more generative subsystems associates each input with
a range of possible outputs.

This small change in emphasis - from functions to conditional distributions -
implies a shift across multiple dimensions of the engineering process,
summarized in the table below.

|        |  Generative   |  Traditional |
| :----: | :-----------: | :----------: |
| Building block | Conditional distributions | Deterministic functions |
| Engineering | Improve output distributions | Add features, debug errors |
| Subsystems | Unique | Interchangeable |
| Code sharing | Components | Frameworks |
| Executions | Training data | Logged for metrics/debugging |
| Interactivity | Within system components | At system boundaries |
| Code vs data | Overlapping | Distinct |

### Building generative software means fitting distributions

Generative AI can be used to quickly sketch out impressive end-to-end
experiences. Prototypes of text-based agents, for instance, often involve little
more than calling an API with a well-chosen prompt. Evolving such prototypes
into production-ready systems can be non-trivial though, as they may show
undesirable behaviors a certain percentage of the time, may fail to respond
correctly to unusual user inputs or may simply fail to meet a softly defined
quality target.

Where traditional engineering focuses on implementing features and fixing bugs,
generative software engineering tunes the distribution represented by the system
as a whole towards some implicitly specified target, e.g., a set of example
outputs, corrections of previous outputs or task rewards.

In some cases, tuning generative systems can be achieved directly using machine
learning techniques and frameworks. For instance, systems that consist of a thin
software wrapper around a monolithic machine learning model may directly train
the underlying model to shape system behavior.

In other cases, particularly when running complex generative flows that involve
multiple interacting subcomponents that cannot be easily optimized end-to-end,
gradient-free methods are necessary to improve system performance. These range
from using programmers (either human or synthetic) to elaborate the algorithmic
guard-rails of the generative process, running experiments between API identical
subsystems or finding ideal parameters (e.g., prompt templates) for individual
components.

### Recursive elaboration and end-to-end compression

In generative software, individual components produce distributions that may be
more or less well-suited to the system's overall goal: An LLM that performs well
on tabular data may perform less well when used as a chat agent. One fine-tuned
diffusion model may produce images that are closer to the target style than
another. This is distinct from the the case of traditional software engineering,
where the choice between two correct, API-identical implementations of a module
is of little importance to the behavior of the system as a whole.

In addition to a choice between different machine learning models, there is a
choice between multiple algorithmic elaborations of a given model. The behavior
of GPT, for example, can be improved by providing algorithmic guardrails that
structure and steer the generation process such as chain-of-thought or
tree-of-thought prompting.

The profusion of base models and possible algorithmic elaborations of base
models suggests that gradient-free search methods will take a central place in
generative software development. This could be as simple as running an
experiment in which one component is compared against another or as complex as
searching the space of software, e.g., using LLM-boosted genetic algorithms.

Once a component has been optimized using such methods, the resulting data can
be used in gradient-based optimization to 'compress' the gained knowledge into
end-to-end trained models. This suggests that generative systems will alternate
gradient-based and gradient-free methods, with the former supplying new training
signal for the latter. (This loop is the core insight behind algorithms in the
AlphaZero family, in which data from a search-based elaboration of a model is
used to train the next iteration of the model.)

### Generative software is collaborative

There are many generative AI components that are mutually API-compatible (e.g.,
text-to-image generation, LLMs) but that cannot be directly ranked in terms of
quality since they represent a unique take on the problem domain. The
combination of API compatibility and variation lends itself to a more
evolutionary, collaborative style than traditional software, where shared effort
tends to centralize into fewer, lower-level frameworks.

This new collaborative approach is already evidenced by both widespread sharing
of prompt templates and the existence of public databases of fine-tuned models.

### Generative software reflects on its executions.

Many popular programming languages offer capacity for reflection, wherein the
structure of code is programmatically interpretable by code. Generative software
additionally requires _deep reflection_, that is, the ability for code to
reflect on code execution.

Sampling a generative model is computationally intensive and provides valuable
information that may be used as system feedback or training data. Since
generative software may include complex, recursive algorithms over generative
components, this suggests a need for not simply tracking inputs and outputs
at system boundaries, but at every level of execution.

This is particularly relevant to higher order generative flows, in which the
output of one generative step structures the execution of the next: Such a
system can use call traces as feedback to improve its output, in the same way
that a human may use a debugger to step through an execution to debug an issue.

### Humans are in the loop

Unlike traditional systems, where user interactions happen at system boundaries,
AI and humans are often equal participants in generative computations. An
example is Reinforcement Learning from Human Feedback (RLHF), where humans
provide feedback on AI output, only to be replaced by an AI model that mimics
their feedback behavior during the training process. In other cases, critical
generative flows (e.g., the drafting of legal documents) may require a human
verification step, without which the system provides subpar results.

The choice between sampling human input and sampling a generative model is
involves considerations such as cost and quality, and may be subject to change
during the development of a system. For example, an early deployment of a system
may heavily sample outputs or feedback from human participants, until there is
enough data to bootstrap the target distribution using automated generative
components.

Therefore, a generative system should be designed to utilize human input in the
same way it would interact with an API or subsystem.

### Data is code, code is data

Traditional software systems tend to allow a clear distinction between code
(written by developers) and data (generated by the user and the system). In
generative systems this distinction breaks down: Approaches such as
[AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) or
[Voyager](https://github.com/MineDojo/Voyager) use generative AI to generate
programs (specified in code or plain text), which in turn may be interpreted by
generative AI systems; prompts for chat-based generative AI could equally be
considered code or data.

## Development and Contributing

Enact is currently in alpha release. The framework is open source and Apache
licensed. We are actively looking for contributors that are excited about the
vision.

You can download the source code, report issues and create pull requests at
https://github.com/agentic-ai/enact.

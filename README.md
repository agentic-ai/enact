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

## Why enact?

The rise of generative AI models is transforming the software development
process.

Traditional software relies primarily on functional buildings blocks in which
inputs and system state directly determine outputs. In contrast, modern software
increasingly utilizes generative AI elements, in which each input is associated
with a range of possible outputs.

This seemingly small change in emphasis - from functions to conditional
distributions - implies a shift across multiple dimensions of the engineering
process, summarized in the table below.

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
one of the major goals of generative software engineering is then to fit the
conditional distribution represented by the system as a whole to a target
distribution. This target may be specified implicitly, e.g., via sampled human
inputs or corrections, or explicitly, e.g., by a task reward or example outputs.

In many cases, the process of ongoing finetuning and improvement does not
have a clearly defined endpoint. This is a departure from traditional software
engineering, where a feature - once it is implemented and bug-free - requires
only maintenance-level engineering work.

In some cases, fine-tuning generative systems can be achieved directly using
machine learning techniques and frameworks. For instance, systems that consist
of a thin software wrapper around a single large language model (LLM) or
text-to-image model may directly train the underlying model to shape system
behavior.

In cases where multiple generative components work together or are called in
autoregressive feedback loops, the system as a whole may need to be fit to the
target distribution, which in addition to ML training may involve A/B testing
between API-identical subsystems and the refinement of simple generative
components into complex algorithmic flows that structure the generation process.

### Fitting generative software is a recursive process.

In traditional software engineering, the choice between two API-identical
implementations primarily revolves around practicalities such as performance or
maintainability. However, in generative software, individual components produce
distributions that may be more or less well-fitted to the system's overall goal,
for example, one foundation text-to-image model may outperform another when the
task is to generate images in a certain style, even though it might not be a
better image generator in general.

Generative system outputs are distributions that optimized towards some, often
implicitly specified, target. Data selection, training parameters, model
composition, sampling of feedback and self-improvement flows produce systems
whose conditional output distributions represent a unique, opinionated take on
the problem they were trained to solve. Therefore the development of generative
systems motivates ongoing reevaluation, tuning and replacement of subsystems,
more so than in traditional engineering applications.

This optimization process involves, on the one hand, recursive elaboration, in
which a component is replaced by a structured generative flow or algorithm. On
the other hand, it involves 'compression' of the resulting data into end-to-end
trained models. For example, a simple call into an LLM can be elaborated by
exploring a tree of possible completions before settling on a final result, and
the resulting executions could be used to fine-tune the original LLM.


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
structure of code is programmatically interpretable by other code. In addition
to this, generative software requires the ability to reflect on code executions.

A generative software system specifies a distribution of outputs only
implicitly. Sampling from it may be computationally intensive and provides
valuable information that may be used as system feedback or training data. Since
generative software may include complex, recursive algorithms over generative
components, this suggests a need for not simply tracking inputs and outputs
at system boundaries, but at every level of execution.

This is particularly relevant to higher order generative flows, in which the
output of one generative step structures the execution of the next: Such a
system can use call traces as feedback to improve its output, in the same way
that a human may use a debugger to step through an execution to debug an issue.

In contrast, conventional software tracks executions at a low level of
resolution, e.g., using logging or monitoring frameworks, since their primary
use is to ensure system health and debug errors that occur in deployment.


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

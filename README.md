# enact - A Framework for Generative Software.

Enact is a python framework for building generative software, specifically
software that integrates with machine learning models or APIs that generate
distributions of outputs.

The advent of generative AI is driving changes in the way software is built.
The unique challenges of implementing, maintaining and improving generative
systems indicate a need to rethink the software stack from a first-principles
perspective. See [why-enact](#why-enact) for a more in-depth discussion.

The design philosophy of enact is to provide an easy-to-use python framework
that addresses the needs of emerging AI-based systems in a fundamental manner.
Enact is designed as a core framework that provides low-level primitives
required by generative software systems.

To this end, enact provides support for the following features:
* The ability to commit data, generative components and executions to
  persistent storage in a versioned manner.
* Journaled executions of generative python components.
* The ability to rewind and replay past executions.
* Easy interchangeability of human and AI-driven subsystems.
* Support for all of the above features in higher-order generative flows, i.e.,
  generative programs that generate and execute other generative flows.
* A simple hash-based storage model that simplifies implementation of
  distributed and asynchronous generative systems.

## Installation and overview

Enact is available as a [pypi package](https://pypi.org/project/enact/) and can
be installed via:

```bash
pip install enact
```

Enact wraps generative components as python dataclasses with annotated input and
output types:

```python
import enact

import dataclasses
import random

@enact.typed_invokable(input_type=type(None), output_type=int)
@dataclasses.dataclass
class RollDie(enact.Invokable):
  """An enact invokable that rolls a die."""
  sides: int

  def call(self) -> int:
    return random.randint(1, self.sides)

roll_die = RollDie(sides=6)
print(roll_die())  # Print score
```

Data and generative components can be committed to and checked out of persistent
storage.

```python
with enact.FileStore('./enact_store') as store:
  roll_die_v0 = enact.commit(roll_die)  # Return a reference.
  roll_die.sides = 20
  print(roll_die())                     # Roll 20 sided die.
  roll_die = roll_die_v0.checkout()     # Check out 6 sided die.
  print(roll_die())                     # Roll 6 sided die.

```

Executions can be journaled with the `invoke` command.

```python
@enact.typed_invokable(int, int)
@dataclasses.dataclass
class RollDice(enact.Invokable):
  """Roll the indicated number of dice."""
  die: enact.Invokable

  def call(self, num_rolls: int) -> int:
    return sum(self.die() for _ in range(num_rolls))

roll_dice = RollDice(roll_die)

with store:
  num_rolls = enact.commit(3)  # commit input to store.
  invocation = roll_dice.invoke(num_rolls)  # create journaled execution.
  print(invocation.get_output())  # Print sum of 3 rolls
```

Invocations allow investigating details of the execution and allow for
advanced features such as rewind/replay.

```python
# Print individual dice rolls.
with store:
  for i in range(3):
    roll_result = invocation.get_child(i).get_output()
    print(f'Roll {i} was {roll_result}')

# Rewind the last roll and replay it.
with store:
  two_rolls = invocation.rewind(1)        # Rewind by one call.
  print(two_rolls.replay().get_output())
```

See the [quickstart](examples/quickstart.ipynb) and
[enact concepts](examples/enact_concepts.ipynb) for more information.

## Documentation

Full documentation is work in progress. Please take a look at the
'examples' directory.

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

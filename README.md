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
* A simple hash-based storage model that simplifies distributed and
  asynchronous generative flows.

## Installation and overview

Enact is available as a [pypi package](https://pypi.org/project/enact/) and can
be installed via:

```bash
pip install enact
```

Enact defines generative components as python classes with annotated input and
output types:

```python
import enact

import dataclasses
import random

@enact.typed_invokable(
  input_type=enact.NoneResource,
  output_type=enact.Int)
@dataclasses.dataclass
class RollDie(enact.Invokable):
  """An enact invokable that rolls a die."""
  sides: int

  def call(self):
    return enact.Int(random.randint(1, self.sides))


@enact.typed_invokable(
  input_type=enact.Int,
  output_type=enact.Int)
@dataclasses.dataclass
class RollDice(enact.Invokable):
  """An enact invokable that rolls a specified number of dice."""
  roll: enact.Invokable
  def call(self, num_dice: enact.Int):
    return enact.Int(sum(self.roll() for _ in range(num_dice)))

roll_dice = RollDice(RollDie(6))
print(roll_dice(enact.Int(3)))   # Print sum of 3 rolls.
```

Executions can be journaled and committed to persistent storage. A journaled
execution supports rewinding and replaying.

```python
with enact.FileStore('/tmp/my_store') as store:
  num_rolls = enact.commit(enact.Int(3))  # commit input to store.
  invocation = roll_dice.invoke(num_rolls)  # create journaled execution.

  print(invocation.get_output())  # Print sum of 3 rolls
  for i in range(3):
    print(invocation.get_child(i).get_output())  # Print each die roll.

  invocation = invocation.rewind()  # Rewind by one dice roll.
  invocation = invocation.replay()  # Replay dice roll 1 & 2, resample roll 3.
  print(invocation.get_output())
```

Human input can be flexibly swapped in for generative components:

```python
@enact.typed_invokable(enact.NoneResource, enact.Int)
class HumanRollsDie(enact.Invokable):
  def call(self):
    return enact.request_input(
      enact.Int, context='Please roll a six-sided die')

with store:
  roll_dice = RollDice(HumanRollsDie())
  inv_gen = enact.InvocationGenerator(roll_dice, num_rolls)
  for input_request in inv_gen:
    inv_gen.set_input(enact.Int(6))  # Provide a roll of 6.

print(inv_gen.invocation.get_output())  # Prints '18'.
```

## Documentation

Full documentation is work in progress. A quickstart tutorial, explanation of
enact concepts and examples can be found in the
[examples directory](https://github.com/agentic-ai/enact/tree/main/examples).

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

|        |  Traditional  |  Generative  |
| :----: | :-----------: | :----------: |
| Building block | Deterministic functions | Conditional distributions |
| Engineering | Add features, debug errors | Improve output distributions |
| Subsystems | Interchangeable | Unique |
| Code sharing | Frameworks | Components |
| Executions | Logged for metrics/debugging | Training data |
| Interactivity | At system boundaries | Within system components |
| Code vs data | Distinct | Overlapping |

### Building generative software means fitting distributions

In traditional software, a large part of engineering effort revolves around
implementing features and debugging errors. In contrast, a generative software
system may be feature-complete and bug-free, but still not suitable for
deployment. Consider the following examples:

* A search chatbot that is sometimes rude and unhelpful
* An autonomous agent that recursively sets itself goals and accomplishes tasks,
  but has a tendency to drift off into behavioral loops.
* An AI avatar generator that produces accurate but unattractive portrait
  images.

System correctness is no longer merely a question of specific inputs leading
to outputs that are either correct or incorrect, but of the _distribution of
outputs_ satisfying some implicitly or explicitly defined quality target.

In cases where generativity is localized, e.g., when the software is a thin
wrapper around a large language model (LLM) or image generator, machine-learning
tools and techniques can directly be used to improve the system, but in cases
where multiple generative components work together to produce an output the
system must be fitted to the target distribution as a whole.

### Generative software requires recursively swapping subsystems

In traditional software engineering, the choice between two API-identical
implementations primarily revolves around practicalities such as performance or
maintainability. However, in generative software, individual components produce
distributions that may be more or less well-fitted to the system's overall goal,
for example:

* A foundation text-to-image model may perform well on a wide range of
prompts, whereas an API-identical fine-tuned version may be less general but
produce better results for images of a particular style.
* An instruction-tuned LLM and an LLM trained as a chatbot both autoregressively
extend token sequences, but one may be better suited towards data processing
applications, while another will make a better math tutor.
* Different components may have vastly different execution costs.

Generative system outputs are distributions that optimized towards some - often
implicitly specified - target. Data selection, training parameters, model
composition, sampling of feedback and self-improvement flows produce systems
whose conditional output distributions represent a unique, opinionated take on
the problem they were trained to solve. Therefore the development of generative
systems motivates ongoing reevaluation, tuning and replacement of subsystems,
more so than in traditional engineering applications.

### Generative software should be shareable

There are large numbers of generative AI components that are mutually
API-compatible (e.g., text-to-image generation, LLMs) but cannot be directly
ranked in terms of quality since they represent a unique take on the problem
domain. The combination of API compatibility and variation lends itself to a
more evolutionary collaborative style than traditional software, where
shared effort tends to centralize into fewer, lower-level frameworks.

This new collaborative approach is already evidenced by widespread sharing of
prompt templates and the existence of public databases of fine-tuned models.

### Generative software reflects on its executions.

In conventional software, executions are typically tracked at a low level
of resolution, since their primary use is to track metrics and debug errors.
A generative system represents an implicitly specified distribution, and
its execution history provides valuable information that may be used for
analysis or training.

* System outputs can be corrected, scored or critiqued by humans or AI
models to produce data for fine-tuning.
* Data from one generative model can be distilled into another.
* Complex orchestrations of different ML models can be replaced by end-to-end
trained models once sufficient data is available.
* Failed executions are potential input data for generative systems, e.g., in
the case of program synthesis and 'self-healing programs'.

### Humans are in the loop

Unlike traditional systems, where user interactions happen at system boundaries,
AI and humans are often equal participants in generative computations. An
example is Reinforcement Learning from Human Feedback (RLHF), where humans
provide feedback on AI output, only to be replaced by an AI model that mimics
their feedback behavior during the training process. Critical generative flows
(e.g., the drafting of legal documents) may require a human verification step.

The choice between sampling human input and sampling a generative model is
one of cost, quality and timing constraints, and may be subject to change during
the development of a system. Therefore, ideally, a generative system will be able
to sample output from a human in the same way that it would call into an API or
subsystem.

### Data is code, code is data

Traditional software systems tend to allow a clear distinction between code
(written by developers) and data (generated by the user and the system). In
generative systems this distinction breaks down: Approaches such as AutoGPT or
Voyager use generative AI to generate programs (specified in code or plain
text), which in turn may be interpreted by generative AI systems; prompts for
chat-based generative AI could equally be considered code or data.

## Development and Contributing

Enact is currently in alpha release. The framework is open source and Apache
licensed. We are actively looking for contributors that are excited about the
vision.

You can download the source code, report issues and create pull requests at
https://github.com/agentic-ai/enact.

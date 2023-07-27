# enact - A Framework for Generative Software.

Enact is a python framework for building generative software, that is,
software that calls into generative machine learning models or APIs.
The enact framework makes it easy to:
* serialize and persist data, programs and program executions,
* programmatically reflect on program executions and human feedback,
* automatically generate UIs for complex generative flows which
  alternate human and AI-driven steps,
* explore the tree of possible system executions,
* sample human input in a way that is API identical to calling
  a generative AI component,
* write higher-order generative flows, e.g., generative AI components that
  produce or modulate other generative AI components.

See [here](#why-enact) for an explanation of the significance of these features
in the context of generative software.


## Installation and Quick start

Enact is available as a [pypi package](https://pypi.org/project/enact/) and can
be installed via:

```bash
pip install enact
```

### Defining and storing custom resources
The atomic unit of enact is the `Resource`, a data-carrying object that can be
serialized (e.g., as JSON) and is addressable via a unique hash. Resources are
implemented as python dataclasses and need to be registered with enact to allow
for deserialization:

```python
import enact
import dataclasses

@enact.register
@dataclasses.dataclass
class MyResource(enact.Resource):
  x: int
  y: float
```

Resources can be committed to a store, which will return a reference. References
store the hash of the committed resource as a unique identifier.

```python
with enact.Store() as store:
  ref = enact.commit(MyResource(42, 69.0))
  print(ref.id)  # Prints hash digest of the resource.
  print(ref.get())  # Prints "MyResource(x=42, y=69.0)".
```

### Invoking resources
While some resources represent data, others represent executable code. These
resources, called invokables, define a call function and may be annotated
with typing information to allow for advanced features such as automatic
generation of UIs:

```python
@enact.typed_invokable(input_type=enact.NoneResource, output_type=MyResource)
class MyInvokable(enact.Invokable):

  def call(self):
    return MyResource(42, 69.0)
```

Invokables can either be called directly, or they can be invoked, in which
case a full call-graph of the execution is returned in the form of a special
`Invocation` resource. An invocation references inputs, outputs and any
recursive subinvocations.

```python
with store:
  my_invokable = MyInvokable()
  # Simple execution:
  print(my_invokable())  # Prints "MyResource(x=42, y=69.0)".
  # Tracked execution:
  invocation = my_invokable.invoke()
  enact.pprint(invocation)  # Prints call graph of execution.
```

### Creating UIs

Since invokables carry type annotations, enact can auto-generate a UI.

```python
with store:
  ref = enact.commit(my_invokable)
  enact.GUI(ref).launch(share=True)
```

This will open a Gradio UI with a run button that can be used to invoke the
resource.

### Requesting inputs and replaying invocations
Invocations that end in an exception can be continued by replacing the raised
exception with an injected value. This allows suspending an execution in order
to collect information from a human user or other data source.

```python
@enact.typed_invokable(input_type=enact.NoneResource, output_type=MyResource)
class SampleFromHuman(enact.Invokable):

  def call(self):
    request_int = enact.RequestInput(enact.Int)
    request_float = enact.RequestInput(enact.Float)
    return MyResource(
      x=request_int(enact.Str('Please provide an x-value for MyResource.')),
      y=request_float(enact.Str('Please provide a y-value for MyResource.')))

with store:
  h = SampleFromHuman()
  # Run until first input request.
  invocation = h.invoke()
  # Access InputRequest exception.
  input_request = invocation.response().raised()
  print(input_request.input())  # Prints 'Please provide an x-value ...'.
  # Run until second input request.
  invocation = input_request.continue_invocation(invocation, enact.Int(42))
  # Access InputRequest exception.
  input_request = invocation.response().raised()
  print(input_request.input())  # Prints 'Please provide a y-value ...'.
  # Run until completion.
  invocation = input_request.continue_invocation(invocation, enact.Float(69.0))
  print(invocation.response().output())  # Prints 'MyResource(x=42, y=69.0)'.
```

The `continue_invocation` function makes use of the replay feature, which allows
replaying a previous invocation while overriding previously encountered
exceptions with injected inputs:

```python
with store:
  # Run until first exception.
  invocation = h.invoke()
  def override_exception(exc_ref):
    if exc_ref().requested_type == enact.Int:
      return enact.Int(42)
    if exc_ref().requested_type == enact.Float:
      return enact.Float(69.0)
  # Inject first value and run until second exception.
  invocation = invocation.replay(override_exception)
  # Inject second value and run until completion.
  invocation = invocation.replay(override_exception)
  print(invocation.response().output())  # Prints 'MyResource(x=42, y=69.0)'.
```

This mechanism also allows UIs to sample inputs from humans. The type of input
must be preregistered on UI launch:

```python
with store:
  ref = enact.commit(h)
  enact.GUI(ref, input_required_inputs=[enact.Int, enact.Float]).launch(
    share=True)
```

## Usage / Examples

A list of ipython notebook examples, including the code in the quickstart
section can be found in the
[examples](https://github.com/agentic-ai/enact/tree/main/examples) directory.

## Why enact?

The rise of generative AI models is transforming the software development
process.

Traditional software relies primarily on functional buildings blocks in which
inputs and system state directly determine outputs. In contrast, modern software
increasingly utilizes generative elements, in which each input is associated
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
system may be feature complete and bug-free, but still not suitable for
deployment. Consider the following examples:

* A search chatbot that is sometimes rude and unhelpful
* An autonomous agent that recursively sets itself goals and accomplishes tasks,
but has a tendency to drift off into behavioral loops.
* An AI avatar generator that produces accurate but attractive portrait images.

System correctness is no longer merely a question of specific inputs leading
to outputs that are either correct or incorrect, but of the _distribution of
outputs_ satisfying some quality target.

In cases where generativity is localized, e.g., when the software is a thin
wrapper around a large language model (LLM) or image generator, machine-learning
tools and techniques can directly be used to improve the system, but in cases
where multiple generative-components work together to produce an output, or a
single generative model is called repeatedly, the system must be fitted to the
target distribution as a whole.

### Generative software requires recursively swapping subsystems

In traditional software engineering, the choice between two API-identical
implementations primarily revolves around practicalities such as performance or
maintainability. However, in generative software, individual components produce
distributions that may be more or less well-fitted to the system's overall goal,
for example:

* A foundation text-to-image model may perform well on a wide range of
prompts, whereas an API-identical fine-tuned version may be less general but
produce better results for images in a particular style.
* An instruction-tuned LLM and an LLM trained as a chatbot both autoregressively
extend token sequences, but one will tend to be better suited towards data
processing applications, while the other will make a better math tutor.

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

This new collaborative approach is evidenced by:
* Prompt sharing in image generators and LLMs.
* Public databases of fine-tuned models.

### Generative software reflects on its executions.

In conventional software, executions are typically tracked at a low level
of resolution, since their primary use is to track metrics and debug errors.
A generative system represents an implicitly specified distribution, and
its execution history provides valuable information that may be used for
training.

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


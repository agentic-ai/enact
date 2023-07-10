# enact - A Framework for Generative Software.

## Introduction

TODO: Intro, core features.

## Why enact?

With the rise of generative AI models, we are witnessing a significant shift in
how software systems are conceptualized and built.

Traditional software relies on functional buildings blocks in which inputs and
system state directly determine outputs. In contrast, modern AI-powered software
utilizes generative elements, in which each input is associated with a range of
possible outputs.

This seemingly small change - from functions to conditional distributions -
implies a shift in focus across multiple dimensions of the engineering process,
summarized in the table below.

|        |  Traditional  |  Generative  |
| :----: | :-----------: | :----------: |
| Building block | Deterministic functions | Conditional distributions |
| Engineers | Add features, debug errors | Improve output distributions |
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

* A search chatbot that is feature complete, but becomes rude and unhelpful.
* An autonomous agent that recursively accomplishes tasks, but has a tendency to
drift off into behavioral loops.
* An AI avatar generator that produces accurate but unattractive portrait
images.

System correctness is no longer merely a question of specific inputs leading
to correct outputs, but of the distribution of outputs satisfying some
quality target when the system is deployed.

In cases where generativity is localized, e.g., when the software is a thin
wrapper around a large language model (LLM) or image generator, machine-learning
tools and techniques can directly be used to improve the system, but in cases
where multiple generative-components work together to produce an output, or a
single generative model is called repeatedly, the system must be fitted as a whole
to the target distributions.

### Generative software requires recursively swapping subsystems

In traditional software engineering, the choice between two API-identical
implementations primarily revolves around practicalities such as performance or
maintainability. However, in generative software, individual components produce
distributions that may be more or less well-fitted to the system's overall goal,
for example:

* Two ML models that perform an API-identical transformation from an input text
prompt to an output image may be differently suited towards particular styles or
subjects.
* An instruction-tuned LLM and an LLM trained as a chatbot both autoregressively
extend token sequences, but one will tend to be better suited towards data
processing applications, while the other will make a better math tutor.

Generative system outputs are fitted distributions, and their target is only
implicitly specified. Data selection, training parameters, model composition,
sampling of feedback and self-improvement flows produce systems whose
conditional output distributions represent a unique, opinionated take on the
problem they were trained to solve. Therefore the development of generative
systems involves recursively swapping out and comparing subsystems.

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

## Installation

## Quick start

## Usage / Examples

## API Reference

## Contributing




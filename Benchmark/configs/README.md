# Benchmark Configs

_Status updated on 2026-08-03._

## 📌 Current status

No standalone declarative configuration files are currently committed in this
directory. Experiment parameters are supplied through the command-line
interfaces in `../scripts/`, while prepared cohorts and their frozen settings
are recorded in `../results/`.

## 🧭 Configuration policy

Future reusable configs should contain only non-secret settings such as dataset
paths, split definitions, prompt versions, model identifiers, decoding
parameters, and output locations. API keys and other credentials must remain in
environment variables and must not be committed.

See the [benchmark guide](../README.md) and
[script inventory](../scripts/README.md) for the currently supported options.

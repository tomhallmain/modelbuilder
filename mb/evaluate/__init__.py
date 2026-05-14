"""
Evaluation workflows for ``mb evaluate`` (classification checkpoints vs prepared data).

Layout (mirror sibling concerns, avoid duplicating generic code elsewhere):

- ``mb.data`` — ``ModelBuildStepCommand`` pipeline (gather, convert, dataset layout). Eval
  reads those trees; it does not reimplement layout rules: reuse ``mb.data.class_layout``,
  paths from ``mb.pipeline_config``, and media rules from ``mb.data.file_types`` where relevant.
- ``mb.training`` — training loops and hyperparameters. Eval may reuse run/snapshot metadata
  and ``mb.training.run_args`` for provenance; it does not own training.
- ``mb.conversion`` — model format detection / loading at inference boundaries. Prefer
  ``mb.conversion.converters`` (or thin wrappers in this package) instead of a second loader stack.
- ``mb.info_inspect`` — read-only dataset and model summaries for ``mb info``. Eval can call
  those helpers for text reports; keep a single implementation for dataset walk/count logic
  rather than copying it here.
- **This package** — one module per current ``EvaluateSubcommand`` value
  (``metrics``, ``misclassified``, ``compare``). Shared eval-only glue (e.g. building a
  ``DataLoader`` from the same assumptions as training) should live in a small internal
  module here (e.g. ``_runtime.py``) once needed, not under ``mb.data`` (data prep) or
  ``mb.models`` (architecture registration).

See ``EvaluateSubcommand`` in ``mb.models.types`` for future *benchmark* / *calibrate*
ideas before they become CLI subcommands.
"""

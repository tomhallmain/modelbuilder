# `mb` package

Core **Model Builder** library: CLI entry (`mb.cli`), data pipeline, training orchestration, model conversion, and shared utilities.

**Usage and install:** repository [README.md](../README.md). **Design:** [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) (this tree is summarized there).

```text
mb/
  cli.py, config.py
  data/          # gather, convert, dedupe, upscale, dataset
  models/        # types, frameworks (pytorch, keras), registry
  training/      # ModelTrainer, hyperparams
  conversion/    # ONNX, SafeTensors, …
  utils/
```

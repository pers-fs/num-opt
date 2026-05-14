# Contributing

Thank you for your interest in contributing. This document explains how to extend the benchmark with new optimizers, tasks, datasets, and models.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-dev
```

Run the linter and type checker before opening a PR:

```bash
make check
```

---

## Adding a new optimizer

1. Open `benchmark/optimizers/advanced.py` (custom) or create a new file under `benchmark/optimizers/`.
2. Implement your optimizer as a subclass of `torch.optim.Optimizer`.
3. Register it in the `register_advanced_variants()` function (or your new file's registration function):

```python
@register_optimizer("my_optimizer")
def _my_optimizer(params, lr: float = 1e-3):
    return MyOptimizer(params, lr=lr)
```

4. Add its default learning rate to the `DEFAULT_LRS` dict in `benchmark/main.py`.
5. Document it in `all_instructions/optimizers.md` following the existing format (paper, idea, update rule, implementation notes).

---

## Adding a new task

1. Create `benchmark/tasks/<task_name>.py`.
2. Subclass `BaseTask` and implement all abstract methods:
   - `task_name()` — unique string key
   - `primary_metric_key()` — metric name used for model selection
   - `is_better(current, best)` — comparison direction
   - `train_step(model, batch)` → `(loss, outputs, targets)`
   - `eval_step(model, batch)` → `(loss, outputs, targets)`
3. Decorate the class with `@register_task("task_name")`.
4. Import the module inside `ensure_imports()` in `benchmark/config/registry.py`.

---

## Adding a new dataset

1. Create `benchmark/datasets/<task_family>/<dataset_name>.py`.
2. Define a builder function with the signature:

```python
def build_dataset(max_samples: Optional[int], data_root: str) -> Tuple[Dataset, Dataset]:
    ...
    return train_dataset, val_dataset
```

3. Decorate it with `@register_dataset("dataset_name")`.
4. Use `maybe_limit_split` from `benchmark.datasets.base_dataset` to honour `max_samples`.
5. Import the module inside `ensure_imports()` in `benchmark/config/registry.py`.

---

## Adding a new model

1. Create `benchmark/models/<task_family>/<model_name>.py`.
2. Define a builder function:

```python
def build_model(task, dataset) -> nn.Module:
    ...
```

3. Decorate it with `@register_model("model_name")`.
4. Import the module inside `ensure_imports()` in `benchmark/config/registry.py`.

---

## Code style

- Line length: 120 characters.
- Linting: `ruff check benchmark/` — no errors allowed.
- Type annotations: add them to all new public functions.
- Comments: only when the *why* is non-obvious. No docstrings restating the function name.

---

## Pull request checklist

- [ ] `make check` passes with no errors
- [ ] Smoke run passes: `make smoke`
- [ ] New optimizer/task/dataset/model is documented in the relevant `all_instructions/` file
- [ ] Default LR or scheduler preset added where applicable

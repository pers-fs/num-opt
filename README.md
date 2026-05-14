# Numerical Optimizer Benchmark

A reproducible benchmark comparing **16 optimization algorithms** across **8 deep learning task families**, built on PyTorch. Designed for research on optimizer behaviour across diverse problem types.

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: CC0](https://img.shields.io/badge/license-CC0%201.0-lightgrey)](LICENSE)
[![CI](https://github.com/filippostruffi/num-opt/actions/workflows/ci.yml/badge.svg)](https://github.com/filippostruffi/num-opt/actions/workflows/ci.yml)

---

## Overview

Each benchmark run trains a fresh model with a single optimizer, evaluates on a validation set after every epoch, and writes metrics, plots, and checkpoints to an organized output directory. Runs are **resumable**: if an optimizer's output directory already exists it is skipped automatically.

**What gets measured per optimizer:**
- Best validation metric and the epoch at which it occurred
- Best validation loss
- Wall-clock time to best metric and to convergence
- Per-epoch primary metric and loss (JSON + CSV)
- Learning-curve plot (primary metric + val loss on dual axes)

---

## Quick start

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. (Optional) redirect caches to a local directory
export TORCH_HOME=./data
export HF_HOME=./data/huggingface
export HF_DATASETS_CACHE=./data/huggingface
export TOKENIZERS_PARALLELISM=false

# 3. Run
python -m benchmark.main \
  --task image_classification \
  --dataset cifar10 \
  --model cnn_scratch \
  --optimizers all \
  --epochs 5 \
  --batch-size 64 \
  --output-dir ./results
```

Or use the Makefile:

```bash
make install
make smoke      # quick sanity check: cifar10 + 3 optimizers, 128 samples
make run-all    # all task × model × dataset combinations
```

---

## Supported combinations

| Task | Datasets | Models | Primary metric |
|------|----------|--------|---------------|
| `image_classification` | `cifar10`, `mnist` | `cnn_scratch`, `resnet18` | accuracy |
| `semantic_segmentation` | `oxford_iiit_pet`, `voc2012` | `unet`, `deeplabv3_resnet50` | miou |
| `sentiment_analysis` | `sst2`, `imdb` | `lstm_sentiment`, `distilbert_sentiment` | accuracy |
| `ner` | `conll2003_hf`, `wikiann_en` | `lstm_crf_ner`, `lstm_crf_charcnn_ner` | f1 |
| `text_generation` | `wikitext2`, `ptb_text_only_hf` | `gpt_small`, `gru_lm` | perplexity |
| `text_summarization` | `cnn_dailymail`, `aeslc` | `bart_small`, `tiny_transformer_seq2seq` | rouge |
| `machine_translation` | `iwslt14_en_de`, `europarl_bilingual` | `transformer_seq2seq`, `lstm_seq2seq` | bleu |
| `question_answering` | `squad_v1`, `tweet_qa` | `transformer_qa`, `bilstm_attention_qa` | f1 |

---

## Optimizers

### Standard (`torch.optim`)

| Key | Algorithm | Default LR |
|-----|-----------|-----------|
| `sgd` | SGD | 0.05 |
| `sgd_momentum` | SGD + momentum (0.9) | 0.01 |
| `rmsprop` | RMSprop | 1e-3 |
| `adagrad` | Adagrad | 1e-2 |
| `adadelta` | Adadelta | 1.0 |
| `adam` | Adam | 1e-3 |
| `adamw` | AdamW | 1e-3 |
| `radam` | RAdam | 1e-3 |

### Custom implementations (`benchmark/optimizers/advanced.py`)

| Key | Algorithm | Reference | Default LR |
|-----|-----------|-----------|-----------|
| `lion` | Lion | Chen et al., 2023 | 1e-4 |
| `lars` | LARS | You et al., 2017 | 0.1 |
| `lamb` | LAMB | You et al., 2019 | 1e-3 |
| `adabelief` | AdaBelief | Zhuang et al., 2020 | 1e-3 |
| `yogi` | Yogi | Zaheer et al., 2018 | 1e-3 |
| `adafactor` | Adafactor | Shazeer & Stern, 2018 | 1e-3 |
| `sam` | SAM | Foret et al., 2021 | 1e-3 |
| `gsam` | GSAM (approx.) | Zhuang et al., 2022 | 1e-3 |

See [`all_instructions/optimizers.md`](all_instructions/optimizers.md) for algorithm details and weight-decay heuristics.

---

## CLI reference

```
python -m benchmark.main \
  --task          <task>                      # required
  --dataset       <dataset>                   # required
  --model         <model>                     # required
  --optimizers    <name,name,...|all>         # required; "all" runs every registered optimizer
  --epochs        N                           # default: 1
  --batch-size    N                           # default: 64
  --max-samples   N                           # cap samples per split (useful for debugging)
  --output-dir    PATH                        # default: ./results
  --data-root     PATH                        # dataset/model cache root; default: ./data
  --lr-policy     {per_opt,fixed}             # default: per_opt (uses table above)
  --learning-rate FLOAT                       # used only when --lr-policy fixed
  --lr-overrides  "sgd=0.05,lars=0.1"        # override specific optimizers in per_opt mode
  --scheduler     {auto,cosine,plateau,none}  # default: auto (picks by task family)
  --max-grad-norm FLOAT                       # global-norm gradient clipping; <=0 disables
  --plateau-patience N                        # epochs without improvement before plateau; default: 3
  --num-workers   N                           # DataLoader workers; default: 2
  --seed          N                           # default: 42
```

### Learning rate policy

| Flag | Behaviour |
|------|-----------|
| `--lr-policy per_opt` *(default)* | Each optimizer uses its own tuned default (see table above) |
| `--lr-policy fixed --learning-rate 1e-3` | All optimizers share the same LR |
| `--lr-overrides "sgd=0.05,lars=0.1"` | Override specific optimizers while keeping per-opt defaults for the rest |

### Scheduler and gradient clipping (auto mode)

| Task family | Scheduler | Max grad norm |
|-------------|-----------|---------------|
| `image_classification` | cosine | — |
| `semantic_segmentation` | cosine | 1.0 |
| NLP / sequence tasks | plateau | 1.0 |

---

## Output structure

```
results/
└── <epochs>/
    └── <task>/
        └── <model>/
            └── <dataset>/
                ├── run_metadata.json
                ├── details.txt
                ├── <task>_<model>_<dataset>_summary.csv    # one row per optimizer
                ├── <task>_<model>_<dataset>_extensive.csv  # one row per optimizer × epoch
                └── <optimizer>/
                    ├── <optimizer>_metrics.json
                    └── <optimizer>_<primary_metric>_and_loss.png
```

`summary.csv` — one row per optimizer: best primary metric value/epoch, runtime to best, convergence epoch and time.  
`extensive.csv` — one row per optimizer per epoch: primary metric, val loss, and wall-clock runtimes.

---

## Running all combinations

```bash
# Default: 1 epoch, batch size 64, 2 samples per split
bash all_instructions/run_all.sh

# Override defaults
BS=32 EPOCHS=10 MAX_SAMPLES=5000 OPTIMIZERS="adam,adamw,lion" bash all_instructions/run_all.sh
```

---

## Extending the benchmark

See [CONTRIBUTING.md](CONTRIBUTING.md) for step-by-step instructions on adding a new optimizer, task, dataset, or model.

---

## License

Released under [CC0 1.0 Universal](LICENSE) — no rights reserved.

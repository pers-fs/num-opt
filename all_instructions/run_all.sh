#!/usr/bin/env bash
# run_all.sh — Run every registered task / model / dataset combination.
#
# Defaults can be overridden via environment variables:
#   BS=32 EPOCHS=3 MAX_SAMPLES=1000 bash all_instructions/run_all.sh
#
# Set OPTIMIZERS to run a subset, e.g. OPTIMIZERS="adam,adamw,sgd_momentum"
# Leave OPTIMIZERS unset (or "all") to benchmark every registered optimizer.

set -euo pipefail

BS="${BS:-64}"
EPOCHS="${EPOCHS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
OPTIMIZERS="${OPTIMIZERS:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
DATA_ROOT="${DATA_ROOT:-./data}"

run() {
  local task="$1" dataset="$2" model="$3"
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  task=${task}  dataset=${dataset}  model=${model}"
  echo "══════════════════════════════════════════════════════════════"
  python -m benchmark.main \
    --task        "${task}" \
    --dataset     "${dataset}" \
    --model       "${model}" \
    --optimizers  "${OPTIMIZERS}" \
    --epochs      "${EPOCHS}" \
    --batch-size  "${BS}" \
    --max-samples "${MAX_SAMPLES}" \
    --output-dir  "${OUTPUT_DIR}" \
    --data-root   "${DATA_ROOT}"
}

# ── Image classification ───────────────────────────────────────────────────
run image_classification cifar10 cnn_scratch
run image_classification cifar10 resnet18
run image_classification mnist   cnn_scratch
run image_classification mnist   resnet18

# ── Semantic segmentation ─────────────────────────────────────────────────
run semantic_segmentation oxford_iiit_pet unet
run semantic_segmentation oxford_iiit_pet deeplabv3_resnet50
run semantic_segmentation voc2012         unet
run semantic_segmentation voc2012         deeplabv3_resnet50

# ── Sentiment analysis ────────────────────────────────────────────────────
run sentiment_analysis sst2 lstm_sentiment
run sentiment_analysis sst2 distilbert_sentiment
run sentiment_analysis imdb lstm_sentiment
run sentiment_analysis imdb distilbert_sentiment

# ── Named entity recognition ──────────────────────────────────────────────
run ner conll2003_hf  lstm_crf_ner
run ner conll2003_hf  lstm_crf_charcnn_ner
run ner wikiann_en    lstm_crf_ner
run ner wikiann_en    lstm_crf_charcnn_ner

# ── Text generation ───────────────────────────────────────────────────────
run text_generation wikitext2       gpt_small
run text_generation wikitext2       gru_lm
run text_generation ptb_text_only_hf gpt_small
run text_generation ptb_text_only_hf gru_lm

# ── Text summarization ────────────────────────────────────────────────────
run text_summarization cnn_dailymail        tiny_transformer_seq2seq
run text_summarization cnn_dailymail        bart_small
run text_summarization aeslc                tiny_transformer_seq2seq
run text_summarization aeslc                bart_small

# ── Machine translation ───────────────────────────────────────────────────
run machine_translation iwslt14_en_de      lstm_seq2seq
run machine_translation iwslt14_en_de      transformer_seq2seq
run machine_translation europarl_bilingual lstm_seq2seq
run machine_translation europarl_bilingual transformer_seq2seq

# ── Question answering ────────────────────────────────────────────────────
run question_answering squad_v1   transformer_qa
run question_answering squad_v1   bilstm_attention_qa
run question_answering tweet_qa   transformer_qa
run question_answering tweet_qa   bilstm_attention_qa

echo ""
echo "All combinations finished. Results saved to: ${OUTPUT_DIR}"

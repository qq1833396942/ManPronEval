#!/usr/bin/env bash
set -euo pipefail

cd /data

export HF_HOME=/hf_home
export HF_DATASETS_CACHE=datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WHISPER_MODEL_NAME=/whisper-large-v3
export SOLO_TRAIN_DIR=/data/train
export SOLO_VAL_DIR=/data/val
export SOLO_TEST_DIR=/data/test
export SOLO_OUTPUT_DIR=/data/check_data_largev3_lora_r16_alpha32_output

export LORA_R=16
export LORA_ALPHA=32
export LORA_DROPOUT=0.1

export PER_DEVICE_TRAIN_BATCH_SIZE=1
export PER_DEVICE_EVAL_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export WEIGHT_DECAY=0.01
export USE_GRADIENT_CHECKPOINTING=1
export PREDICTION_LOSS_ONLY=1

export USE_DATASET_MAP=0
export KEEP_IN_MEMORY=0
export USE_LIBROSA_AUDIO=0
export FEATURE_WRITER_BATCH_SIZE=16

export ENABLE_EVAL=1
export PREDICT_WITH_GENERATE=0
export ENABLE_VAL_EARLY_STOP=1
export VAL_EARLY_STOP_PATIENCE=4
export VAL_EARLY_STOP_THRESHOLD=0.0
export VAL_SMOKE_TEST_SAMPLES=64
export ENABLE_TEST_EARLY_STOP=0
export PROBE_EVERY_STEPS=0
export SAVE_TOTAL_LIMIT=20
export RESUME_TRAINING=0

rm -rf "$SOLO_OUTPUT_DIR"
python -u check_data_lora_r16_no_earlystop.py

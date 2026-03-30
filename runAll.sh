#!/bin/bash

set -euo pipefail

# Example batch-training script for multiple scenes.
# Update the paths and GPU id below before running.

BASE_DATA_PATH="/path/to/dataset_root"
BASE_MODEL_PATH="./outputs/experiment_name"
LOG_DIR="./logs/experiment_name"
GPU_ID=0

SCENES=(
  "scene1"
  "scene2"
)

mkdir -p "$BASE_MODEL_PATH"
mkdir -p "$LOG_DIR"

for SCENE in "${SCENES[@]}"; do
  INPUT_PATH="${BASE_DATA_PATH}/${SCENE}"
  OUTPUT_PATH="${BASE_MODEL_PATH}/${SCENE}"
  DEPTH_PATH="${INPUT_PATH}/depths"
  LOG_FILE="${LOG_DIR}/${SCENE}.log"

  echo "========================================"
  echo "Scene: ${SCENE}"
  echo "Input: ${INPUT_PATH}"
  echo "Output: ${OUTPUT_PATH}"

  if [ ! -d "$INPUT_PATH" ]; then
    echo "Skip ${SCENE}: input path does not exist" | tee "$LOG_FILE"
    continue
  fi

  CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    -s "$INPUT_PATH" \
    -d "$DEPTH_PATH" \
    --eval \
    --model_path "$OUTPUT_PATH" \
    > "$LOG_FILE" 2>&1

  echo "Finished ${SCENE}, log saved to ${LOG_FILE}"
done

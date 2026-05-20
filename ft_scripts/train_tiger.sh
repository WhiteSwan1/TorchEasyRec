#!/bin/bash
# ============================================================
# TIGER training script (3-hierarchy, 8192 codes per layer)
#
# Input data: pre-SID-encoded parquet with columns:
#   - history_sids : list<int64>  (length = items × 3, codes in [0, 8192))
#   - label_sids   : list<int64>  (length = 3, codes in [0, 8192))
#   - (optional)    user_id : int64
#
# See ai_report/tiger_sample_data_format.md for the data contract.
# ============================================================

set -e

MODE="train"
MODEL_NAME="tiger"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${WORK_DIR}/tiger.config"
# Substitute with the actual training parquet glob.
DATA_PATH="${TIGER_TRAIN_DATA:-/workspace/fangtinglin/codework/TorchEasyRec/data/tiger_train/*.parquet}"
EVAL_DATA_PATH="${TIGER_EVAL_DATA:-/workspace/fangtinglin/codework/TorchEasyRec/data/tiger_eval/*.parquet}"
MODEL_DIR="${WORK_DIR}/experiments/${MODEL_NAME}"

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-32577}
NNODES=${NNODES:-1}
NPROC=${NPROC:-1}
NODE_RANK=${NODE_RANK:-0}

LOG_FILE="${WORK_DIR}/${MODEL_NAME}_${MODE}.log"
rm -f "${LOG_FILE}"

WORK_DIR="${WORK_DIR}/.."
cd "${WORK_DIR}"

echo "========================================"
echo "  TIGER Training"
echo "  Config:    ${CONFIG}"
echo "  Train:     ${DATA_PATH}"
echo "  Eval:      ${EVAL_DATA_PATH}"
echo "  Model dir: ${MODEL_DIR}"
echo "  GPUs/proc: ${NPROC}"
echo "  Work dir:  ${WORK_DIR}"
echo "========================================"

torchrun \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC}" \
    --node_rank="${NODE_RANK}" \
    -m tzrec.train_eval \
    --pipeline_config_path "${CONFIG}" \
    --train_input_path "${DATA_PATH}" \
    --eval_input_path "${EVAL_DATA_PATH}" \
    --model_dir "${MODEL_DIR}" \
    2>&1 | tee "${LOG_FILE}"

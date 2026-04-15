#!/bin/bash
# =============================================================================
# NAS3R 正式训练脚本 (re10k)
# 使用方法: cd /opt/tiger/mmfinetune/examples/nas3r && bash train_full.sh
# =============================================================================

set -e

WORK_DIR="/opt/tiger/mmfinetune/examples/nas3r"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"

cd "${WORK_DIR}"

echo "============================================"
echo "开始 NAS3R 正式训练 (re10k)"
echo "============================================"

# 检查GPU
nvidia-smi
echo ""

# 正式训练
python -m src.main \
    +experiment=nas3r/random/re10k \
    wandb.mode=disabled \
    wandb.name=nas3r_re10k \
    dataset.re10k.roots=["${LOCAL_DATA_DIR}/re10k"] \
    data_loader.train.batch_size=10 \
    data_loader.train.num_workers=16

echo "训练完成 ✓"

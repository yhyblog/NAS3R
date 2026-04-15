#!/bin/bash
# =============================================================================
# NAS3R 小批量测试脚本
# 用途: 用少量数据快速验证代码、环境、数据流是否正常
# 使用方法: cd /opt/tiger/mmfinetune/examples/nas3r && bash test_small_batch.sh
# =============================================================================

set -e

WORK_DIR="/opt/tiger/mmfinetune/examples/nas3r"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"
SMALL_DATA_DIR="${WORK_DIR}/datasets_small"

cd "${WORK_DIR}"

# ======================== 步骤1: 准备小批量数据 ========================
echo "============================================"
echo "[Step 1] 准备小批量测试数据"
echo "============================================"

# 创建小数据集目录
mkdir -p "${SMALL_DATA_DIR}/re10k/train"
mkdir -p "${SMALL_DATA_DIR}/re10k/test"

# 只复制前5个.torch文件用于测试（而非全部）
echo "复制少量torch文件..."
TRAIN_FILES=($(ls "${LOCAL_DATA_DIR}/re10k/train/"*.torch 2>/dev/null | head -5))
TEST_FILES=($(ls "${LOCAL_DATA_DIR}/re10k/test/"*.torch 2>/dev/null | head -2))

if [ ${#TRAIN_FILES[@]} -eq 0 ]; then
    echo "[ERROR] 未找到训练数据文件，请先运行 setup_and_run.sh 下载数据"
    exit 1
fi

for f in "${TRAIN_FILES[@]}"; do
    cp -f "$f" "${SMALL_DATA_DIR}/re10k/train/"
done

for f in "${TEST_FILES[@]}"; do
    cp -f "$f" "${SMALL_DATA_DIR}/re10k/test/"
done

# 复制 index.json（如果存在）
cp -f "${LOCAL_DATA_DIR}/re10k/train/index.json" "${SMALL_DATA_DIR}/re10k/train/" 2>/dev/null || true
cp -f "${LOCAL_DATA_DIR}/re10k/test/index.json" "${SMALL_DATA_DIR}/re10k/test/" 2>/dev/null || true

# 如果没有 index.json，则生成一个简单的
if [ ! -f "${SMALL_DATA_DIR}/re10k/train/index.json" ]; then
    echo "生成 train index.json..."
    python3 -c "
import json, os, torch
index = {}
data_dir = '${SMALL_DATA_DIR}/re10k/train'
for f in sorted(os.listdir(data_dir)):
    if f.endswith('.torch'):
        chunk = torch.load(os.path.join(data_dir, f), weights_only=True)
        for item in chunk:
            index[item['key']] = f
with open(os.path.join(data_dir, 'index.json'), 'w') as fp:
    json.dump(index, fp)
print(f'生成 train index.json, 共 {len(index)} 个scene')
"
fi

if [ ! -f "${SMALL_DATA_DIR}/re10k/test/index.json" ]; then
    echo "生成 test index.json..."
    python3 -c "
import json, os, torch
index = {}
data_dir = '${SMALL_DATA_DIR}/re10k/test'
for f in sorted(os.listdir(data_dir)):
    if f.endswith('.torch'):
        chunk = torch.load(os.path.join(data_dir, f), weights_only=True)
        for item in chunk:
            index[item['key']] = f
with open(os.path.join(data_dir, 'index.json'), 'w') as fp:
    json.dump(index, fp)
print(f'生成 test index.json, 共 {len(index)} 个scene')
"
fi

echo ""
echo "小批量数据:"
echo "  train: $(ls ${SMALL_DATA_DIR}/re10k/train/*.torch 2>/dev/null | wc -l) torch files"
echo "  test:  $(ls ${SMALL_DATA_DIR}/re10k/test/*.torch 2>/dev/null | wc -l) torch files"

# ======================== 步骤2: 运行小批量训练测试 ========================
echo ""
echo "============================================"
echo "[Step 2] 运行小批量训练测试 (100 steps)"
echo "============================================"

# 使用小数据集路径，减小batch_size，只跑100步
python -m src.main \
    +experiment=nas3r/random/re10k \
    wandb.mode=disabled \
    wandb.name=nas3r_test_small \
    dataset.re10k.roots=["${SMALL_DATA_DIR}/re10k"] \
    data_loader.train.batch_size=2 \
    data_loader.train.num_workers=4 \
    trainer.max_steps=100 \
    trainer.val_check_interval=50 \
    checkpointing.every_n_train_steps=50 \
    train.print_log_every_n_steps=5

echo ""
echo "============================================"
echo "小批量测试完成 ✓"
echo "============================================"
echo "如果上面没有报错，说明环境、数据、模型都正常工作。"
echo "可以用完整数据集开始正式训练了。"

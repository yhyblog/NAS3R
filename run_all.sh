#!/bin/bash
# =============================================================================
# NAS3R 一键脚本 (远程GPU平台)
# 功能: clone代码 → 安装依赖 → 下载re10k数据集 → 上传HDFS → 生成index → 小批量测试
#
# 使用方法 (在远程GPU机器上):
#   cd /opt/tiger/mmfinetune/examples/nas3r && bash run_all.sh
# =============================================================================

set -e

# ======================== 配置区域 ========================
WORK_DIR="/opt/tiger/mmfinetune/examples/nas3r"
HDFS_BASE="hdfs://harunafr/home/byte_data_tt_m/data/yaohaoyang/3d_data/nas3r"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"
SMALL_DATA_DIR="${WORK_DIR}/datasets_small"

# ======================== Step 0: Clone代码 ========================
echo "============================================"
echo "[Step 0] 拉取代码"
echo "============================================"
if [ ! -f "${WORK_DIR}/requirements.txt" ]; then
    mkdir -p /opt/tiger/mmfinetune/examples
    cd /opt/tiger/mmfinetune/examples
    git clone --recurse-submodules https://github.com/yhyblog/NAS3R.git nas3r
    echo "[Step 0] 代码拉取完成 ✓"
else
    echo "[Step 0] 代码已存在，跳过 ✓"
fi
cd "${WORK_DIR}"

# ======================== Step 1: 安装依赖 ========================
echo ""
echo "============================================"
echo "[Step 1] 安装Python依赖"
echo "============================================"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt --no-build-isolation
pip install -e submodules/diff-gaussian-rasterization --no-build-isolation
pip install -q huggingface_hub
echo "[Step 1] 依赖安装完成 ✓"

# ======================== Step 2: 下载 re10k 数据集 ========================
echo ""
echo "============================================"
echo "[Step 2] 下载 re10k 数据集 (yiren-lu/re10k_pixelsplat)"
echo "  数据结构: train/ 330个.torch + test/ 331个.torch (~575GB)"
echo "============================================"

# --- 检查 HDFS 是否可用 ---
HDFS_AVAILABLE=true
hdfs dfs -ls / >/dev/null 2>&1 || HDFS_AVAILABLE=false

# --- 判断数据是否已存在 ---
RE10K_LOCAL_READY=false
RE10K_HDFS_READY=false

# 检查本地是否已有数据
LOCAL_TRAIN_COUNT=$(ls "${LOCAL_DATA_DIR}/re10k/train/"*.torch 2>/dev/null | wc -l)
if [ "$LOCAL_TRAIN_COUNT" -gt 100 ]; then
    echo "[re10k] 本地已有 ${LOCAL_TRAIN_COUNT} 个train文件，跳过下载 ✓"
    RE10K_LOCAL_READY=true
fi

# 检查HDFS是否已有数据
if [ "$RE10K_LOCAL_READY" = false ] && [ "$HDFS_AVAILABLE" = true ]; then
    RE10K_HDFS_COUNT=$(hdfs dfs -ls "${HDFS_BASE}/re10k/train/" 2>/dev/null | grep ".torch" | wc -l)
    if [ "$RE10K_HDFS_COUNT" -gt 100 ]; then
        echo "[re10k] HDFS已有 ${RE10K_HDFS_COUNT} 个train文件"
        RE10K_HDFS_READY=true
    fi
fi

# ---- 下载逻辑 ----
if [ "$RE10K_LOCAL_READY" = false ] && [ "$RE10K_HDFS_READY" = false ]; then
    # 情况A: 哪里都没有，从HuggingFace下载
    echo "[re10k] 从 HuggingFace 下载到本地..."
    mkdir -p "${LOCAL_DATA_DIR}/re10k"

    python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download

local_dir = os.environ.get("LOCAL_DATA_DIR", "") + "/re10k"
print(f"[re10k] 下载到: {local_dir}")
snapshot_download(
    repo_id="yiren-lu/re10k_pixelsplat",
    repo_type="dataset",
    local_dir=local_dir,
    resume_download=True,
)
print("[re10k] HuggingFace 下载完成 ✓")
PYEOF

    # 上传HDFS持久化 (后台执行，不阻塞后续步骤)
    if [ "$HDFS_AVAILABLE" = true ]; then
        echo "[re10k] 后台上传HDFS持久化..."
        hdfs dfs -mkdir -p "${HDFS_BASE}/re10k/train/" "${HDFS_BASE}/re10k/test/"
        hdfs dfs -put -f "${LOCAL_DATA_DIR}/re10k/train/" "${HDFS_BASE}/re10k/" &
        hdfs dfs -put -f "${LOCAL_DATA_DIR}/re10k/test/"  "${HDFS_BASE}/re10k/" &
        HDFS_UPLOAD_PID=$!
        echo "[re10k] HDFS上传已在后台运行 (PID: $HDFS_UPLOAD_PID)"
    fi

elif [ "$RE10K_LOCAL_READY" = false ] && [ "$RE10K_HDFS_READY" = true ]; then
    # 情况B: HDFS有数据，拉到本地
    echo "[re10k] 从HDFS拉取到本地..."
    mkdir -p "${LOCAL_DATA_DIR}/re10k/train" "${LOCAL_DATA_DIR}/re10k/test"
    hdfs dfs -get "${HDFS_BASE}/re10k/train" "${LOCAL_DATA_DIR}/re10k/"
    hdfs dfs -get "${HDFS_BASE}/re10k/test"  "${LOCAL_DATA_DIR}/re10k/"
    echo "[re10k] HDFS拉取完成 ✓"
fi

# 验证下载结果
echo ""
echo "--- re10k 本地数据统计 ---"
echo "  train: $(ls ${LOCAL_DATA_DIR}/re10k/train/*.torch 2>/dev/null | wc -l) / 330 torch files"
echo "  test:  $(ls ${LOCAL_DATA_DIR}/re10k/test/*.torch  2>/dev/null | wc -l) / 331 torch files"

# ======================== Step 3: 生成 index.json ========================
echo ""
echo "============================================"
echo "[Step 3] 生成 index.json (数据集没有自带)"
echo "============================================"

export LOCAL_DATA_DIR
python3 << 'PYEOF'
import json, os, torch

data_base = os.environ["LOCAL_DATA_DIR"] + "/re10k"

for split in ["train", "test"]:
    data_dir = os.path.join(data_base, split)
    index_path = os.path.join(data_dir, "index.json")

    # 如果已存在就跳过
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
        print(f"  {split}/index.json 已存在 ({len(existing)} scenes)，跳过 ✓")
        continue

    if not os.path.isdir(data_dir):
        print(f"  {split}/ 目录不存在，跳过")
        continue

    print(f"  扫描 {split}/ 生成 index.json ...")
    index = {}
    torch_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".torch")])
    for i, fname in enumerate(torch_files):
        fpath = os.path.join(data_dir, fname)
        try:
            chunk = torch.load(fpath, weights_only=True)
            for item in chunk:
                index[item["key"]] = fname
        except Exception as e:
            print(f"    [WARN] {fname} 加载失败: {e}")
        if (i + 1) % 50 == 0:
            print(f"    已处理 {i+1}/{len(torch_files)} files...")

    with open(index_path, "w") as f:
        json.dump(index, f)
    print(f"  {split}/index.json 生成完成: {len(index)} scenes ✓")
PYEOF

echo "[Step 3] index.json 生成完成 ✓"

# ======================== Step 4: 小批量测试 ========================
echo ""
echo "============================================"
echo "[Step 4] 小批量测试 (5个chunk, batch_size=2, 100步)"
echo "============================================"

# 准备小数据集 (只取前几个chunk，避免用全量数据测试)
mkdir -p "${SMALL_DATA_DIR}/re10k/train" "${SMALL_DATA_DIR}/re10k/test"

TRAIN_FILES=($(ls "${LOCAL_DATA_DIR}/re10k/train/"*.torch 2>/dev/null | head -5))
TEST_FILES=($(ls "${LOCAL_DATA_DIR}/re10k/test/"*.torch 2>/dev/null | head -2))

if [ ${#TRAIN_FILES[@]} -eq 0 ]; then
    echo "[ERROR] 训练数据为空! 请检查数据下载。"
    echo "  预期路径: ${LOCAL_DATA_DIR}/re10k/train/*.torch"
    ls -la "${LOCAL_DATA_DIR}/re10k/" 2>/dev/null || true
    exit 1
fi

for f in "${TRAIN_FILES[@]}"; do ln -sf "$f" "${SMALL_DATA_DIR}/re10k/train/"; done
for f in "${TEST_FILES[@]}";  do ln -sf "$f" "${SMALL_DATA_DIR}/re10k/test/";  done

# 为小数据集生成 index.json
export SMALL_DATA_DIR
python3 << 'PYEOF'
import json, os, torch

data_base = os.environ["SMALL_DATA_DIR"] + "/re10k"
for split in ["train", "test"]:
    data_dir = os.path.join(data_base, split)
    if not os.path.isdir(data_dir):
        continue
    index = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".torch"):
            real_path = os.path.realpath(os.path.join(data_dir, fname))
            chunk = torch.load(real_path, weights_only=True)
            for item in chunk:
                index[item["key"]] = fname
    with open(os.path.join(data_dir, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"  小数据集 {split}/index.json: {len(index)} scenes")
PYEOF

echo ""
echo "小数据集: train=$(ls ${SMALL_DATA_DIR}/re10k/train/*.torch | wc -l) chunks, test=$(ls ${SMALL_DATA_DIR}/re10k/test/*.torch 2>/dev/null | wc -l) chunks"
echo ""

# GPU信息
echo "--- GPU信息 ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU信息不可用"
echo ""

# 跑小批量
echo "开始训练 (batch_size=2, max_steps=100)..."
python -m src.main \
    +experiment=nas3r/random/re10k \
    wandb.mode=disabled \
    wandb.name=nas3r_test_small \
    "dataset.re10k.roots=[${SMALL_DATA_DIR}/re10k]" \
    data_loader.train.batch_size=2 \
    data_loader.train.num_workers=4 \
    trainer.max_steps=100 \
    trainer.val_check_interval=50 \
    checkpointing.every_n_train_steps=50 \
    train.print_log_every_n_steps=5

echo ""
echo "============================================"
echo " 全部完成！小批量测试通过。"
echo "============================================"
echo ""
echo "下一步 - 正式训练:"
echo "  cd ${WORK_DIR}"
echo "  python -m src.main +experiment=nas3r/random/re10k \\"
echo "      wandb.mode=disabled \\"
echo "      \"dataset.re10k.roots=[${LOCAL_DATA_DIR}/re10k]\" \\"
echo "      data_loader.train.batch_size=10"

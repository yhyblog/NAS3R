#!/bin/bash
# =============================================================================
# NAS3R 一键脚本 (远程GPU平台)
# 功能: clone代码 → 安装依赖 → 下载re10k数据集 → 上传HDFS → 生成index → 小批量测试
#
# 使用方法:
#   cd /opt/tiger/mmfinetune/examples/nas3r && bash run_all.sh
# =============================================================================

set -e

# ======================== 配置区域 ========================
WORK_DIR="/opt/tiger/mmfinetune/examples/nas3r"
HDFS_BASE="hdfs://harunafr/home/byte_data_tt_m/data/yaohaoyang/3d_data/nas3r"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"
SMALL_DATA_DIR="${WORK_DIR}/datasets_small"

# ======================== Step 0: Clone 或恢复代码 ========================
echo "============================================"
echo "[Step 0] 拉取/恢复代码"
echo "============================================"

mkdir -p /opt/tiger/mmfinetune/examples
cd /opt/tiger/mmfinetune/examples

if [ ! -d "${WORK_DIR}/.git" ]; then
    # 全新 clone
    echo "[Step 0] 目录不存在，开始 clone..."
    GIT_SSL_NO_VERIFY=true git clone --recurse-submodules https://github.com/yhyblog/NAS3R.git nas3r || \
    git clone --recurse-submodules git://github.com/yhyblog/NAS3R.git nas3r || \
    git clone --recurse-submodules https://ghproxy.com/https://github.com/yhyblog/NAS3R.git nas3r || \
    { echo "[ERROR] 所有 clone 方式均失败，请检查网络"; exit 1; }
    echo "[Step 0] 代码拉取完成 ✓"
else
    echo "[Step 0] .git 目录已存在，恢复可能被删除的文件..."
    cd "${WORK_DIR}"
    # 恢复所有被删除/修改的tracked文件（不影响 untracked 文件如 datasets/）
    git checkout -- . || true
    echo "[Step 0] 文件恢复完成 ✓"
fi

cd "${WORK_DIR}"

# ======================== Step 1: 安装依赖 ========================
echo ""
echo "============================================"
echo "[Step 1] 安装Python依赖"
echo "============================================"

# 打印当前镜像环境信息
echo "[Step 1] 当前环境:"
echo "  Python: $(python3 --version 2>&1)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  CUDA(torch): $(python3 -c 'import torch; print(torch.version.cuda)' 2>&1)"
echo "  nvcc: $(nvcc --version 2>&1 | grep release || echo '不可用')"
echo ""

python3 -m pip install --upgrade pip

# *** 不安装 torch/torchvision — 直接使用镜像自带的 byted-torch 2.7.1 + cu126 ***
# 原始项目要求 torch==2.5.1，但 2.7.1 API 兼容，且重装可能破坏 byted-torch

# 过滤掉 requirements.txt 中不兼容/已有的包，以及 pytorch3d（单独处理）
grep -v -E "scikit-video|^torch==|^torchvision==|^torchaudio==|pytorch3d" requirements.txt > requirements_fixed.txt
pip install -r requirements_fixed.txt --no-build-isolation

# scikit-video 原版不兼容 Python 3.11+，安装社区维护的 fork
pip install sk-video || pip install scikit-video || \
    { echo "[WARN] scikit-video 安装失败，尝试 patch..."; \
      pip install scikit-video --no-deps 2>/dev/null || true; }

# 单独安装 pytorch3d（从源码编译，需要匹配当前 torch+CUDA）
echo "[Step 1] 安装 pytorch3d ..."
if python3 -c "import pytorch3d" 2>/dev/null; then
    echo "[Step 1] pytorch3d 已安装 ✓"
else
    # 自动检测当前 GPU 的 compute capability，只编译对应架构（大幅加速）
    GPU_ARCH=$(python3 -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'{cap[0]}.{cap[1]}')
else:
    print('8.0')
" 2>/dev/null)
    echo "[Step 1] 检测到 GPU 架构: sm_${GPU_ARCH//./_}"
    export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
    export FORCE_CUDA=1
    export MAX_JOBS=4  # 限制并行编译数，避免 OOM 卡死

    echo "[Step 1] 开始编译 pytorch3d（仅编译 sm_${GPU_ARCH//./_}，MAX_JOBS=4）..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation || \
        { echo "[WARN] pytorch3d 源码编译失败，部分功能可能不可用（不影响核心训练）"; }
fi

# ---------- 安装 diff-gaussian-rasterization 子模块 ----------
echo "[Step 1] 检查 diff-gaussian-rasterization 子模块..."

# 1) 确保子模块 URL 正确并初始化
git submodule sync --recursive || true
git submodule update --init --recursive || true

# 2) 如果 git submodule 失败，手动 clone
if [ ! -f "./submodules/diff-gaussian-rasterization/setup.py" ]; then
    echo "[Step 1] 子模块不完整，手动 clone..."
    rm -rf ./submodules/diff-gaussian-rasterization
    mkdir -p ./submodules
    git clone --recurse-submodules -b camera \
        https://github.com/ranrhuang/diff-gaussian-rasterization.git \
        ./submodules/diff-gaussian-rasterization || true
fi

# 3) 确保 third_party/glm 存在（上次报错的根因）
GLM_DIR="./submodules/diff-gaussian-rasterization/third_party/glm"
if [ ! -f "${GLM_DIR}/glm/glm.hpp" ]; then
    echo "[Step 1] GLM 库缺失，手动下载..."
    rm -rf "${GLM_DIR}"
    git clone https://github.com/g-truc/glm.git "${GLM_DIR}" || \
    {
        echo "[Step 1] git clone GLM 失败，尝试下载 release 包..."
        mkdir -p "${GLM_DIR}"
        curl -sL https://github.com/g-truc/glm/releases/download/1.0.1/glm-1.0.1-light.zip -o /tmp/glm.zip && \
        unzip -qo /tmp/glm.zip -d "${GLM_DIR}/.." && rm -f /tmp/glm.zip || \
        { echo "[ERROR] 无法获取 GLM 库"; exit 1; }
    }
    echo "[Step 1] GLM 库就绪 ✓"
fi

# 4) 编译安装
if [ -f "./submodules/diff-gaussian-rasterization/setup.py" ]; then
    echo "[Step 1] 编译安装 diff-gaussian-rasterization ..."
    cd ./submodules/diff-gaussian-rasterization
    # 清理旧的编译缓存
    rm -rf build/ dist/ *.egg-info
    cd "${WORK_DIR}"
    pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation || \
        { echo "[WARN] diff-gaussian-rasterization 安装失败，先跳过"; }
else
    echo "[WARN] diff-gaussian-rasterization setup.py 不存在，跳过编译"
fi

pip install -q huggingface_hub
echo "[Step 1] 依赖安装完成 ✓"

# ======================== Step 2: 下载 re10k 数据集 ========================
echo ""
echo "============================================"
echo "[Step 2] 下载 re10k 数据集 (yiren-lu/re10k_pixelsplat)"
echo "  数据结构: train/ 330个.torch + test/ 331个.torch (~575GB)"
echo "============================================"

mkdir -p "${LOCAL_DATA_DIR}" "${SMALL_DATA_DIR}"

# 磁盘空间检查
echo "[Step 2] 当前磁盘空间:"
df -h "${WORK_DIR}" || true
echo ""

# --- 检查 HDFS 是否可用 ---
HDFS_AVAILABLE=true
hdfs dfs -ls / >/dev/null 2>&1 || HDFS_AVAILABLE=false

# --- 判断数据是否已存在 ---
RE10K_LOCAL_READY=false
RE10K_HDFS_READY=false

LOCAL_TRAIN_COUNT=$(find "${LOCAL_DATA_DIR}/re10k/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
if [ "$LOCAL_TRAIN_COUNT" -gt 100 ]; then
    echo "[re10k] 本地已有 ${LOCAL_TRAIN_COUNT} 个train文件，跳过下载 ✓"
    RE10K_LOCAL_READY=true
fi

if [ "$RE10K_LOCAL_READY" = false ] && [ "$HDFS_AVAILABLE" = true ]; then
    RE10K_HDFS_COUNT=$(hdfs dfs -ls "${HDFS_BASE}/re10k/train/" 2>/dev/null | grep ".torch" | wc -l)
    if [ "$RE10K_HDFS_COUNT" -gt 100 ]; then
        echo "[re10k] HDFS已有 ${RE10K_HDFS_COUNT} 个train文件"
        RE10K_HDFS_READY=true
    fi
fi

# ---- 下载逻辑 ----
if [ "$RE10K_LOCAL_READY" = false ] && [ "$RE10K_HDFS_READY" = false ]; then
    echo "[re10k] 从 HuggingFace 下载到本地..."
    mkdir -p "${LOCAL_DATA_DIR}/re10k"
    export LOCAL_DATA_DIR

    python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download

base = os.environ["LOCAL_DATA_DIR"]
local_dir = os.path.join(base, "re10k")
os.makedirs(local_dir, exist_ok=True)

print(f"[re10k] 下载到: {local_dir}")
snapshot_download(
    repo_id="yiren-lu/re10k_pixelsplat",
    repo_type="dataset",
    local_dir=local_dir,
)
print("[re10k] HuggingFace 下载完成 ✓")
PYEOF

    if [ "$HDFS_AVAILABLE" = true ]; then
        echo "[re10k] 后台上传HDFS持久化..."
        hdfs dfs -mkdir -p "${HDFS_BASE}/re10k/train/" "${HDFS_BASE}/re10k/test/"
        hdfs dfs -put -f "${LOCAL_DATA_DIR}/re10k/train/" "${HDFS_BASE}/re10k/" &
        PID1=$!
        hdfs dfs -put -f "${LOCAL_DATA_DIR}/re10k/test/" "${HDFS_BASE}/re10k/" &
        PID2=$!
        echo "[re10k] HDFS上传已在后台运行 (PID train=${PID1}, test=${PID2})"
    fi

elif [ "$RE10K_LOCAL_READY" = false ] && [ "$RE10K_HDFS_READY" = true ]; then
    echo "[re10k] 从HDFS拉取到本地..."
    mkdir -p "${LOCAL_DATA_DIR}/re10k/train" "${LOCAL_DATA_DIR}/re10k/test"
    hdfs dfs -get "${HDFS_BASE}/re10k/train" "${LOCAL_DATA_DIR}/re10k/"
    hdfs dfs -get "${HDFS_BASE}/re10k/test"  "${LOCAL_DATA_DIR}/re10k/"
    echo "[re10k] HDFS拉取完成 ✓"
fi

echo ""
echo "--- re10k 本地数据统计 ---"
echo "  train: $(find ${LOCAL_DATA_DIR}/re10k/train -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) / 330 torch files"
echo "  test:  $(find ${LOCAL_DATA_DIR}/re10k/test  -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) / 331 torch files"

# ======================== Step 3: 生成 index.json ========================
echo ""
echo "============================================"
echo "[Step 3] 生成 index.json (数据集没有自带)"
echo "============================================"

export LOCAL_DATA_DIR
python3 << 'PYEOF'
import json, os, torch

data_base = os.path.join(os.environ["LOCAL_DATA_DIR"], "re10k")

for split in ["train", "test"]:
    data_dir = os.path.join(data_base, split)
    index_path = os.path.join(data_dir, "index.json")

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

mkdir -p "${SMALL_DATA_DIR}/re10k/train" "${SMALL_DATA_DIR}/re10k/test"

TRAIN_FILES=($(find "${LOCAL_DATA_DIR}/re10k/train" -maxdepth 1 -name "*.torch" 2>/dev/null | sort | head -5))
TEST_FILES=($(find "${LOCAL_DATA_DIR}/re10k/test"  -maxdepth 1 -name "*.torch" 2>/dev/null | sort | head -2))

if [ ${#TRAIN_FILES[@]} -eq 0 ]; then
    echo "[ERROR] 训练数据为空! 请检查数据下载。"
    echo "  预期路径: ${LOCAL_DATA_DIR}/re10k/train/*.torch"
    ls -la "${LOCAL_DATA_DIR}/re10k/" 2>/dev/null || true
    exit 1
fi

for f in "${TRAIN_FILES[@]}"; do ln -sf "$f" "${SMALL_DATA_DIR}/re10k/train/"; done
for f in "${TEST_FILES[@]}";  do ln -sf "$f" "${SMALL_DATA_DIR}/re10k/test/";  done

export SMALL_DATA_DIR
python3 << 'PYEOF'
import json, os, torch

data_base = os.path.join(os.environ["SMALL_DATA_DIR"], "re10k")
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
echo "小数据集: train=$(find ${SMALL_DATA_DIR}/re10k/train -maxdepth 1 -name '*.torch' | wc -l) chunks, test=$(find ${SMALL_DATA_DIR}/re10k/test -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) chunks"
echo ""

echo "--- GPU信息 ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU信息不可用"
echo ""

echo "开始小批量测试 (单卡, batch_size=2, max_steps=100, 跳过validation)..."
CUDA_VISIBLE_DEVICES=0 python -m src.main \
    +experiment=nas3r/random/re10k \
    wandb.mode=disabled \
    wandb.name=nas3r_test_small \
    "dataset.re10k.roots=[${SMALL_DATA_DIR}/re10k]" \
    data_loader.train.batch_size=2 \
    data_loader.train.num_workers=4 \
    trainer.max_steps=100 \
    trainer.val_check_interval=999999 \
    +trainer.num_sanity_val_steps=0 \
    checkpointing.every_n_train_steps=50 \
    train.print_log_every_n_steps=5

echo ""
echo "============================================"
echo " 全部完成！小批量测试通过。"
echo "============================================"
echo ""
echo "下一步 - 正式多卡训练 (需要 torchrun):"
echo "  cd ${WORK_DIR}"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "  torchrun --nproc_per_node=${NUM_GPUS} -m src.main +experiment=nas3r/random/re10k \\"
echo "      wandb.mode=disabled \\"
echo "      \"dataset.re10k.roots=[${LOCAL_DATA_DIR}/re10k]\" \\"
echo "      data_loader.train.batch_size=10"
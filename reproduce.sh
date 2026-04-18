#!/bin/bash
# =============================================================================
# NAS3R 一键复现脚本（远程 GPU 平台，镜像: py311.cu126 + byted-torch 2.7.1）
#
# 使用:
#   cd /opt/tiger/mmfinetune/examples && bash reproduce.sh
#
# 包含:
#   Step 0: 拉代码/恢复源码
#   Step 1: 安装依赖（跳过镜像自带的 torch）
#   Step 2: 编译 diff-gaussian-rasterization（含 GLM 修复）
#   Step 3: 编译 pytorch3d（只编当前 GPU 架构）
#   Step 4: 下载 re10k 数据集（~575GB）
#   Step 5: 生成 index.json
#   Step 6: Patch model_wrapper.py（wandb disabled 兼容）
#   Step 7: 启动 8 卡训练（bf16, batch=2, nohup 后台运行）
# =============================================================================

set -e

# ======================== 配置 ========================
WORK_DIR="/opt/tiger/mmfinetune/examples/nas3r"
PARENT_DIR="/opt/tiger/mmfinetune/examples"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"
HDFS_BASE="hdfs://harunafr/home/byte_data_tt_m/data/yaohaoyang/3d_data/nas3r"

# ==================== Step 0: 拉代码/恢复源码 ====================
echo "============================================"
echo "[Step 0] 准备代码"
echo "============================================"
mkdir -p "${PARENT_DIR}"
cd "${PARENT_DIR}"

if [ ! -d "${WORK_DIR}/.git" ]; then
    echo "[Step 0] 首次 clone..."
    GIT_SSL_NO_VERIFY=true git clone --recurse-submodules https://github.com/yhyblog/NAS3R.git nas3r || \
    git clone --recurse-submodules https://ghproxy.com/https://github.com/yhyblog/NAS3R.git nas3r || \
    { echo "[ERROR] clone 失败"; exit 1; }
else
    echo "[Step 0] 恢复被删文件..."
    cd "${WORK_DIR}"
    git checkout -- . || true
fi
cd "${WORK_DIR}"
echo "[Step 0] ✓"

# ==================== Step 1: 安装依赖 ====================
echo ""
echo "============================================"
echo "[Step 1] 安装依赖"
echo "============================================"
echo "[环境] Python=$(python3 --version 2>&1) | PyTorch=$(python3 -c 'import torch;print(torch.__version__)') | CUDA=$(python3 -c 'import torch;print(torch.version.cuda)')"

python3 -m pip install --upgrade pip -q

# 关键：不安装 torch/torchvision（用镜像自带的 byted-torch）
grep -v -E "scikit-video|^torch==|^torchvision==|^torchaudio==|pytorch3d" requirements.txt > requirements_fixed.txt
pip install -r requirements_fixed.txt --no-build-isolation

# scikit-video 的 py3.11 兼容版
pip install -q sk-video huggingface_hub

echo "[Step 1] ✓"

# ==================== Step 2: 编译 diff-gaussian-rasterization ====================
echo ""
echo "============================================"
echo "[Step 2] 编译 diff-gaussian-rasterization"
echo "============================================"

git submodule sync --recursive >/dev/null 2>&1 || true
git submodule update --init --recursive || true

# 如果子模块没拉下来，手动 clone
if [ ! -f "./submodules/diff-gaussian-rasterization/setup.py" ]; then
    rm -rf ./submodules/diff-gaussian-rasterization
    mkdir -p ./submodules
    git clone --recurse-submodules -b camera \
        https://github.com/ranrhuang/diff-gaussian-rasterization.git \
        ./submodules/diff-gaussian-rasterization
fi

# 修复 GLM 库缺失（上次踩坑点）
GLM_DIR="./submodules/diff-gaussian-rasterization/third_party/glm"
if [ ! -f "${GLM_DIR}/glm/glm.hpp" ]; then
    echo "[Step 2] 下载 GLM 库..."
    rm -rf "${GLM_DIR}"
    git clone https://github.com/g-truc/glm.git "${GLM_DIR}"
fi

# 清理旧编译缓存
rm -rf ./submodules/diff-gaussian-rasterization/build/ \
       ./submodules/diff-gaussian-rasterization/*.egg-info

pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation

echo "[Step 2] ✓"

# ==================== Step 3: 编译 pytorch3d ====================
echo ""
echo "============================================"
echo "[Step 3] 安装 pytorch3d"
echo "============================================"

if python3 -c "import pytorch3d" 2>/dev/null; then
    echo "[Step 3] pytorch3d 已安装 ✓"
else
    # 只编译当前 GPU 架构，加速编译
    GPU_ARCH=$(python3 -c "import torch;cap=torch.cuda.get_device_capability(0);print(f'{cap[0]}.{cap[1]}')" 2>/dev/null || echo "8.0")
    echo "[Step 3] 目标架构: sm_${GPU_ARCH//./_}（编译约 10-20 分钟）"

    export TORCH_CUDA_ARCH_LIST="${GPU_ARCH}"
    export FORCE_CUDA=1
    export MAX_JOBS=4

    pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
fi
echo "[Step 3] ✓"

# ==================== Step 4: 下载 re10k ====================
echo ""
echo "============================================"
echo "[Step 4] 准备 re10k 数据集（~575GB）"
echo "============================================"

mkdir -p "${LOCAL_DATA_DIR}/re10k"

LOCAL_TRAIN_COUNT=$(find "${LOCAL_DATA_DIR}/re10k/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
LOCAL_TEST_COUNT=$(find "${LOCAL_DATA_DIR}/re10k/test" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)

if [ "$LOCAL_TRAIN_COUNT" -gt 300 ] && [ "$LOCAL_TEST_COUNT" -gt 300 ]; then
    echo "[Step 4] 本地已有完整数据 (train=${LOCAL_TRAIN_COUNT}, test=${LOCAL_TEST_COUNT})，跳过 ✓"
else
    # 优先从 HDFS 拉（如果已上传过）
    if hdfs dfs -ls "${HDFS_BASE}/re10k/train/" 2>/dev/null | grep -q ".torch"; then
        HDFS_COUNT=$(hdfs dfs -ls "${HDFS_BASE}/re10k/train/" 2>/dev/null | grep ".torch" | wc -l)
        if [ "$HDFS_COUNT" -gt 300 ]; then
            echo "[Step 4] 从 HDFS 拉取到本地（比 HuggingFace 快）..."
            hdfs dfs -get "${HDFS_BASE}/re10k/train" "${LOCAL_DATA_DIR}/re10k/"
            hdfs dfs -get "${HDFS_BASE}/re10k/test"  "${LOCAL_DATA_DIR}/re10k/"
        fi
    fi

    # HDFS 没有就从 HuggingFace 下
    LOCAL_TRAIN_COUNT=$(find "${LOCAL_DATA_DIR}/re10k/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    if [ "$LOCAL_TRAIN_COUNT" -lt 300 ]; then
        echo "[Step 4] 从 HuggingFace 下载（约 1-3 小时）..."
        export LOCAL_DATA_DIR
        python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download
local_dir = os.path.join(os.environ["LOCAL_DATA_DIR"], "re10k")
os.makedirs(local_dir, exist_ok=True)
snapshot_download(repo_id="yiren-lu/re10k_pixelsplat", repo_type="dataset", local_dir=local_dir)
print("[re10k] HuggingFace 下载完成")
PYEOF

        # 下完后台上传 HDFS 持久化（下次复现就不用再下）
        if hdfs dfs -ls / >/dev/null 2>&1; then
            echo "[Step 4] 后台上传 HDFS..."
            hdfs dfs -mkdir -p "${HDFS_BASE}/re10k/" 2>/dev/null
            nohup hdfs dfs -put -f "${LOCAL_DATA_DIR}/re10k/train" "${HDFS_BASE}/re10k/" > /tmp/hdfs_train.log 2>&1 &
            nohup hdfs dfs -put -f "${LOCAL_DATA_DIR}/re10k/test"  "${HDFS_BASE}/re10k/" > /tmp/hdfs_test.log 2>&1 &
        fi
    fi
fi

echo "[Step 4] train: $(find ${LOCAL_DATA_DIR}/re10k/train -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) / 330"
echo "[Step 4] test:  $(find ${LOCAL_DATA_DIR}/re10k/test  -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) / 331"
echo "[Step 4] ✓"

# ==================== Step 5: 生成 index.json ====================
echo ""
echo "============================================"
echo "[Step 5] 生成 index.json"
echo "============================================"

export LOCAL_DATA_DIR
python3 << 'PYEOF'
import json, os, torch
base = os.path.join(os.environ["LOCAL_DATA_DIR"], "re10k")
for split in ["train", "test"]:
    d = os.path.join(base, split)
    idx_path = os.path.join(d, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            print(f"  {split}/index.json 已存在 ({len(json.load(f))} scenes) ✓")
        continue
    if not os.path.isdir(d): continue
    print(f"  扫描 {split}/ ...")
    index = {}
    files = sorted(f for f in os.listdir(d) if f.endswith(".torch"))
    for i, fn in enumerate(files):
        try:
            for item in torch.load(os.path.join(d, fn), weights_only=True):
                index[item["key"]] = fn
        except Exception as e:
            print(f"    [WARN] {fn}: {e}")
        if (i+1) % 50 == 0: print(f"    {i+1}/{len(files)}")
    with open(idx_path, "w") as f:
        json.dump(index, f)
    print(f"  {split}/index.json: {len(index)} scenes ✓")
PYEOF
echo "[Step 5] ✓"

# ==================== Step 6: Patch model_wrapper.py ====================
echo ""
echo "============================================"
echo "[Step 6] Patch model_wrapper.py (wandb disabled 兼容)"
echo "============================================"

python3 << 'PYEOF'
path = "src/model/model_wrapper.py"
with open(path) as f:
    content = f.read()

old = '''        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)'''

new = '''        # Skip video logging if wandb is disabled
        if wandb.run is None:
            return

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=getattr(value, "_fps", 30))'''

if "wandb.run is None" in content:
    print("[Step 6] 已 patch ✓")
elif old in content:
    with open(path, "w") as f:
        f.write(content.replace(old, new))
    print("[Step 6] patch 成功 ✓")
else:
    print("[Step 6] [WARN] 模式未匹配，可能源码版本变了，跳过 patch")
PYEOF
echo "[Step 6] ✓"

# ==================== Step 7: 启动 8 卡训练 ====================
echo ""
echo "============================================"
echo "[Step 7] 启动 8 卡训练"
echo "============================================"

# NCCL 配置（调通过的）
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
unset NCCL_NET_PLUGIN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 先杀掉残留进程，释放显存
pkill -9 -f "src.main" 2>/dev/null || true
sleep 3

TRAIN_LOG="${WORK_DIR}/train_full.log"
echo "[Step 7] 日志: ${TRAIN_LOG}"
echo "[Step 7] 配置: 8卡 × batch=2 × bf16 × 400k 步（约 3 天）"
echo ""
echo "[启动命令]"
cat << EOF
nohup python -m src.main \\
    +experiment=nas3r/random/re10k \\
    wandb.mode=disabled \\
    wandb.name=nas3r_re10k_8gpu \\
    "dataset.re10k.roots=[${LOCAL_DATA_DIR}/re10k]" \\
    data_loader.train.batch_size=2 \\
    data_loader.train.num_workers=8 \\
    trainer.val_check_interval=999999 \\
    +trainer.num_sanity_val_steps=0 \\
    +trainer.precision=bf16-mixed \\
    checkpointing.every_n_train_steps=5000 \\
    > ${TRAIN_LOG} 2>&1 &
EOF
echo ""

nohup python -m src.main \
    +experiment=nas3r/random/re10k \
    wandb.mode=disabled \
    wandb.name=nas3r_re10k_8gpu \
    "dataset.re10k.roots=[${LOCAL_DATA_DIR}/re10k]" \
    data_loader.train.batch_size=2 \
    data_loader.train.num_workers=8 \
    trainer.val_check_interval=999999 \
    +trainer.num_sanity_val_steps=0 \
    +trainer.precision=bf16-mixed \
    checkpointing.every_n_train_steps=5000 \
    > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "[Step 7] 训练已启动，PID: ${TRAIN_PID}"
echo ""
echo "============================================"
echo " 全部完成 ✓"
echo "============================================"
echo ""
echo "监控命令:"
echo "  tail -f ${TRAIN_LOG}"
echo "  watch -n 1 nvidia-smi"
echo "  grep 'train step' ${TRAIN_LOG} | tail -10"
echo ""
echo "停止训练:"
echo "  pkill -9 -f 'src.main'"

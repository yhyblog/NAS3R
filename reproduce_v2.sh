#!/bin/bash
# =============================================================================
# NAS3R 一键复现 v2（应用丁老师 TORCH_CUDA_ARCH_LIST 优化）
#
# 环境：
#   镜像: py311.cu126 + byted-torch 2.7.1
#   GPU:  8 × H100 80GB
#
# 关键改进（相比 v1）:
#   1. 所有 CUDA 扩展指定 TORCH_CUDA_ARCH_LIST="9.0"（丁老师建议，省显存）
#   2. 启动训练时 batch_size=4（可省出显存后翻倍）
#   3. 数据和 checkpoint 优先从 HDFS 拉取（避免重下 575G）
#   4. 训练日志用时间戳命名（不覆盖上次的）
#
# 使用:
#   cd /opt/tiger/mmfinetune/examples && bash reproduce_v2.sh
# =============================================================================

set -e

# ======================== 配置 ========================
WORK_DIR="/opt/tiger/mmfinetune/examples/nas3r"
PARENT_DIR="/opt/tiger/mmfinetune/examples"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"
HDFS_BASE="hdfs://harunafr/home/byte_data_tt_m/data/yaohaoyang/3d_data/nas3r"

# 关键：指定 H100 架构，避免 fat binary 浪费显存
export TORCH_CUDA_ARCH_LIST="9.0"
export FORCE_CUDA=1

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

# ==================== Step 1: 安装 Python 依赖 ====================
echo ""
echo "============================================"
echo "[Step 1] 安装 Python 依赖"
echo "============================================"
echo "[环境] Python=$(python3 --version 2>&1) | PyTorch=$(python3 -c 'import torch;print(torch.__version__)') | CUDA=$(python3 -c 'import torch;print(torch.version.cuda)')"

python3 -m pip install --upgrade pip -q

# 跳过镜像自带的 torch，跳过 pytorch3d/scikit-video（单独装）
grep -v -E "scikit-video|^torch==|^torchvision==|^torchaudio==|pytorch3d" requirements.txt > requirements_fixed.txt
pip install -r requirements_fixed.txt --no-build-isolation

# scikit-video 的 py3.11 兼容版
pip install -q sk-video huggingface_hub

echo "[Step 1] ✓"

# ==================== Step 2: 编译 diff-gaussian-rasterization (TORCH_CUDA_ARCH_LIST=9.0) ====================
echo ""
echo "============================================"
echo "[Step 2] 编译 diff-gaussian-rasterization (丁老师优化: 只编 sm_90)"
echo "============================================"

git submodule sync --recursive >/dev/null 2>&1 || true
git submodule update --init --recursive || true

if [ ! -f "./submodules/diff-gaussian-rasterization/setup.py" ]; then
    rm -rf ./submodules/diff-gaussian-rasterization
    mkdir -p ./submodules
    git clone --recurse-submodules -b camera \
        https://github.com/ranrhuang/diff-gaussian-rasterization.git \
        ./submodules/diff-gaussian-rasterization
fi

# GLM 库
GLM_DIR="./submodules/diff-gaussian-rasterization/third_party/glm"
if [ ! -f "${GLM_DIR}/glm/glm.hpp" ]; then
    echo "[Step 2] 下载 GLM 库..."
    rm -rf "${GLM_DIR}"
    git clone https://github.com/g-truc/glm.git "${GLM_DIR}"
fi

# 清理旧编译缓存（关键：必须清，否则 TORCH_CUDA_ARCH_LIST 设置不生效）
rm -rf ./submodules/diff-gaussian-rasterization/build/ \
       ./submodules/diff-gaussian-rasterization/*.egg-info

echo "[Step 2] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation

echo "[Step 2] ✓"

# ==================== Step 3: 编译 pytorch3d (TORCH_CUDA_ARCH_LIST=9.0) ====================
echo ""
echo "============================================"
echo "[Step 3] 安装 pytorch3d (只编 sm_90)"
echo "============================================"

if python3 -c "import pytorch3d" 2>/dev/null; then
    echo "[Step 3] pytorch3d 已安装 ✓"
else
    export MAX_JOBS=4
    echo "[Step 3] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}（约 15 分钟）"
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
fi
echo "[Step 3] ✓"

# ==================== Step 4: 准备 re10k 数据 ====================
echo ""
echo "============================================"
echo "[Step 4] 准备 re10k 数据（优先 HDFS，回退 HuggingFace）"
echo "============================================"

mkdir -p "${LOCAL_DATA_DIR}/re10k"

LOCAL_TRAIN_COUNT=$(find "${LOCAL_DATA_DIR}/re10k/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
LOCAL_TEST_COUNT=$(find "${LOCAL_DATA_DIR}/re10k/test" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)

# 只要 train > 4000 就认为是完整的（我们上次确认是 4866 个）
if [ "$LOCAL_TRAIN_COUNT" -gt 4000 ] && [ "$LOCAL_TEST_COUNT" -gt 400 ]; then
    echo "[Step 4] 本地已有数据 (train=${LOCAL_TRAIN_COUNT}, test=${LOCAL_TEST_COUNT})，跳过 ✓"
else
    # 优先从 HDFS 拉
    HDFS_TRAIN_COUNT=$(hdfs dfs -ls "${HDFS_BASE}/re10k/train/" 2>/dev/null | grep -c ".torch" || echo 0)
    if [ "$HDFS_TRAIN_COUNT" -gt 4000 ]; then
        echo "[Step 4] 从 HDFS 拉取（${HDFS_TRAIN_COUNT} 个文件，约 30-60 分钟）..."
        hdfs dfs -get "${HDFS_BASE}/re10k/train" "${LOCAL_DATA_DIR}/re10k/"
        hdfs dfs -get "${HDFS_BASE}/re10k/test"  "${LOCAL_DATA_DIR}/re10k/"
        echo "[Step 4] HDFS 拉取完成 ✓"
    else
        # HDFS 没数据，从 HuggingFace 下
        echo "[Step 4] HDFS 无数据，从 HuggingFace 下载（约 1 小时，~575GB）..."
        export LOCAL_DATA_DIR
        python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download
local_dir = os.path.join(os.environ["LOCAL_DATA_DIR"], "re10k")
os.makedirs(local_dir, exist_ok=True)
snapshot_download(repo_id="yiren-lu/re10k_pixelsplat", repo_type="dataset", local_dir=local_dir)
print("[re10k] HuggingFace 下载完成")
PYEOF

        # 下完后台上传 HDFS
        if hdfs dfs -ls / >/dev/null 2>&1; then
            echo "[Step 4] 后台上传 HDFS（不阻塞训练）..."
            hdfs dfs -mkdir -p "${HDFS_BASE}/re10k/" 2>/dev/null || true
            # 先清掉 HDFS 上的残留
            hdfs dfs -rm -r -f "${HDFS_BASE}/re10k/train" 2>/dev/null || true
            hdfs dfs -rm -r -f "${HDFS_BASE}/re10k/test"  2>/dev/null || true
            nohup hdfs dfs -put "${LOCAL_DATA_DIR}/re10k/train" "${HDFS_BASE}/re10k/" > /tmp/hdfs_train.log 2>&1 &
            nohup hdfs dfs -put "${LOCAL_DATA_DIR}/re10k/test"  "${HDFS_BASE}/re10k/" > /tmp/hdfs_test.log 2>&1 &
            echo "[Step 4] HDFS 后台上传已启动"
        fi
    fi
fi

echo "[Step 4] train: $(find ${LOCAL_DATA_DIR}/re10k/train -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) files"
echo "[Step 4] test:  $(find ${LOCAL_DATA_DIR}/re10k/test  -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) files"
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
    print(f"  {split}: {len(files)} files")
    for i, fn in enumerate(files):
        try:
            for item in torch.load(os.path.join(d, fn), weights_only=True):
                index[item["key"]] = fn
        except Exception as e:
            print(f"    [WARN] {fn}: {e}")
        if (i+1) % 500 == 0: print(f"    {i+1}/{len(files)}")
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

# ==================== Step 7: 尝试从 HDFS 拉之前的 checkpoint ====================
echo ""
echo "============================================"
echo "[Step 7] 检查是否有 HDFS checkpoint 可 resume"
echo "============================================"

CKPT_HDFS_DIR="${HDFS_BASE}/checkpoints"
LOCAL_CKPT_DIR="${WORK_DIR}/outputs/resume"
mkdir -p "${LOCAL_CKPT_DIR}"

if hdfs dfs -test -e "${CKPT_HDFS_DIR}/last.ckpt" 2>/dev/null; then
    echo "[Step 7] 发现 HDFS checkpoint，下载..."
    hdfs dfs -get -f "${CKPT_HDFS_DIR}/last.ckpt" "${LOCAL_CKPT_DIR}/last.ckpt"
    RESUME_CKPT="${LOCAL_CKPT_DIR}/last.ckpt"
    echo "[Step 7] ✓ 将从 ${RESUME_CKPT} resume"
else
    RESUME_CKPT=""
    echo "[Step 7] 无 HDFS checkpoint，从头开始训练"
fi

# ==================== Step 8: 启动 8 卡训练 ====================
echo ""
echo "============================================"
echo "[Step 8] 启动 8 卡训练"
echo "============================================"

# NCCL 配置（调通过的）
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
unset NCCL_NET_PLUGIN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 杀残留进程
pkill -9 -f "src.main" 2>/dev/null || true
sleep 3

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="${WORK_DIR}/train_${TIMESTAMP}.log"

# 构建训练命令
CKPT_ARG=""
if [ -n "${RESUME_CKPT}" ]; then
    CKPT_ARG="checkpointing.load=${RESUME_CKPT} checkpointing.resume=true"
fi

echo "[Step 8] 日志: ${TRAIN_LOG}"
echo "[Step 8] 配置: 8卡 × batch=4 × bf16"
echo "[Step 8] resume: ${RESUME_CKPT:-无}"
echo ""

nohup python -m src.main \
    +experiment=nas3r/random/re10k \
    wandb.mode=disabled \
    wandb.name=nas3r_re10k_8gpu \
    "dataset.re10k.roots=[${LOCAL_DATA_DIR}/re10k]" \
    data_loader.train.batch_size=4 \
    data_loader.train.num_workers=8 \
    trainer.val_check_interval=999999 \
    +trainer.num_sanity_val_steps=0 \
    +trainer.precision=bf16-mixed \
    checkpointing.every_n_train_steps=5000 \
    ${CKPT_ARG} \
    > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "[Step 8] 训练已启动，PID: ${TRAIN_PID}"

# ==================== Step 9: 启动 checkpoint 自动备份到 HDFS ====================
echo ""
echo "============================================"
echo "[Step 9] 启动 checkpoint 自动备份守护进程"
echo "============================================"

cat > /tmp/ckpt_backup.sh << EOF
#!/bin/bash
# 每 30 分钟把最新的 last.ckpt 备份到 HDFS
while true; do
    sleep 1800
    LATEST_CKPT=\$(find ${WORK_DIR}/outputs/exp_nas3r_re10k_8gpu -name "last.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | awk '{print \$2}')
    if [ -n "\$LATEST_CKPT" ] && [ -f "\$LATEST_CKPT" ]; then
        echo "[\$(date)] 备份 \$LATEST_CKPT -> HDFS"
        hdfs dfs -mkdir -p "${CKPT_HDFS_DIR}" 2>/dev/null
        hdfs dfs -put -f "\$LATEST_CKPT" "${CKPT_HDFS_DIR}/last.ckpt"
    fi
done
EOF
chmod +x /tmp/ckpt_backup.sh
nohup bash /tmp/ckpt_backup.sh > /tmp/ckpt_backup.log 2>&1 &
BACKUP_PID=$!
echo "[Step 9] 备份守护 PID: ${BACKUP_PID}"

# ==================== 结束 ====================
echo ""
echo "============================================"
echo " 全部完成 ✓"
echo "============================================"
echo ""
echo "训练日志: ${TRAIN_LOG}"
echo ""
echo "监控命令:"
echo "  tail -f ${TRAIN_LOG}"
echo "  watch -n 1 nvidia-smi"
echo "  grep 'train step' ${TRAIN_LOG} | tail -10"
echo ""
echo "停止训练:"
echo "  pkill -9 -f 'src.main'"
echo "  pkill -9 -f 'ckpt_backup.sh'"

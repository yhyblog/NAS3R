#!/bin/bash
# =============================================================================
# NAS3R 复现 v3（针对性优化：数据没备份的情况）
#
# 重点:
#   1. 先启动数据下载+HDFS备份（最长耗时）
#   2. 下载的同时并行编译 CUDA 扩展（丁老师 sm_90 优化）
#   3. 编译快，数据慢，两者并行节省时间
#   4. 下载完成后自动启动训练
#
# 使用:
#   cd /opt/tiger/mmfinetune/examples && bash reproduce_v3.sh
# =============================================================================

set -e

WORK_DIR="/opt/tiger/mmfinetune/examples/nas3r"
PARENT_DIR="/opt/tiger/mmfinetune/examples"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"
HDFS_BASE="hdfs://harunafr/home/byte_data_tt_m/data/yaohaoyang/3d_data/nas3r"

# 丁老师的建议：只编 H100 架构
export TORCH_CUDA_ARCH_LIST="9.0"
export FORCE_CUDA=1

# ==================== Step 0: 拉代码 ====================
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
    echo "[Step 0] 恢复可能被删的文件..."
    cd "${WORK_DIR}"
    git checkout -- . || true
fi
cd "${WORK_DIR}"
echo "[Step 0] ✓"

# ==================== Step 1: 启动数据下载（后台，最耗时）====================
echo ""
echo "============================================"
echo "[Step 1] 启动数据下载 + HDFS 备份（后台并行）"
echo "============================================"

mkdir -p "${LOCAL_DATA_DIR}/re10k"

LOCAL_TRAIN_COUNT=$(find "${LOCAL_DATA_DIR}/re10k/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)

if [ "$LOCAL_TRAIN_COUNT" -gt 4000 ]; then
    echo "[Step 1] 本地已有完整数据 (train=${LOCAL_TRAIN_COUNT})，跳过下载 ✓"
    DATA_READY=true
else
    DATA_READY=false

    # 先装 huggingface_hub 避免下载脚本失败
    pip install -q huggingface_hub

    # 后台下载 + 下完立即启动 HDFS 备份
    cat > /tmp/dl_and_backup.sh << 'DLEOF'
#!/bin/bash
set -e
WORK_DIR="/opt/tiger/mmfinetune/examples/nas3r"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"
HDFS_BASE="hdfs://harunafr/home/byte_data_tt_m/data/yaohaoyang/3d_data/nas3r"

echo "[$(date)] 开始 HuggingFace 下载..."
export LOCAL_DATA_DIR
python3 << PYEOF
import os
from huggingface_hub import snapshot_download
local_dir = os.path.join(os.environ["LOCAL_DATA_DIR"], "re10k")
os.makedirs(local_dir, exist_ok=True)
snapshot_download(
    repo_id="yiren-lu/re10k_pixelsplat",
    repo_type="dataset",
    local_dir=local_dir,
    max_workers=8,
)
print("HuggingFace 下载完成")
PYEOF

echo "[$(date)] 下载完成，开始生成 index.json"

# 生成 index.json
python3 << 'IDXEOF'
import json, os, torch
base = "/opt/tiger/mmfinetune/examples/nas3r/datasets/re10k"
for split in ["train", "test"]:
    d = os.path.join(base, split)
    idx_path = os.path.join(d, "index.json")
    if os.path.exists(idx_path):
        print(f"  {split}/index.json 已存在")
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
        if (i+1) % 500 == 0: print(f"    {i+1}/{len(files)}")
    with open(idx_path, "w") as f:
        json.dump(index, f)
    print(f"  {split}/index.json: {len(index)} scenes")
IDXEOF

echo "[$(date)] index.json 生成完成，开始备份到 HDFS"

# 启动 HDFS 备份（独立后台，不阻塞）
if hdfs dfs -ls / >/dev/null 2>&1; then
    hdfs dfs -mkdir -p "${HDFS_BASE}/re10k/" 2>/dev/null || true
    hdfs dfs -rm -r -f "${HDFS_BASE}/re10k/train" 2>/dev/null || true
    hdfs dfs -rm -r -f "${HDFS_BASE}/re10k/test"  2>/dev/null || true
    nohup hdfs dfs -put "${LOCAL_DATA_DIR}/re10k/train" "${HDFS_BASE}/re10k/" > /tmp/hdfs_train.log 2>&1 &
    nohup hdfs dfs -put "${LOCAL_DATA_DIR}/re10k/test"  "${HDFS_BASE}/re10k/" > /tmp/hdfs_test.log 2>&1 &
    echo "[$(date)] HDFS 备份已启动（train PID=$!, 查看 /tmp/hdfs_*.log）"
fi

# 写个标记文件，主脚本根据这个判断数据就绪
touch /tmp/data_ready.flag
echo "[$(date)] 全部完成，数据已就绪"
DLEOF
    chmod +x /tmp/dl_and_backup.sh

    rm -f /tmp/data_ready.flag
    nohup bash /tmp/dl_and_backup.sh > /tmp/data_download.log 2>&1 &
    DL_PID=$!
    echo "[Step 1] 后台下载启动，PID: ${DL_PID}"
    echo "[Step 1] 查看下载进度: tail -f /tmp/data_download.log"
fi

# ==================== Step 2: 安装 Python 依赖（和下载并行）====================
echo ""
echo "============================================"
echo "[Step 2] 安装 Python 依赖（和数据下载并行）"
echo "============================================"

python3 -m pip install --upgrade pip -q
grep -v -E "scikit-video|^torch==|^torchvision==|^torchaudio==|pytorch3d" requirements.txt > requirements_fixed.txt
pip install -r requirements_fixed.txt --no-build-isolation
pip install -q sk-video huggingface_hub
echo "[Step 2] ✓"

# ==================== Step 3: 编译 diff-gaussian-rasterization ====================
echo ""
echo "============================================"
echo "[Step 3] 编译 diff-gaussian-rasterization (sm_90)"
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

GLM_DIR="./submodules/diff-gaussian-rasterization/third_party/glm"
if [ ! -f "${GLM_DIR}/glm/glm.hpp" ]; then
    rm -rf "${GLM_DIR}"
    git clone https://github.com/g-truc/glm.git "${GLM_DIR}"
fi

# 必须清 build 缓存，TORCH_CUDA_ARCH_LIST 才生效
rm -rf ./submodules/diff-gaussian-rasterization/build/ \
       ./submodules/diff-gaussian-rasterization/*.egg-info

echo "[Step 3] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation
echo "[Step 3] ✓"

# ==================== Step 4: 编译 pytorch3d ====================
echo ""
echo "============================================"
echo "[Step 4] 安装 pytorch3d (sm_90, 约 15 分钟)"
echo "============================================"

if python3 -c "import pytorch3d" 2>/dev/null; then
    echo "[Step 4] pytorch3d 已安装 ✓"
else
    export MAX_JOBS=4
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
fi
echo "[Step 4] ✓"

# ==================== Step 5: Patch model_wrapper.py ====================
echo ""
echo "============================================"
echo "[Step 5] Patch model_wrapper.py"
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
    print("[Step 5] 已 patch ✓")
elif old in content:
    with open(path, "w") as f:
        f.write(content.replace(old, new))
    print("[Step 5] patch 成功 ✓")
else:
    print("[Step 5] [WARN] 模式未匹配")
PYEOF
echo "[Step 5] ✓"

# ==================== Step 6: 等数据下载完成 ====================
echo ""
echo "============================================"
echo "[Step 6] 等待数据下载完成"
echo "============================================"

if [ "$DATA_READY" = "true" ]; then
    # 本地有数据，但可能没 index.json，补一下
    python3 << 'PYEOF'
import json, os, torch
base = "/opt/tiger/mmfinetune/examples/nas3r/datasets/re10k"
for split in ["train", "test"]:
    d = os.path.join(base, split)
    idx_path = os.path.join(d, "index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            print(f"  {split}/index.json 已存在 ({len(json.load(f))} scenes)")
        continue
    if not os.path.isdir(d): continue
    print(f"  生成 {split}/index.json ...")
    index = {}
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".torch")):
        try:
            for item in torch.load(os.path.join(d, fn), weights_only=True):
                index[item["key"]] = fn
        except: pass
    with open(idx_path, "w") as f:
        json.dump(index, f)
    print(f"  {split}/index.json: {len(index)} scenes")
PYEOF
else
    echo "[Step 6] 等待 /tmp/data_ready.flag（预计 30-90 分钟）..."
    WAIT_COUNT=0
    while [ ! -f /tmp/data_ready.flag ]; do
        sleep 60
        WAIT_COUNT=$((WAIT_COUNT + 1))
        # 每 5 分钟打印进度
        if [ $((WAIT_COUNT % 5)) -eq 0 ]; then
            CURRENT_SIZE=$(du -sh ${LOCAL_DATA_DIR}/re10k 2>/dev/null | awk '{print $1}')
            echo "[$(date)] 等待中... 已下载: ${CURRENT_SIZE}"
        fi
    done
    echo "[Step 6] 数据就绪 ✓"
fi

echo "[Step 6] train: $(find ${LOCAL_DATA_DIR}/re10k/train -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) files"
echo "[Step 6] test:  $(find ${LOCAL_DATA_DIR}/re10k/test  -maxdepth 1 -name '*.torch' 2>/dev/null | wc -l) files"

# ==================== Step 7: 启动 8 卡训练 ====================
echo ""
echo "============================================"
echo "[Step 7] 启动 8 卡训练"
echo "============================================"

export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
unset NCCL_NET_PLUGIN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -9 -f "src.main" 2>/dev/null || true
sleep 3

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="${WORK_DIR}/train_${TIMESTAMP}.log"
echo "[Step 7] 日志: ${TRAIN_LOG}"
echo "[Step 7] 配置: 8卡 × batch=4 × bf16"
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
    > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "[Step 7] 训练 PID: ${TRAIN_PID}"

# ==================== Step 8: 启动 checkpoint 自动备份守护 ====================
echo ""
echo "============================================"
echo "[Step 8] 启动 checkpoint 自动备份"
echo "============================================"

CKPT_HDFS_DIR="${HDFS_BASE}/checkpoints"

cat > /tmp/ckpt_backup.sh << EOF
#!/bin/bash
# 每 30 分钟备份 last.ckpt 到 HDFS
while true; do
    sleep 1800
    LATEST=\$(find ${WORK_DIR}/outputs -name "last.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | awk '{print \$2}')
    if [ -n "\$LATEST" ] && [ -f "\$LATEST" ]; then
        echo "[\$(date)] 备份 \$LATEST -> ${CKPT_HDFS_DIR}/"
        hdfs dfs -mkdir -p "${CKPT_HDFS_DIR}" 2>/dev/null
        hdfs dfs -put -f "\$LATEST" "${CKPT_HDFS_DIR}/last.ckpt" && echo "  备份成功" || echo "  备份失败"
    fi
done
EOF
chmod +x /tmp/ckpt_backup.sh
pkill -f "ckpt_backup.sh" 2>/dev/null || true
nohup bash /tmp/ckpt_backup.sh > /tmp/ckpt_backup.log 2>&1 &
BACKUP_PID=$!
echo "[Step 8] 备份守护 PID: ${BACKUP_PID}"

echo ""
echo "============================================"
echo " 全部完成 ✓"
echo "============================================"
echo ""
echo "监控命令:"
echo "  tail -f ${TRAIN_LOG}"
echo "  tail -f /tmp/data_download.log"
echo "  tail -f /tmp/hdfs_train.log"
echo "  tail -f /tmp/ckpt_backup.log"
echo ""
echo "查看进度:"
echo "  grep 'train step' ${TRAIN_LOG} | tail -5"
echo "  nvidia-smi"
echo ""
echo "停止全部:"
echo "  pkill -9 -f 'src.main'"
echo "  pkill -9 -f 'ckpt_backup.sh'"
echo "  pkill -9 -f 'dl_and_backup.sh'"

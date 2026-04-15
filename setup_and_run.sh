#!/bin/bash
# =============================================================================
# NAS3R 部署脚本 - 用于远程GPU平台 (mmfinetune)
# 使用方法: cd /opt/tiger/mmfinetune/examples && bash setup_and_run.sh
# =============================================================================

set -e

# ======================== 配置区域 ========================
PROJECT_NAME="nas3r"
WORK_DIR="/opt/tiger/mmfinetune/examples/${PROJECT_NAME}"
HDFS_BASE="hdfs://harunafr/home/byte_data_tt_m/data/yaohaoyang/3d_data/nas3r"
LOCAL_DATA_DIR="${WORK_DIR}/datasets"

# ======================== 步骤0: 创建项目目录 ========================
echo "============================================"
echo "[Step 0] 创建项目目录: ${WORK_DIR}"
echo "============================================"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# 如果代码还没有clone过来, 需要先把代码复制/clone到这里
# 假设代码已经通过某种方式放到了 ${WORK_DIR}
if [ ! -f "requirements.txt" ]; then
    echo "[INFO] 项目代码不在 ${WORK_DIR}，请先将NAS3R代码复制到此目录"
    echo "[INFO] 例如: cp -r /path/to/NAS3R-master/* ${WORK_DIR}/"
    exit 1
fi

# ======================== 步骤1: 安装依赖 ========================
echo "============================================"
echo "[Step 1] 安装Python依赖"
echo "============================================"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt --no-build-isolation
pip install -e submodules/diff-gaussian-rasterization --no-build-isolation

echo "[Step 1] 依赖安装完成 ✓"

# ======================== 步骤2: 下载数据到HDFS ========================
echo "============================================"
echo "[Step 2] 检查HDFS数据 & 下载"
echo "============================================"

# 安装huggingface工具（如果没有）
pip install -q huggingface_hub datasets

# 创建HDFS目标目录
hdfs dfs -mkdir -p "${HDFS_BASE}/re10k" 2>/dev/null || true
hdfs dfs -mkdir -p "${HDFS_BASE}/acid" 2>/dev/null || true

# 检查HDFS上是否已有数据
RE10K_EXISTS=$(hdfs dfs -ls "${HDFS_BASE}/re10k/train/" 2>/dev/null | wc -l)
ACID_EXISTS=$(hdfs dfs -ls "${HDFS_BASE}/acid/train/" 2>/dev/null | wc -l)

TEMP_DOWNLOAD="/tmp/nas3r_data"
mkdir -p "${TEMP_DOWNLOAD}"

if [ "$RE10K_EXISTS" -lt 2 ]; then
    echo "[Step 2.1] 下载 re10k 数据集..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='yiren-lu/re10k_pixelsplat',
    repo_type='dataset',
    local_dir='${TEMP_DOWNLOAD}/re10k',
    resume_download=True,
)
print('re10k 下载完成')
"
    echo "[Step 2.1] 上传 re10k 到 HDFS..."
    hdfs dfs -put -f "${TEMP_DOWNLOAD}/re10k/"* "${HDFS_BASE}/re10k/"
    echo "[Step 2.1] re10k 上传HDFS完成 ✓"
else
    echo "[Step 2.1] re10k 数据已在HDFS上，跳过下载 ✓"
fi

if [ "$ACID_EXISTS" -lt 2 ]; then
    echo "[Step 2.2] 下载 ACID 数据集..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Sensen02/ACID_PixelSplat',
    repo_type='dataset',
    local_dir='${TEMP_DOWNLOAD}/acid',
    resume_download=True,
)
print('ACID 下载完成')
"
    echo "[Step 2.2] 上传 ACID 到 HDFS..."
    hdfs dfs -put -f "${TEMP_DOWNLOAD}/acid/"* "${HDFS_BASE}/acid/"
    echo "[Step 2.2] ACID 上传HDFS完成 ✓"
else
    echo "[Step 2.2] ACID 数据已在HDFS上，跳过下载 ✓"
fi

# 清理临时文件
rm -rf "${TEMP_DOWNLOAD}"

echo "[Step 2] 数据准备完成 ✓"

# ======================== 步骤3: 从HDFS拉取数据到本地 ========================
echo "============================================"
echo "[Step 3] 从HDFS拉取数据到本地"
echo "============================================"
mkdir -p "${LOCAL_DATA_DIR}/re10k/train"
mkdir -p "${LOCAL_DATA_DIR}/re10k/test"
mkdir -p "${LOCAL_DATA_DIR}/acid/train"
mkdir -p "${LOCAL_DATA_DIR}/acid/test"

# 拉取re10k
echo "[Step 3.1] 拉取 re10k..."
hdfs dfs -get -f "${HDFS_BASE}/re10k/train/"* "${LOCAL_DATA_DIR}/re10k/train/" 2>/dev/null || \
hdfs dfs -get -f "${HDFS_BASE}/re10k/"* "${LOCAL_DATA_DIR}/re10k/" 2>/dev/null || \
echo "[WARN] re10k 拉取可能需要手动调整路径"

hdfs dfs -get -f "${HDFS_BASE}/re10k/test/"* "${LOCAL_DATA_DIR}/re10k/test/" 2>/dev/null || true

# 拉取acid
echo "[Step 3.2] 拉取 ACID..."
hdfs dfs -get -f "${HDFS_BASE}/acid/train/"* "${LOCAL_DATA_DIR}/acid/train/" 2>/dev/null || \
hdfs dfs -get -f "${HDFS_BASE}/acid/"* "${LOCAL_DATA_DIR}/acid/" 2>/dev/null || \
echo "[WARN] ACID 拉取可能需要手动调整路径"

hdfs dfs -get -f "${HDFS_BASE}/acid/test/"* "${LOCAL_DATA_DIR}/acid/test/" 2>/dev/null || true

echo "[Step 3] 数据拉取完成 ✓"

# 验证数据
echo ""
echo "--- 数据目录结构 ---"
echo "re10k train: $(ls ${LOCAL_DATA_DIR}/re10k/train/*.torch 2>/dev/null | wc -l) torch files"
echo "re10k test:  $(ls ${LOCAL_DATA_DIR}/re10k/test/*.torch 2>/dev/null | wc -l) torch files"
echo "acid train:  $(ls ${LOCAL_DATA_DIR}/acid/train/*.torch 2>/dev/null | wc -l) torch files"
echo "acid test:   $(ls ${LOCAL_DATA_DIR}/acid/test/*.torch 2>/dev/null | wc -l) torch files"

echo ""
echo "============================================"
echo "环境和数据准备完成！"
echo "============================================"

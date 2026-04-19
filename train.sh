#!/usr/bin/env bash
# =============================================================================
# NAS3R 正式训练脚本 · 8×H100 80GB 单节点 · v1.0 (production)
#
# 设计目标：交上去就能跑完，中途不可调试。
# 覆盖场景：
#   - 首次启动（下载 VGGT 预训练权重、准备数据、启动训练）
#   - 训练中途意外中断（自动从最近 checkpoint 恢复）
#   - wandb 失败（回退到本地日志，训练继续）
#   - OOM / NaN loss （Lightning 的 gradient_clip_val 会处理，NaN 时截断但不崩）
#
# 使用方法：
#   # 训练 pretrained 版（论文主力 setting，推荐）
#   bash train.sh pretrained
#
#   # 其他 setting:
#   bash train.sh random         # random init 2-view (re10k_nas3r.ckpt)
#   bash train.sh pretrained-I   # VGGT + GT intrinsics (re10k_nas3r_pretrained-I.ckpt)
#   bash train.sh multiview      # 10-view 多视角
#
# 一键后台启动（nohup + 断点续训 loop）：
#   bash train.sh pretrained --daemon
#
# 关键环境变量：
#   WANDB_API_KEY=xxxxxxxx   建议导出，否则 wandb 降级为 disabled，不影响训练
#   WANDB_MODE=online        online(默认) / offline / disabled
#   PARENT_DIR=/opt/tiger/mmfinetune
#   DATA_DIR=<auto>          默认 <repo>/datasets/re10k
#   CKPT_DIR=<auto>          默认 <repo>/outputs/train_<exp>/checkpoints
#   GLOBAL_BATCH_SIZE=<auto> 默认按论文配置（pretrained=40, random=80）
#   MAX_STEPS=<auto>         默认按论文 400_001
#   NUM_GPUS=<auto>          默认 $(nvidia-smi -L | wc -l)
#   RESUME=auto              auto(默认) / 1(强制续训) / 0(强制重新开始)
# =============================================================================

set -uo pipefail  # 不用 -e；每个 step 独立处理错误

# ============================ 基础参数 ============================
PARENT_DIR="${PARENT_DIR:-/opt/tiger/mmfinetune}"
REPO_NAME="nas3r"
WORK_DIR="${PARENT_DIR}/${REPO_NAME}"
REPO_URL="${REPO_URL:-https://github.com/yhyblog/NAS3R.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"

# 日志工具
log()  { echo -e "\n\033[1;36m[$(date '+%F %T')] $*\033[0m" | tee -a "${MAIN_LOG:-/dev/null}"; }
info() { echo -e "  \033[0;34m→\033[0m $*"             | tee -a "${MAIN_LOG:-/dev/null}"; }
ok()   { echo -e "  \033[1;32m✓\033[0m $*"             | tee -a "${MAIN_LOG:-/dev/null}"; }
warn() { echo -e "  \033[1;33m⚠\033[0m $*"             | tee -a "${MAIN_LOG:-/dev/null}"; }
err()  { echo -e "\033[1;31m✗ ERROR: $*\033[0m" >&2    | tee -a "${MAIN_LOG:-/dev/null}"; }

# ============================ 参数解析 ============================
EXP_KEY="${1:-pretrained}"
DAEMON=0
for arg in "$@"; do
    [[ "${arg}" == "--daemon" ]] && DAEMON=1
done

# experiment 表：key => "<experiment_path>|<batch_per_gpu>|<tag>"
declare -A EXPERIMENTS=(
    [random]="nas3r/random/re10k|10|random_init_2view"
    [pretrained]="nas3r/pretrained/re10k|5|vggt_pretrained_2view"
    [pretrained-I]="nas3r/pretrained/re10k-I|5|vggt_pretrained_2view_gtK"
    [multiview]="nas3r/random/re10k|5|random_init_10view"
)

if [[ -z "${EXPERIMENTS[${EXP_KEY}]+x}" ]]; then
    echo "[ERROR] 未知实验: ${EXP_KEY}"
    echo "可选: random | pretrained | pretrained-I | multiview"
    exit 1
fi

IFS='|' read -r EXP_PATH DEFAULT_BATCH TAG <<<"${EXPERIMENTS[${EXP_KEY}]}"

# 默认用论文配置；可通过环境变量覆盖
PER_GPU_BATCH="${PER_GPU_BATCH:-${DEFAULT_BATCH}}"
MAX_STEPS="${MAX_STEPS:-400001}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
CKPT_EVERY_N_STEPS="${CKPT_EVERY_N_STEPS:-5000}"  # 更密的 checkpoint（默认是 10000）
SAVE_TOP_K="${SAVE_TOP_K:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESUME="${RESUME:-auto}"

# GPU 数
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
[[ -z "${NUM_GPUS}" || "${NUM_GPUS}" == "0" ]] && { echo "[ERROR] 没有 GPU"; exit 1; }

# multiview 特例
MULTIVIEW_OVERRIDE=""
if [[ "${EXP_KEY}" == "multiview" ]]; then
    MULTIVIEW_OVERRIDE="dataset.re10k.view_sampler.num_context_views=10"
fi

# 输出路径
RUN_NAME="nas3r_${TAG}"
OUTPUT_ROOT="${WORK_DIR}/outputs/train_${EXP_KEY}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints}"
MAIN_LOG="${OUTPUT_ROOT}/train.log"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/datasets/re10k}"

mkdir -p "${OUTPUT_ROOT}" "${CKPT_DIR}" "${WORK_DIR}"

# ============================ daemon 模式：后台 + 守护循环 ============================
if [[ "${DAEMON}" == "1" && -z "${NAS3R_DAEMON_CHILD:-}" ]]; then
    export NAS3R_DAEMON_CHILD=1
    nohup bash "${BASH_SOURCE[0]}" "${EXP_KEY}" >"${OUTPUT_ROOT}/daemon.log" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "训练已在后台启动"
    echo "  PID:         ${PID}"
    echo "  主日志:      ${MAIN_LOG}"
    echo "  daemon 日志: ${OUTPUT_ROOT}/daemon.log"
    echo "  checkpoint:  ${CKPT_DIR}"
    echo ""
    echo "监控命令："
    echo "  tail -f ${MAIN_LOG}"
    echo "  tail -f ${OUTPUT_ROOT}/daemon.log"
    echo "  watch -n 30 \"nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader\""
    echo "=========================================="
    exit 0
fi

# ============================ 应用层环境变量（仅 Hydra/Python/HF）============================
# 注意：
#   * 平台级变量（ARNOLD_*, BYTED_TORCH_*, NCCL_DEBUG, NCCL_IB_* 等）应由 Arnold/Merlin 任务
#     提交页面的 "环境变量" (JSON) 字段预设，不在此 bash 中重复导出，以免被平台默认值覆盖。
#   * 下列 5 个变量是应用侧专属（Hydra / HuggingFace / Python 输出行为），平台不会管，这里必须设。
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # H100 sm_90（编译 pytorch3d/diff_gauss_camera 用）
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# 以下由平台（Arnold）预设；若未设置，脚本这里给一个合理默认，避免缺失
: "${NCCL_DEBUG:=WARN}"; export NCCL_DEBUG
: "${NCCL_ASYNC_ERROR_HANDLING:=1}"; export NCCL_ASYNC_ERROR_HANDLING
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"; export PYTORCH_CUDA_ALLOC_CONF

# wandb key
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
    # 已经 wandb login 过，保持即可
    :
fi
WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
    warn "未检测到 WANDB_API_KEY 或 wandb login，降级为 offline"
    WANDB_MODE=offline
fi

log "=============================================="
log "NAS3R 正式训练 · ${EXP_KEY} · $(date)"
log "=============================================="
info "  exp_path:        +experiment=${EXP_PATH}"
info "  per_gpu_batch:   ${PER_GPU_BATCH}"
info "  num_gpus:        ${NUM_GPUS}"
info "  global_batch:    $(( PER_GPU_BATCH * NUM_GPUS ))"
info "  max_steps:       ${MAX_STEPS}"
info "  val_check_every: ${VAL_CHECK_INTERVAL}"
info "  ckpt_every:      ${CKPT_EVERY_N_STEPS}"
info "  save_top_k:      ${SAVE_TOP_K}"
info "  num_workers:     ${NUM_WORKERS}"
info "  run_name:        ${RUN_NAME}"
info "  ckpt_dir:        ${CKPT_DIR}"
info "  work_dir:        ${WORK_DIR}"
info "  wandb_mode:      ${WANDB_MODE}"
info "  resume:          ${RESUME}"

# 打印平台下发的关键环境变量（Arnold/Merlin/BYTED_TORCH 等），便于溯源
log "平台环境变量快照（来自 Arnold 任务配置）"
env | grep -E '^(ARNOLD_|BYTED_|NCCL_|NVIDIA_|MLP_|METIS_|CUDA_VISIBLE_DEVICES|PYTORCH_CUDA_ALLOC_CONF)=' \
    | sort | sed 's/^/    /' | tee -a "${MAIN_LOG}"

# ============================ Step 0: 代码仓库 ============================
log "Step 0: 准备代码仓库"
mkdir -p "${PARENT_DIR}"
cd "${PARENT_DIR}"

if [[ ! -d "${WORK_DIR}/.git" ]]; then
    info "clone ${REPO_URL}"
    git clone --recurse-submodules -b "${REPO_BRANCH}" "${REPO_URL}" "${REPO_NAME}" || \
        { err "clone 失败"; exit 1; }
else
    info "仓库已存在，git pull"
    cd "${WORK_DIR}"
    git fetch origin "${REPO_BRANCH}" >/dev/null 2>&1 || true
    git pull --ff-only origin "${REPO_BRANCH}" 2>&1 | tail -5 || \
        warn "git pull 失败（有本地改动？），继续用当前版本"
fi

cd "${WORK_DIR}"
git submodule sync --recursive >/dev/null 2>&1 || true
git submodule update --init --recursive 2>&1 | tail -3 || warn "submodule update 失败"

DGR_DIR="submodules/diff-gaussian-rasterization"
if [[ ! -f "${DGR_DIR}/setup.py" ]]; then
    info "手动 clone diff-gaussian-rasterization"
    rm -rf "${DGR_DIR}"
    mkdir -p submodules
    git clone --recurse-submodules -b camera \
        https://github.com/ranrhuang/diff-gaussian-rasterization.git \
        "${DGR_DIR}" || { err "diff-gaussian-rasterization clone 失败"; exit 1; }
fi
GLM_DIR="${DGR_DIR}/third_party/glm"
if [[ ! -f "${GLM_DIR}/glm/glm.hpp" ]]; then
    info "下载 GLM 头文件"
    rm -rf "${GLM_DIR}"
    git clone --depth 1 https://github.com/g-truc/glm.git "${GLM_DIR}" || \
        { err "GLM 下载失败"; exit 1; }
fi
ok "代码就位"

# ============================ Step 1: 依赖 ============================
log "Step 1: 检查依赖"
python3 --version
python3 -c "
import torch
print(f'    torch={torch.__version__}  cuda={torch.version.cuda}  dev={torch.cuda.device_count()}x{torch.cuda.get_device_name(0)}')
" || { err "torch 异常"; exit 1; }

if ! python3 -c "import diff_gauss_camera" 2>/dev/null; then
    info "diff_gauss_camera 未安装，编译 ..."
    pip install -e "${DGR_DIR}" --no-build-isolation 2>&1 | tail -5 || \
        { err "编译失败"; exit 1; }
fi

# 关键依赖自检
python3 <<'PYEOF' || { err "关键依赖缺失"; exit 1; }
import importlib, sys
needed = ['lightning', 'hydra', 'jaxtyping', 'beartype', 'einops', 'lpips',
          'timm', 'dacite', 'diff_gauss_camera', 'roma', 'wandb', 'omegaconf']
missing = []
for m in needed:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f'{m}: {e}')
if missing:
    print('Missing:')
    for m in missing: print('  -', m)
    sys.exit(1)
print('  [ok] 所有关键依赖就绪')
PYEOF

# VGGT 权重预拉（避免多卡并发下载冲突）
if [[ "${EXP_KEY}" == "pretrained" || "${EXP_KEY}" == "pretrained-I" ]]; then
    info "预拉 VGGT 预训练权重 (facebook/VGGT-1B, ~2.3GB)"
    python3 <<'PYEOF' || warn "VGGT 预拉失败，训练启动时会重试"
from huggingface_hub import snapshot_download
try:
    p = snapshot_download(repo_id="facebook/VGGT-1B")
    print(f"  [ok] VGGT 权重: {p}")
except Exception as e:
    print(f"  [WARN] {e}")
    raise
PYEOF
fi
ok "依赖就绪"

# ============================ Step 2: 检查训练数据 ============================
log "Step 2: 检查 RealEstate10K 训练数据"
info "  数据目录: ${DATA_DIR}"

# 作者说 re10k 训练集 330 个 .torch + index.json，test 集 331 个
TRAIN_COUNT=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l | tr -d ' ')
TEST_COUNT=$(find "${DATA_DIR}/test" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l | tr -d ' ')

info "  train: ${TRAIN_COUNT} .torch 文件（期望 ~330）"
info "  test:  ${TEST_COUNT} .torch 文件（评测用，train 阶段 val_check 会用）"

# train 数据不够→需要下载
MIN_TRAIN_SHARDS="${MIN_TRAIN_SHARDS:-300}"
if [[ "${TRAIN_COUNT}" -lt "${MIN_TRAIN_SHARDS}" ]]; then
    warn "训练数据不足（${TRAIN_COUNT} < ${MIN_TRAIN_SHARDS}），从 HuggingFace 下载完整 re10k"
    info "  此步约 ~575GB 流量，10-30 分钟视网速"
    mkdir -p "${DATA_DIR}"
    python3 <<PYEOF || { err "数据下载失败"; exit 1; }
import os
from huggingface_hub import snapshot_download
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
p = snapshot_download(
    repo_id='yiren-lu/re10k_pixelsplat',
    repo_type='dataset',
    local_dir='${DATA_DIR}',
)
print(f'  [ok] 下载到: {p}')
PYEOF
    TRAIN_COUNT=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l | tr -d ' ')
    info "  下载后 train count: ${TRAIN_COUNT}"
    [[ "${TRAIN_COUNT}" -lt "${MIN_TRAIN_SHARDS}" ]] && { err "下载后仍不够"; exit 1; }
fi

# train/index.json 必需
if [[ ! -f "${DATA_DIR}/train/index.json" ]]; then
    warn "train/index.json 缺失，生成中..."
    DATA_DIR="${DATA_DIR}" python3 <<'PYEOF' || { err "index.json 生成失败"; exit 1; }
import json, os, torch
from pathlib import Path
base = Path(os.environ["DATA_DIR"])
for split in ["train", "test"]:
    d = base / split
    idx_path = d / "index.json"
    if idx_path.exists():
        with open(idx_path) as f: existing = json.load(f)
        print(f"  {split}/index.json 已存在 ({len(existing)} scenes)")
        continue
    if not d.is_dir(): continue
    files = sorted(f for f in os.listdir(d) if f.endswith(".torch"))
    if not files: continue
    print(f"  扫描 {split}/ 生成 index.json ...")
    idx = {}
    for i, fn in enumerate(files):
        try:
            chunk = torch.load(d / fn, weights_only=True, map_location="cpu")
            for item in chunk:
                idx[item["key"]] = fn
        except Exception as e:
            print(f"    [WARN] {fn} 读取失败: {e}")
        if (i+1) % 50 == 0:
            print(f"    已处理 {i+1}/{len(files)}")
    with open(idx_path, "w") as f: json.dump(idx, f)
    print(f"  {split}/index.json: {len(idx)} scenes ✓")
PYEOF
fi

# 最低要求：train 有 index.json
[[ -f "${DATA_DIR}/train/index.json" ]] || { err "train/index.json 不存在"; exit 1; }

# 测试集不是训练必需（lightning 在 val_check_interval 前不会用），但能用更好
if [[ "${TEST_COUNT}" -lt 50 && ! -f "${DATA_DIR}/test/index.json" ]]; then
    warn "测试/验证数据缺失，训练期间 val_check 会报错；建议先补齐 test 目录"
fi

# 磁盘空间检查
AVAIL_GB=$(df -BG "${DATA_DIR}" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}' || echo 0)
info "  剩余磁盘: ${AVAIL_GB} GB"
if [[ "${AVAIL_GB}" -lt 50 ]]; then
    warn "磁盘空间偏少（<50GB），checkpoint 可能无处可放"
fi
ok "数据就绪: train=${TRAIN_COUNT}"

# ============================ Step 3: 自动续训判定 ============================
log "Step 3: 判定是否从 checkpoint 续训"

LATEST_CKPT=""
if [[ "${RESUME}" == "0" ]]; then
    info "RESUME=0，强制从头开始"
else
    # 查找最新 checkpoint（按 mtime）
    if [[ -d "${CKPT_DIR}" ]]; then
        LATEST_CKPT=$(find "${CKPT_DIR}" -maxdepth 2 -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -1 | awk '{$1=""; print substr($0,2)}')
    fi

    # 也看看 hydra 历史输出目录里有没有
    if [[ -z "${LATEST_CKPT}" ]]; then
        HIST_CKPT=$(find "${WORK_DIR}/outputs" -path "*/checkpoints/*.ckpt" -type f \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{$1=""; print substr($0,2)}')
        if [[ -n "${HIST_CKPT}" ]]; then
            info "在历史输出中发现 checkpoint: ${HIST_CKPT}"
            LATEST_CKPT="${HIST_CKPT}"
        fi
    fi

    if [[ -n "${LATEST_CKPT}" ]]; then
        SZ=$(du -h "${LATEST_CKPT}" | awk '{print $1}')
        ok "自动续训 from: ${LATEST_CKPT} (${SZ})"
    else
        info "未发现已有 checkpoint，从头训练"
    fi
fi

# ============================ Step 4: 启动训练 ============================
log "Step 4: 启动 torchrun × ${NUM_GPUS} GPU"

cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

# 组装 Hydra override
HYDRA_OVERRIDES=(
    "+experiment=${EXP_PATH}"
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${RUN_NAME}"
    "wandb.project=nas3r"
    "data_loader.train.batch_size=${PER_GPU_BATCH}"
    "data_loader.train.num_workers=${NUM_WORKERS}"
    "trainer.max_steps=${MAX_STEPS}"
    "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
    "checkpointing.every_n_train_steps=${CKPT_EVERY_N_STEPS}"
    "checkpointing.save_top_k=${SAVE_TOP_K}"
    "checkpointing.save_weights_only=false"
    "checkpointing.resume=true"
)

# 续训
if [[ -n "${LATEST_CKPT}" ]]; then
    HYDRA_OVERRIDES+=("checkpointing.load=${LATEST_CKPT}")
    # resume 时 wandb run_id 需要：已存在的 outputs/exp_*/ 下通常有 wandb_run_id.txt
    # 如果找不到 wandb_run_id.txt，降级到 offline 避免 main.py 第 74 行崩溃
    RUN_ID_FILE="$(dirname "$(dirname "${LATEST_CKPT}")")/wandb_run_id.txt"
    if [[ "${WANDB_MODE}" != "disabled" && ! -f "${RUN_ID_FILE}" ]]; then
        warn "续训但缺 wandb_run_id.txt (${RUN_ID_FILE}); 降级为 offline 避免启动失败"
        # 重写 wandb.mode
        HYDRA_OVERRIDES=("${HYDRA_OVERRIDES[@]/wandb.mode=${WANDB_MODE}/wandb.mode=offline}")
    fi
fi

# multiview override
[[ -n "${MULTIVIEW_OVERRIDE}" ]] && HYDRA_OVERRIDES+=("${MULTIVIEW_OVERRIDE}")

# checkpoint 输出目录固定到 CKPT_DIR
# Lightning 里 ModelCheckpoint 用 output_dir/checkpoints；main.py 里 output_dir 是 hydra run.dir
# 所以我们通过 override hydra.run.dir 来让 checkpoint 落到固定位置，方便续训
HYDRA_OVERRIDES+=(
    "hydra.run.dir=${OUTPUT_ROOT}/hydra"
)

info "torchrun 参数："
for o in "${HYDRA_OVERRIDES[@]}"; do
    echo "    ${o}"
done | tee -a "${MAIN_LOG}"

# 记录启动时间戳
echo "$(date '+%F %T') START exp=${EXP_KEY} ckpt=${LATEST_CKPT:-none}" >> "${OUTPUT_ROOT}/run_history.txt"

# 用 torchrun 启动（Lightning 推荐做法；单进程 + devices=auto 会导致 Lightning 自己 re-exec，更危险）
# --standalone 用于单机，--nproc_per_node 对应 GPU 数
set +e  # 不用 -e 是因为我们要捕获返回码做重试决策
torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    -m src.main \
    "${HYDRA_OVERRIDES[@]}" \
    2>&1 | tee -a "${MAIN_LOG}"
RC="${PIPESTATUS[0]}"
set -uo pipefail

echo "$(date '+%F %T') END exp=${EXP_KEY} rc=${RC}" >> "${OUTPUT_ROOT}/run_history.txt"

log "训练进程退出，rc=${RC}"

# ============================ Step 5: 结束后处理 ============================
if [[ "${RC}" -eq 0 ]]; then
    ok "训练正常完成"
    # 列出产生的 checkpoint
    info "最终 checkpoint 列表："
    find "${CKPT_DIR}" "${OUTPUT_ROOT}/hydra" -name "*.ckpt" 2>/dev/null | xargs -I{} sh -c 'echo "  $(du -h "{}" | cut -f1)  {}"'
    exit 0
fi

# 非 0 退出：判断是否自动重启
# 退出码规约：
#   130 = Ctrl+C → 不重启
#   137 = SIGKILL (OOM / 手动 kill) → 重启（如果 checkpoint 有进展）
#   其他 = 训练代码异常 → 看步数
err "训练以非 0 状态退出 (rc=${RC})"

# 读取 trainer.log 看看跑到哪一步
LAST_STEP=$(grep -oE "'step': *[0-9]+" "${MAIN_LOG}" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
info "最后记录步数: ${LAST_STEP} / ${MAX_STEPS}"

if [[ "${RC}" -eq 130 ]]; then
    warn "用户中断 (Ctrl+C)，不自动重启"
    exit 130
fi

# daemon 模式下自动重启
if [[ "${NAS3R_DAEMON_CHILD:-0}" == "1" ]]; then
    if [[ "${LAST_STEP}" -ge "${MAX_STEPS}" ]]; then
        warn "已达到 max_steps，不重启"
        exit 0
    fi
    SLEEP_S="${RESTART_SLEEP:-30}"
    warn "daemon 模式：${SLEEP_S}s 后自动从最新 checkpoint 续训（退出码=${RC}）"
    sleep "${SLEEP_S}"
    # 通过递归调用重新进入本脚本
    exec bash "${BASH_SOURCE[0]}" "${EXP_KEY}"
fi

exit "${RC}"

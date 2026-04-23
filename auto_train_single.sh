#!/usr/bin/env bash
# =============================================================================
# NAS3R 单卡 H800 训练脚本
#
# 动机：多卡 DDP 在字节/GCP 镜像上 NCCL 偶发死锁（每 10-13h 一次），反复崩
# 溃影响效率。单卡不走任何 NCCL collective，根本不会有那个问题。
#
# 默认配置：
#   - 单卡（GPU 0），batch_size=20（H800 80GB 能吃下）
#   - max_steps=80000（原 400k 的 1/5），约 3-5 天跑完
#   - 每 1000 step 存 ckpt，保留 3 份
#   - DISABLE_VAL 默认 1（val 只是监控，不影响训练权重）
#   - 用 hang watchdog 监控日志 freshness（防止 dataloader worker 静默卡死）
#
# 用法：
#   # 一般情况（从零开始训）
#   bash auto_train_single.sh
#
#   # 从已有 ckpt 恢复（比如多卡训到一半想切单卡继续）
#   RESUME_CKPT=/path/to/xxx.ckpt bash auto_train_single.sh
#
# 常用环境变量：
#   GPU_ID=0             用哪张卡
#   BATCH_SIZE=20        单卡 batch（80GB 极限约 24-28，保守 20）
#   MAX_STEPS=80000      总训练步数
#   NUM_WORKERS=8        dataloader 工作线程
#   CKPT_EVERY=1000      每多少 step 存 ckpt
#   WANDB_NAME=xxx       实验名（默认带时间戳）
# =============================================================================

set -uo pipefail

# ============================ 配置 ============================
PARENT_DIR="${PARENT_DIR:-/opt/tiger/mmfinetune}"
REPO_NAME="nas3r"
WORK_DIR="${PARENT_DIR}/${REPO_NAME}"

EXPERIMENT="${EXPERIMENT:-nas3r/random/re10k}"
WANDB_NAME="${WANDB_NAME:-nas3r_single_$(date +%m%d_%H%M)}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-80000}"

CKPT_EVERY="${CKPT_EVERY:-1000}"
CKPT_KEEP="${CKPT_KEEP:-3}"
VAL_INTERVAL="${VAL_INTERVAL:-999999999}"  # 默认关 val，免得再踩 Lightning 坑
if [[ "${ENABLE_VAL:-0}" == "1" ]]; then
    VAL_INTERVAL="${VAL_INTERVAL_WHEN_ON:-5000}"
fi

DATA_DIR="${DATA_DIR:-${WORK_DIR}/datasets/re10k}"
OUT_ROOT="${OUT_ROOT:-${WORK_DIR}/outputs/train}"
MONITOR_DIR="${MONITOR_DIR:-${WORK_DIR}/outputs/monitor}"
mkdir -p "${OUT_ROOT}" "${MONITOR_DIR}"

OUT_PATH="${OUT_ROOT}/${WANDB_NAME}"
mkdir -p "${OUT_PATH}"

# H100/H800 都是 sm_90
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export FORCE_CUDA=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 单卡 → 只让它看见一张 GPU
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

log()  { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
info() { echo -e "  \033[0;34m→\033[0m $*"; }
ok()   { echo -e "  \033[1;32m✓\033[0m $*"; }
warn() { echo -e "  \033[1;33m⚠\033[0m $*"; }
err()  { echo -e "\033[1;31m✗ ERROR: $*\033[0m" >&2; }

# ============================ 自动 resume ============================
RESUME_ARG=()
if [[ "${NO_RESUME:-0}" == "1" ]]; then
    info "NO_RESUME=1: 从头开始训练（忽略已有 checkpoint）"
elif [[ -n "${RESUME_CKPT:-}" ]]; then
    if [[ -f "${RESUME_CKPT}" ]]; then
        info "从指定 ckpt resume: ${RESUME_CKPT}"
        RESUME_ARG=( "checkpointing.load=${RESUME_CKPT}" )
    else
        err "RESUME_CKPT 不存在: ${RESUME_CKPT}"
        exit 1
    fi
else
    # 只找本次 WANDB_NAME 对应实验的 ckpt（避免用了之前多卡的 ckpt）
    latest_ckpt=$(find "${WORK_DIR}/outputs/exp_${WANDB_NAME}" -name "*.ckpt" 2>/dev/null | sort -V | tail -1)
    if [[ -n "${latest_ckpt}" && -f "${latest_ckpt}" ]]; then
        info "自动 resume 从本次实验的最新 ckpt: ${latest_ckpt}"
        RESUME_ARG=( "checkpointing.load=${latest_ckpt}" )
    else
        info "未找到本次实验的 ckpt，从头开始训练"
        info "  （如要从多卡 ckpt 继续，请显式设 RESUME_CKPT=/path/to/xxx.ckpt）"
    fi
fi

# ============================ GPU 监控（轻量） ============================
start_gpu_monitor() {
    local log_file="${MONITOR_DIR}/gpu_mem_${WANDB_NAME}.csv"
    info "启动 GPU 监控 → ${log_file}"
    (
        echo "timestamp,gpu_idx,mem_used_mb,mem_total_mb,util_pct" > "${log_file}"
        while true; do
            nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits -i "${GPU_ID}" 2>/dev/null | \
                awk -F', ' '{print $1","$2","$3","$4","$5}' >> "${log_file}" 2>/dev/null || true
            sleep 60
        done
    ) >/dev/null 2>&1 &
    echo $! > "${MONITOR_DIR}/monitor.pid"
}

stop_gpu_monitor() {
    if [[ -f "${MONITOR_DIR}/monitor.pid" ]]; then
        kill "$(cat ${MONITOR_DIR}/monitor.pid)" 2>/dev/null || true
        rm -f "${MONITOR_DIR}/monitor.pid"
    fi
}

# ============================ Hang Watchdog（兜底） ============================
# 单卡没有 NCCL 死锁风险，但还是加个 watchdog 防 dataloader worker 卡死。
# 默认 15 分钟日志静默就强杀（单卡 load 数据偶尔会慢，不能设太短）。
start_hang_watchdog() {
    local log_file="$1"
    local timeout_sec="${HANG_TIMEOUT_SEC:-900}"
    (
        local waited=0
        while [[ ! -f "${log_file}" && ${waited} -lt 120 ]]; do
            sleep 5
            waited=$((waited + 5))
        done
        [[ ! -f "${log_file}" ]] && exit 0

        while true; do
            sleep 60
            [[ ! -f "${log_file}" ]] && exit 0
            local now mtime stale
            now=$(date +%s)
            mtime=$(stat -c '%Y' "${log_file}" 2>/dev/null || echo 0)
            stale=$((now - mtime))
            if [[ ${stale} -gt ${timeout_sec} ]]; then
                echo "" >> "${log_file}"
                echo "[HANG WATCHDOG] 日志静默 ${stale}s > ${timeout_sec}s，强杀 python" >> "${log_file}"
                pkill -9 -f "src.main" 2>/dev/null || true
                exit 0
            fi
        done
    ) >/dev/null 2>&1 &
    echo $! > "${MONITOR_DIR}/hang_watchdog.pid"
    info "启动 Hang Watchdog (pid=$(cat ${MONITOR_DIR}/hang_watchdog.pid), timeout=${timeout_sec}s)"
}

stop_hang_watchdog() {
    if [[ -f "${MONITOR_DIR}/hang_watchdog.pid" ]]; then
        kill "$(cat ${MONITOR_DIR}/hang_watchdog.pid)" 2>/dev/null || true
        rm -f "${MONITOR_DIR}/hang_watchdog.pid"
    fi
}

trap 'stop_gpu_monitor; stop_hang_watchdog' EXIT

# ============================ 训练 ============================
main_train() {
    log "NAS3R 单卡训练"
    info "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    info "experiment: ${EXPERIMENT}"
    info "wandb name: ${WANDB_NAME}"
    info "batch:      ${BATCH_SIZE}  workers: ${NUM_WORKERS}"
    info "max_steps:  ${MAX_STEPS}  (原 400k 的 $((MAX_STEPS * 100 / 400000))%)"
    info "ckpt:       每 ${CKPT_EVERY} step 存，保留最新 ${CKPT_KEEP} 份"
    info "val:        每 ${VAL_INTERVAL} step（DISABLE 时是 9.99 亿）"
    info "data:       ${DATA_DIR}"
    info "output:     ${OUT_PATH}"

    cd "${WORK_DIR}"

    # 自动重启循环
    local MAX_RESTARTS="${MAX_RESTARTS:-10}"
    local attempt=0
    local rc=0
    local run_resume_arg=( "${RESUME_ARG[@]}" )

    start_gpu_monitor

    while true; do
        attempt=$((attempt + 1))
        info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        info "训练启动第 ${attempt} 次"
        [[ ${#run_resume_arg[@]} -gt 0 ]] && info "  ${run_resume_arg[0]}"
        info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        local per_run_log="${OUT_PATH}/train_attempt${attempt}.log"
        rm -f "${per_run_log}"
        start_hang_watchdog "${per_run_log}"

        python3 -m src.main \
            +experiment="${EXPERIMENT}" \
            wandb.mode=disabled \
            wandb.name="${WANDB_NAME}" \
            "dataset.re10k.roots=[${DATA_DIR}]" \
            data_loader.train.batch_size="${BATCH_SIZE}" \
            data_loader.train.num_workers="${NUM_WORKERS}" \
            checkpointing.every_n_train_steps="${CKPT_EVERY}" \
            checkpointing.save_top_k="${CKPT_KEEP}" \
            checkpointing.save_weights_only=false \
            checkpointing.resume=true \
            trainer.val_check_interval="${VAL_INTERVAL}" \
            trainer.max_steps="${MAX_STEPS}" \
            "${run_resume_arg[@]}" \
            2>&1 | tee "${per_run_log}"
        rc=${PIPESTATUS[0]}

        stop_hang_watchdog

        # 合并日志
        cat "${per_run_log}" >> "${OUT_PATH}/train.log"

        if [[ ${rc} -eq 0 ]]; then
            ok "训练正常完成 (attempt=${attempt})"
            break
        fi

        if [[ "${NO_AUTORESTART:-0}" == "1" ]]; then
            err "训练退出 (rc=${rc})，NO_AUTORESTART=1 不重启"
            break
        fi

        if [[ ${attempt} -ge ${MAX_RESTARTS} ]]; then
            err "已重启 ${attempt} 次达上限，放弃"
            break
        fi

        warn "训练崩溃 (rc=${rc})，自动重启（第 $((attempt+1)) 次）..."
        pkill -9 -f "src.main" 2>/dev/null || true
        sleep 10

        # 找最新 ckpt
        local new_ckpt
        new_ckpt=$(ls -t "${WORK_DIR}"/outputs/*/*/checkpoints/*.ckpt 2>/dev/null | head -1)
        if [[ -z "${new_ckpt}" || ! -f "${new_ckpt}" ]]; then
            err "找不到 ckpt 可以 resume，放弃"
            break
        fi
        info "从最新 ckpt 恢复: ${new_ckpt}"
        run_resume_arg=( "checkpointing.load=${new_ckpt}" )
    done

    stop_gpu_monitor
    return ${rc}
}

# ============================ 入口 ============================
t0=$(date +%s)
main_train
dt=$(( $(date +%s) - t0 ))
log "总耗时: $((dt/3600))h $((dt%3600/60))m $((dt%60))s"

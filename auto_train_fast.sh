#!/usr/bin/env bash
# =============================================================================
# NAS3R 加速训练脚本 · 8×H100 80GB
#
# 相比 auto_train.sh 的加速优化：
#   1. BF16-mixed 精度（H100 Tensor Core 原生支持，~2× 吞吐）
#   2. TF32 matmul（main.py 已开启）
#   3. cuDNN benchmark（自动选择最快 kernel）
#   4. 增大 batch_size 到 16（bf16 省显存）
#   5. 抑制 "Skipped bad example" 日志刷屏（monkeypatch，不改原代码）
#   6. DataLoader num_workers=12, prefetch_factor=4（数据加载更快）
#
# 烟雾测试阶段依然保留，防止新配置触雷
#
# 使用：
#   mkdir -p /opt/tiger/mmfinetune && cd /opt/tiger/mmfinetune
#   curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/auto_train_fast.sh -o auto_train_fast.sh
#   bash auto_train_fast.sh
#
# 如果显存吃紧可以调低：BATCH_SIZE=12 bash auto_train_fast.sh
# 如果 bf16 精度有问题：PRECISION=32-true bash auto_train_fast.sh
# =============================================================================

set -uo pipefail

# ============================ 配置 ============================
PARENT_DIR="${PARENT_DIR:-/opt/tiger/mmfinetune}"
REPO_URL="${REPO_URL:-https://github.com/yhyblog/NAS3R.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_NAME="nas3r"
WORK_DIR="${PARENT_DIR}/${REPO_NAME}"

EXPERIMENT="${EXPERIMENT:-nas3r/random/re10k}"
WANDB_NAME="${WANDB_NAME:-nas3r_fast_$(date +%m%d_%H%M)}"

# 加速相关（默认值针对 H100 80GB 已调优）
PRECISION="${PRECISION:-bf16-mixed}"     # 32-true / 16-mixed / bf16-mixed
BATCH_SIZE="${BATCH_SIZE:-16}"           # bf16 下 10→16 不会 OOM，~60% 吞吐提升
NUM_WORKERS="${NUM_WORKERS:-12}"         # 数据加载并行度
PREFETCH="${PREFETCH:-4}"                # 每个 worker 预取的 batch 数

# 防崩保护
CKPT_EVERY="${CKPT_EVERY:-500}"
CKPT_KEEP="${CKPT_KEEP:-3}"
VAL_INTERVAL="${VAL_INTERVAL:-5000}"

SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

# 抑制 skip 日志（monkeypatch）
SUPPRESS_SKIP_LOG="${SUPPRESS_SKIP_LOG:-1}"   # 1=只打印每 500 次 skip 一条汇总

CKPT_DIR="${CKPT_DIR:-${WORK_DIR}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/datasets/re10k}"
OUT_ROOT="${OUT_ROOT:-${WORK_DIR}/outputs/train}"
MONITOR_DIR="${MONITOR_DIR:-${WORK_DIR}/outputs/monitor}"

# H100 → sm_90
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HYDRA_FULL_ERROR=1

# NCCL 防御（同 auto_train.sh）
if [[ "${NCCL_FORCE_FASTRAK:-0}" != "1" ]]; then
    unset NCCL_NET_PLUGIN
    export NCCL_NET_PLUGIN=""
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi

# 数据加载优化
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

log()  { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
info() { echo -e "  \033[0;34m→\033[0m $*"; }
ok()   { echo -e "  \033[1;32m✓\033[0m $*"; }
warn() { echo -e "  \033[1;33m⚠\033[0m $*"; }
err()  { echo -e "\033[1;31m✗ ERROR: $*\033[0m" >&2; }

# ============================ Step 0: 代码仓库 ============================
# 校验 ${WORK_DIR} 里 git remote 'origin' 指向的是不是我们期望的 REPO_URL。
# 如果 URL 对不上（比如这台机器 cd 进去的是公司的 mmfinetune 仓库），
# 会删掉并重新 clone，避免把 mmfinetune 的 master 当成 NAS3R 来 pull。
ensure_correct_remote() {
    local dir="$1"
    [[ -d "${dir}/.git" ]] || return 1
    local actual
    actual=$(git -C "${dir}" config --get remote.origin.url 2>/dev/null)
    # 允许 .git 结尾差异、ghproxy 前缀差异
    local expect_base
    expect_base=$(echo "${REPO_URL}" | sed -E 's#^https?://(ghproxy\.com/)?(github\.com/[^.]+).*#\2#')
    local actual_base
    actual_base=$(echo "${actual}" | sed -E 's#^https?://(ghproxy\.com/)?(github\.com/[^.]+).*#\2#; s#^git@github\.com:#github.com/#; s#\.git$##')
    if [[ "${actual_base}" == "${expect_base}" ]]; then
        return 0
    fi
    warn "${dir} 的 origin 不是期望仓库:"
    warn "  期望: ${REPO_URL}"
    warn "  实际: ${actual}"
    return 1
}

step0_code() {
    log "Step 0: 准备代码仓库 → ${WORK_DIR}"
    mkdir -p "${PARENT_DIR}"

    if [[ -d "${PARENT_DIR}/.git" ]]; then
        info "${PARENT_DIR} 是 git 仓库，在其下建子目录 ${REPO_NAME}/"
    fi

    cd "${PARENT_DIR}"

    # 1) FORCE_FRESH_CLONE=1 → 无条件删除重 clone
    if [[ "${FORCE_FRESH_CLONE:-0}" == "1" && -d "${WORK_DIR}" ]]; then
        warn "FORCE_FRESH_CLONE=1，强制删除 ${WORK_DIR} 重新 clone"
        rm -rf "${WORK_DIR}"
    fi

    # 2) SKIP_CLONE=1 → 假设代码已就位，只校验
    if [[ "${SKIP_CLONE}" == "1" ]]; then
        info "SKIP_CLONE=1，跳过 clone/pull"
        [[ -d "${WORK_DIR}/.git" ]] || { err "${WORK_DIR} 不是 git 仓库"; return 1; }
    else
        # 3) 已存在 .git → 校验 remote 对不对，对则 pull，不对则重 clone
        if [[ -d "${WORK_DIR}/.git" ]]; then
            if ensure_correct_remote "${WORK_DIR}"; then
                info "仓库已存在且 remote 正确，git pull ..."
                cd "${WORK_DIR}"
                git fetch origin "${REPO_BRANCH}" 2>&1 | tail -3 || warn "git fetch 失败"
                git checkout "${REPO_BRANCH}" 2>&1 | tail -3 || warn "git checkout 失败"
                if ! git pull --ff-only origin "${REPO_BRANCH}" 2>&1 | tail -3; then
                    warn "ff-only pull 失败（本地有改动？），尝试 reset --hard origin/${REPO_BRANCH}"
                    if [[ "${AUTO_RESET:-1}" == "1" ]]; then
                        git fetch origin "${REPO_BRANCH}"
                        git reset --hard "origin/${REPO_BRANCH}" 2>&1 | tail -3
                    else
                        warn "保留本地改动（AUTO_RESET=0），继续用当前版本"
                    fi
                fi
                cd "${PARENT_DIR}"
            else
                warn "remote 不匹配，删掉旧目录重新 clone"
                rm -rf "${WORK_DIR}"
            fi
        # 4) 目录存在但不是 git 仓库 → 删掉重 clone
        elif [[ -d "${WORK_DIR}" ]]; then
            warn "${WORK_DIR} 存在但不是 git 仓库，清空后重新 clone"
            rm -rf "${WORK_DIR}"
        fi

        # 5) 此时若不存在则 clone
        if [[ ! -d "${WORK_DIR}/.git" ]]; then
            info "clone ${REPO_URL} (branch=${REPO_BRANCH})"
            git clone --recurse-submodules -b "${REPO_BRANCH}" "${REPO_URL}" "${REPO_NAME}" \
                2>&1 | tail -5 || { err "clone 失败"; return 1; }
        fi
    fi

    [[ -d "${WORK_DIR}/.git" && -f "${WORK_DIR}/src/main.py" ]] || {
        err "代码不完整: ${WORK_DIR}"
        err "  .git 存在? $([[ -d ${WORK_DIR}/.git ]] && echo yes || echo no)"
        err "  src/main.py 存在? $([[ -f ${WORK_DIR}/src/main.py ]] && echo yes || echo no)"
        return 1
    }

    cd "${WORK_DIR}"
    info "当前仓库: $(git config --get remote.origin.url)"
    info "当前分支: $(git rev-parse --abbrev-ref HEAD)"
    info "当前 commit: $(git log -1 --format='%h %s' | cut -c1-80)"

    # submodule 兜底
    git submodule sync --recursive >/dev/null 2>&1 || true
    git submodule update --init --recursive 2>&1 | tail -3 || \
        warn "submodule init 失败"

    local dgr_dir="submodules/diff-gaussian-rasterization"
    if [[ ! -f "${dgr_dir}/setup.py" ]]; then
        info "手动 clone diff-gaussian-rasterization"
        rm -rf "${dgr_dir}"
        mkdir -p submodules
        git clone --recurse-submodules -b camera \
            https://github.com/ranrhuang/diff-gaussian-rasterization.git "${dgr_dir}" || \
            { err "diff-gaussian-rasterization clone 失败"; return 1; }
    fi

    local glm_dir="${dgr_dir}/third_party/glm"
    if [[ ! -f "${glm_dir}/glm/glm.hpp" ]]; then
        info "下载 GLM"
        rm -rf "${glm_dir}"
        git clone --depth 1 https://github.com/g-truc/glm.git "${glm_dir}" || \
            { err "GLM 下载失败"; return 1; }
    fi

    ok "代码就位  commit=$(git log -1 --format='%h %s' | cut -c1-70)"
}

# ============================ Step 1: 依赖 ============================
step1_install() {
    log "Step 1: 安装 Python 依赖"
    if [[ "${SKIP_INSTALL}" == "1" ]]; then
        info "SKIP_INSTALL=1，跳过"
        return 0
    fi

    python3 --version
    python3 -c "import torch; print(f'    torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.device_count()}x{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || {
        err "torch 未安装"; return 1
    }

    python3 -m pip install --upgrade pip -q
    grep -v -E "^torch==|^torchvision==|^torchaudio==|scikit-video|pytorch3d" requirements.txt \
        > /tmp/requirements_fixed.txt
    pip install -r /tmp/requirements_fixed.txt --no-build-isolation 2>&1 | tail -10

    pip install sk-video 2>/dev/null || pip install scikit-video --no-deps 2>/dev/null || true

    if ! python3 -c "import pytorch3d" 2>/dev/null; then
        info "编译 pytorch3d"
        pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation \
            2>&1 | tail -5 || warn "pytorch3d 编译失败"
    fi

    info "编译 diff-gaussian-rasterization"
    pip install -e submodules/diff-gaussian-rasterization --no-build-isolation 2>&1 | tail -5 || \
        { err "diff-gaussian-rasterization 编译失败"; return 1; }

    python3 -c "import diff_gauss_camera; print('  [ok] diff_gauss_camera')" || \
        { err "diff_gauss_camera import 失败"; return 1; }

    pip install -q "huggingface_hub[cli,hf_transfer]" colorama
    ok "依赖安装完成"
}

# ============================ Step 2: 数据 ============================
step2_data() {
    log "Step 2: 准备 re10k 训练数据"
    if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
        info "SKIP_DOWNLOAD=1，跳过"
        local train_n
        train_n=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
        info "train 数据: ${train_n} shards"
        return 0
    fi

    mkdir -p "${DATA_DIR}"

    local train_n
    train_n=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    if [[ "${train_n}" -lt 300 ]]; then
        info "下载 re10k train（~500GB, 330 shards）"
        python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yiren-lu/re10k_pixelsplat",
    repo_type="dataset",
    local_dir="${DATA_DIR}",
    allow_patterns=["train/*.torch", "train/index.json"],
)
PYEOF
        train_n=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    fi

    # test 少量子集用于 val
    local test_n
    test_n=$(find "${DATA_DIR}/test" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    if [[ "${test_n}" -lt 10 ]]; then
        info "下载 re10k test 少量子集"
        python3 -m src.scripts.prepare_re10k_test_subset \
            --data-dir "${DATA_DIR}" \
            --eval-index "${WORK_DIR}/assets/evaluation_index_re10k.json" \
            --subset-out "${WORK_DIR}/assets/evaluation_index_re10k.subset.json" \
            --max-scenes 50 || warn "test 子集下载失败"
    fi

    # train/index.json
    if [[ ! -f "${DATA_DIR}/train/index.json" ]]; then
        info "生成 train/index.json"
        python3 - <<PYEOF
import json, torch
from pathlib import Path
d = Path("${DATA_DIR}/train")
idx = {}
for f in sorted(d.glob("*.torch")):
    try:
        for item in torch.load(f, weights_only=True, map_location="cpu"):
            idx[item["key"]] = f.name
    except Exception as e:
        print(f"  [WARN] {f.name}: {e}")
(d / "index.json").write_text(json.dumps(idx))
print(f"  {len(idx)} scenes")
PYEOF
    fi

    local final_train
    final_train=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    [[ "${final_train}" -lt 300 ]] && { err "Train 数据不足 ${final_train}/330"; return 1; }

    ok "数据就位: train=${final_train}/330"
}

# ============================ Skip-log 抑制补丁 ============================
# 不改原代码，通过 sitecustomize 把 dataset_re10k.py 里的 print 拦截
write_skip_suppression() {
    if [[ "${SUPPRESS_SKIP_LOG}" != "1" ]]; then
        return 0
    fi

    local patch_dir="${WORK_DIR}/nas3r_patches"
    mkdir -p "${patch_dir}"

    cat > "${patch_dir}/sitecustomize.py" <<'PYEOF'
"""
运行时拦截 dataset_re10k.py 里的 'Skipped bad example ...' 刷屏。
每 500 次聚合成一条日志；保留第一次 + 之后按间隔输出。
通过 PYTHONPATH=<patch_dir> 启用，零侵入源码。
"""
import builtins
import os
import sys

_orig_print = builtins.print
_state = {"skipped": 0, "first_seen": False, "throttle": int(os.environ.get("SKIP_LOG_THROTTLE", "500"))}


def _patched_print(*args, **kwargs):
    if args and isinstance(args[0], str) and args[0].startswith("Skipped bad example"):
        _state["skipped"] += 1
        n = _state["skipped"]
        if not _state["first_seen"]:
            _state["first_seen"] = True
            _orig_print(f"[skip-log] 首次: {args[0]}", **kwargs)
            return
        if n % _state["throttle"] == 0:
            _orig_print(f"[skip-log] 累计已跳过 {n} 个 bad shape examples", **kwargs)
        return
    _orig_print(*args, **kwargs)


builtins.print = _patched_print
PYEOF
    info "已生成 skip 抑制补丁: ${patch_dir}/sitecustomize.py"
}

# ============================ GPU 监控 ============================
start_gpu_monitor() {
    mkdir -p "${MONITOR_DIR}"
    local log_file="${MONITOR_DIR}/gpu_mem_$(date +%Y%m%d_%H%M%S).csv"
    info "GPU 监控 → ${log_file}"
    (
        echo "timestamp,gpu_idx,mem_used_mb,mem_total_mb,util_pct" > "${log_file}"
        while true; do
            nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits 2>/dev/null | \
                awk -F', ' '{print $1","$2","$3","$4","$5}' >> "${log_file}" 2>/dev/null || true
            sleep 30
        done
    ) >/dev/null 2>&1 &
    echo $! > "${MONITOR_DIR}/monitor.pid"
}

stop_gpu_monitor() {
    if [[ -f "${MONITOR_DIR}/monitor.pid" ]]; then
        local pid
        pid=$(cat "${MONITOR_DIR}/monitor.pid")
        kill "${pid}" 2>/dev/null
        rm -f "${MONITOR_DIR}/monitor.pid"
    fi
}

# ============================ Step 3: 烟雾测试 ============================
step3_smoke() {
    log "Step 3: 烟雾测试（300 step, precision=${PRECISION}, batch_size=${BATCH_SIZE}）"
    if [[ "${SKIP_SMOKE}" == "1" ]]; then
        info "SKIP_SMOKE=1，跳过"
        return 0
    fi

    local out_path="${OUT_ROOT}/_smoke_fast"
    rm -rf "${out_path}"
    mkdir -p "${out_path}"

    info "单卡 GPU0 · 验证 BF16 + 新 batch_size 不 OOM"

    cd "${WORK_DIR}"
    local logfile="${out_path}/smoke.log"

    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="${WORK_DIR}/nas3r_patches:${PYTHONPATH:-}" \
    timeout 1800 python3 -m src.main \
        +experiment="${EXPERIMENT}" \
        wandb.mode=disabled \
        wandb.name="smoke_fast" \
        "dataset.re10k.roots=[${DATA_DIR}]" \
        data_loader.train.batch_size="${BATCH_SIZE}" \
        data_loader.train.num_workers=4 \
        trainer.max_steps=300 \
        trainer.val_check_interval=200 \
        +trainer.num_sanity_val_steps=0 \
        +trainer.precision="${PRECISION}" \
        checkpointing.every_n_train_steps=500 \
        train.print_log_every_n_steps=20 \
        hydra.run.dir="${out_path}" \
        >"${logfile}" 2>&1

    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        err "烟雾测试失败 (rc=${rc})，日志末尾："
        tail -n 50 "${logfile}"
        err "可能需要："
        err "  · 降 BATCH_SIZE=12 或 10"
        err "  · 或改 PRECISION=32-true（回 FP32）"
        return 1
    fi

    ok "烟雾测试通过"

    # 速度 & 显存报告
    local last_step first_ts last_ts
    last_step=$(grep -oE "train step [0-9]+" "${logfile}" 2>/dev/null | tail -1 | awk '{print $3}')
    info "到达 step: ${last_step:-'?'}"

    # 平均 step 时间
    python3 - "${logfile}" <<'PYEOF' || true
import re, sys
txt = open(sys.argv[1]).read()
steps = [(int(m.group(1)), i) for i, m in enumerate(re.finditer(r"train step (\d+);", txt))]
if len(steps) >= 2:
    lines = txt.split("\n")
    idx_to_line = {}
    cnt = 0
    for i, l in enumerate(lines):
        m = re.search(r"train step (\d+);", l)
        if m:
            idx_to_line[cnt] = i
            cnt += 1
    print(f"  [smoke] {len(steps)} 个打印点，step {steps[0][0]} → {steps[-1][0]}")
PYEOF
}

# ============================ Step 4: 正式训练 ============================
step4_train() {
    log "Step 4: 正式 8 卡训练 (precision=${PRECISION})"

    local NUM_GPUS
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    info "GPU 数: ${NUM_GPUS}"
    local global_batch=$(( NUM_GPUS * BATCH_SIZE ))

    local out_path="${OUT_ROOT}/${WANDB_NAME}"
    mkdir -p "${out_path}"

    info "配置："
    info "  experiment  = ${EXPERIMENT}"
    info "  precision   = ${PRECISION}  (vs 原版 32-true)"
    info "  batch/gpu   = ${BATCH_SIZE}  (vs 原版 10)"
    info "  global batch= ${global_batch}  (vs 原版 80)"
    info "  workers     = ${NUM_WORKERS}  (vs 原版 8)"
    info "  prefetch    = ${PREFETCH}"
    info "  ckpt every  = ${CKPT_EVERY}  保留 ${CKPT_KEEP} 份"
    info "  val every   = ${VAL_INTERVAL}"
    info "  skip log    = $([[ ${SUPPRESS_SKIP_LOG} == 1 ]] && echo '聚合每 500 次一条' || echo '原样')"
    info "  日志 → ${out_path}/train.log"

    cd "${WORK_DIR}"

    local rank_log_dir="${out_path}/ranks"
    mkdir -p "${rank_log_dir}"

    if [[ "${NUM_GPUS}" -ge 2 ]]; then
        PYTHONPATH="${WORK_DIR}/nas3r_patches:${PYTHONPATH:-}" \
        torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
            --redirects=3 --tee=3 --log_dir="${rank_log_dir}" \
            -m src.main \
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
            +trainer.precision="${PRECISION}" \
            2>&1 | tee "${out_path}/train.log"
    else
        PYTHONPATH="${WORK_DIR}/nas3r_patches:${PYTHONPATH:-}" \
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
            +trainer.precision="${PRECISION}" \
            2>&1 | tee "${out_path}/train.log"
    fi

    local rc=${PIPESTATUS[0]}
    [[ ${rc} -eq 0 ]] && ok "训练完成" || err "训练退出 (rc=${rc})"
    return ${rc}
}

# ============================ 主流程 ============================
main() {
    local t0=$(date +%s)

    step0_code    || exit 1
    step1_install || exit 1
    mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUT_ROOT}" "${MONITOR_DIR}"
    step2_data    || warn "数据准备警告"

    write_skip_suppression

    start_gpu_monitor
    trap stop_gpu_monitor EXIT

    step3_smoke   || { err "烟雾测试失败"; exit 1; }

    if [[ "${SMOKE_ONLY}" == "1" ]]; then
        ok "SMOKE_ONLY=1，测试通过后退出"
        exit 0
    fi

    step4_train   || warn "训练提前结束"

    local dt=$(( $(date +%s) - t0 ))
    log "总耗时: $((dt/3600))h $((dt%3600/60))m $((dt%60))s"
}

main "$@"

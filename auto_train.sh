#!/usr/bin/env bash
# =============================================================================
# NAS3R 自动训练脚本 · 8×H100 80GB 单节点
#
# 功能：
#   Step 0: clone / pull 代码 + 手动兜底 submodule + GLM 头文件
#   Step 1: pip 依赖 + pytorch3d + diff-gaussian-rasterization 编译
#   Step 2: 下载 train 数据（re10k 全量 training set）+ test 数据
#   Step 3: 烟雾测试（300 step，验证 LPIPS 修复 + 显存稳定）
#   Step 4: 正式训练（8 卡 DDP，按 README 设置）
#
# 使用：
#   # 新机器一键完整流程
#   mkdir -p /opt/tiger/mmfinetune && cd /opt/tiger/mmfinetune
#   curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/auto_train.sh -o auto_train.sh
#   bash auto_train.sh
#
# 环境变量：
#   PARENT_DIR=/opt/tiger/mmfinetune   仓库父目录
#   EXPERIMENT=nas3r/random/re10k      训练配置（也可以是 nas3r/pretrained/re10k 等）
#   WANDB_NAME=nas3r_re10k             wandb run 名
#   BATCH_SIZE=10                      per-GPU batch_size
#   SKIP_CLONE=0 / SKIP_INSTALL=0 / SKIP_DOWNLOAD=0 / SKIP_SMOKE=0
#   SMOKE_ONLY=0                       1 则只跑烟雾测试不进入正式训练
# =============================================================================

set -uo pipefail

# ============================ 配置 ============================
PARENT_DIR="${PARENT_DIR:-/opt/tiger/mmfinetune}"
REPO_URL="${REPO_URL:-https://github.com/yhyblog/NAS3R.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_NAME="nas3r"
WORK_DIR="${PARENT_DIR}/${REPO_NAME}"

EXPERIMENT="${EXPERIMENT:-nas3r/random/re10k}"
WANDB_NAME="${WANDB_NAME:-nas3r_re10k_$(date +%m%d_%H%M)}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

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
# 抗显存碎片（验证多轮后的关键）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ============================ NCCL 防御 ============================
# 问题现象：GCP/字节训练镜像默认设 NCCL_NET_PLUGIN=libnccl-net-gcp-fastrak.so
# 该插件只支持 NCCL v7 接口，而镜像里 NCCL 2.26 需要 v8-v10 → DDP 初始化 OK
# 但真正做 collective 时 SIGSEGV，进程无 traceback 消失。
#
# 修复：默认用 socket 后端（最稳，H100 NVLink 内部带宽仍然够用）。
# 如果你的镜像确实有配对的 FasTrak v8+，可以 NCCL_FORCE_FASTRAK=1 绕开这段。
if [[ "${NCCL_FORCE_FASTRAK:-0}" != "1" ]]; then
    # 清空可能存在的不兼容 plugin
    unset NCCL_NET_PLUGIN
    export NCCL_NET_PLUGIN=""
    # 回落到 socket
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"    # NVLink 保留
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    # 有故障时看得清楚
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
fi

# ============================ NCCL 偶发卡死对策 ============================
# 现象：纯训练 DDP 跑 10-13 小时后偶发某个 AllReduce 不返回（NumelIn=2189907
# 的梯度 bucket），8 个 rank 全卡在同一个 work id，1800s 超时后进程崩溃。
# 这不是代码 bug，是字节/GCP 镜像里 NCCL 在 socket 后端上偶发的通信卡死。
#
# 对策：
# 1) 打开 ASYNC_ERROR_HANDLING，让 collective 出错时立刻抛异常（而不是等 watchdog）
# 2) 降低 HEARTBEAT_TIMEOUT：从默认 600s 降到 300s，卡 5 分钟就终止，尽快暴露
# 3) TORCH_NCCL_BLOCKING_WAIT=0 + TORCH_NCCL_ASYNC_ERROR_HANDLING=1：非阻塞但能抛错
# 4) flight recorder 开着，下次崩能抓到真正卡住的 collective 调用栈
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-300}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}"
# AllReduce 超时：从默认 1800s 降到 600s（10 分钟没响应就认定挂死）
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"
# 额外稳定性：关闭 NCCL socket 侧的 retry，出错就直接抛
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-2}"

log()  { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
info() { echo -e "  \033[0;34m→\033[0m $*"; }
ok()   { echo -e "  \033[1;32m✓\033[0m $*"; }
warn() { echo -e "  \033[1;33m⚠\033[0m $*"; }
err()  { echo -e "\033[1;31m✗ ERROR: $*\033[0m" >&2; }

# ============================ Step 0: 代码仓库 ============================
step0_code() {
    log "Step 0: 准备代码仓库 → ${WORK_DIR}"
    mkdir -p "${PARENT_DIR}"

    # 前置检查：PARENT_DIR 本身可能是公司的 mmfinetune 仓库
    if [[ -d "${PARENT_DIR}/.git" ]]; then
        info "注意：${PARENT_DIR} 本身是 git 仓库（应该是公司的 mmfinetune 仓库）"
        info "      将在其下建子目录 ${REPO_NAME}/，nas3r 代码独立管理"
    fi

    cd "${PARENT_DIR}"

    if [[ "${SKIP_CLONE}" == "1" ]]; then
        info "SKIP_CLONE=1，跳过 clone"
        [[ -d "${WORK_DIR}/.git" ]] || { err "${WORK_DIR} 不是 git 仓库"; return 1; }
    elif [[ -d "${WORK_DIR}/.git" ]]; then
        info "仓库已存在，git pull"
        cd "${WORK_DIR}"
        git fetch origin "${REPO_BRANCH}" >/dev/null 2>&1 || true
        git checkout "${REPO_BRANCH}" >/dev/null 2>&1 || true
        git pull --ff-only origin "${REPO_BRANCH}" 2>&1 | tail -3 || \
            warn "git pull 失败，继续用当前版本"
    elif [[ -d "${WORK_DIR}" ]]; then
        # 目录存在但不是 git 仓库（可能是残留）
        warn "${WORK_DIR} 已存在但不是 git 仓库，清空后重新 clone"
        rm -rf "${WORK_DIR}"
        info "clone ${REPO_URL} → ${WORK_DIR}"
        git clone --recurse-submodules -b "${REPO_BRANCH}" "${REPO_URL}" "${REPO_NAME}" || \
            { err "代码 clone 失败"; return 1; }
    else
        info "clone ${REPO_URL} → ${WORK_DIR}"
        git clone --recurse-submodules -b "${REPO_BRANCH}" "${REPO_URL}" "${REPO_NAME}" || \
            { err "代码 clone 失败"; return 1; }
    fi

    # 确认 clone 成功
    [[ -d "${WORK_DIR}/.git" && -f "${WORK_DIR}/src/main.py" ]] || {
        err "代码 clone 后仍不完整: ${WORK_DIR}"
        err "  .git 存在? $([[ -d ${WORK_DIR}/.git ]] && echo yes || echo no)"
        err "  src/main.py 存在? $([[ -f ${WORK_DIR}/src/main.py ]] && echo yes || echo no)"
        return 1
    }

    cd "${WORK_DIR}"
    info "当前 commit: $(git log -1 --oneline)"

    # submodule 兜底
    info "初始化 submodules"
    git submodule sync --recursive >/dev/null 2>&1 || true
    git submodule update --init --recursive 2>&1 | tail -5 || \
        warn "git submodule update 失败，进入手动兜底"

    local dgr_dir="submodules/diff-gaussian-rasterization"
    local dgr_url="https://github.com/ranrhuang/diff-gaussian-rasterization.git"
    if [[ ! -f "${dgr_dir}/setup.py" ]]; then
        info "手动 clone diff-gaussian-rasterization"
        rm -rf "${dgr_dir}"
        mkdir -p submodules
        git clone --recurse-submodules -b camera "${dgr_url}" "${dgr_dir}" || \
            { err "diff-gaussian-rasterization clone 失败"; return 1; }
    fi

    local glm_dir="${dgr_dir}/third_party/glm"
    if [[ ! -f "${glm_dir}/glm/glm.hpp" ]]; then
        info "下载 GLM 头文件"
        rm -rf "${glm_dir}"
        git clone --depth 1 https://github.com/g-truc/glm.git "${glm_dir}" || \
            { err "GLM 下载失败"; return 1; }
    fi

    ok "代码就位  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
}

# ============================ Step 1: 依赖 ============================
step1_install() {
    log "Step 1: 安装 Python 依赖"
    if [[ "${SKIP_INSTALL}" == "1" ]]; then
        info "SKIP_INSTALL=1，跳过"
        return 0
    fi

    info "环境："
    python3 --version
    python3 -c "import torch; print(f'    torch={torch.__version__}  cuda={torch.version.cuda}  dev={torch.cuda.device_count()}x{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || {
        err "torch 未安装"
        return 1
    }

    python3 -m pip install --upgrade pip -q
    grep -v -E "^torch==|^torchvision==|^torchaudio==|scikit-video|pytorch3d" requirements.txt \
        > /tmp/requirements_fixed.txt
    pip install -r /tmp/requirements_fixed.txt --no-build-isolation 2>&1 | tail -10

    pip install sk-video 2>/dev/null || pip install scikit-video --no-deps 2>/dev/null || true

    if ! python3 -c "import pytorch3d" 2>/dev/null; then
        info "编译 pytorch3d (arch=${TORCH_CUDA_ARCH_LIST})"
        pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation \
            2>&1 | tail -5 || warn "pytorch3d 编译失败"
    else
        info "pytorch3d 已安装"
    fi

    info "编译 diff-gaussian-rasterization"
    pip install -e submodules/diff-gaussian-rasterization --no-build-isolation 2>&1 | tail -10 || \
        { err "diff-gaussian-rasterization 编译失败"; return 1; }

    python3 -c "import diff_gauss_camera; print('  [ok] diff_gauss_camera import 成功')" || \
        { err "diff_gauss_camera import 失败"; return 1; }

    pip install -q "huggingface_hub[cli,hf_transfer]" colorama
    ok "依赖安装完成"
}

# ============================ Step 2: 下载训练数据 ============================
step2_data() {
    log "Step 2: 下载 re10k 训练数据"
    if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
        info "SKIP_DOWNLOAD=1，跳过"
        # 验证
        local train_n test_n
        train_n=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
        test_n=$(find "${DATA_DIR}/test" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
        info "数据现状: train=${train_n} test=${test_n}"
        return 0
    fi

    mkdir -p "${DATA_DIR}"

    # 检查已有数据
    local train_n test_n
    train_n=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    test_n=$(find "${DATA_DIR}/test" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)

    # re10k_pixelsplat 的 train 共 330 个 shard
    if [[ "${train_n}" -gt 300 ]]; then
        info "Train 数据已就绪 (${train_n}/330 shards)"
    else
        info "下载 re10k train shard（全量 ~575GB，只含 train split，耗时视带宽）"
        # 全量 train，test 只要评测需要的子集（由 prepare_re10k_test_subset 处理）
        python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yiren-lu/re10k_pixelsplat",
    repo_type="dataset",
    local_dir="${DATA_DIR}",
    allow_patterns=["train/*.torch", "train/index.json"],
)
print("  train snapshot done")
PYEOF
        train_n=$(find "${DATA_DIR}/train" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    fi

    # test 集用评测子集准备脚本
    if [[ "${test_n}" -lt 10 ]]; then
        info "下载 re10k test 子集（用于 validation）"
        python3 -m src.scripts.prepare_re10k_test_subset \
            --data-dir "${DATA_DIR}" \
            --eval-index "${WORK_DIR}/assets/evaluation_index_re10k.json" \
            --subset-out "${WORK_DIR}/assets/evaluation_index_re10k.subset.json" \
            --max-scenes 50 || \
            warn "test 子集下载失败，validation 可能缺数据"
        test_n=$(find "${DATA_DIR}/test" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    else
        info "Test 数据已存在 (${test_n} shards)"
    fi

    # 生成 train index.json（如缺失）
    if [[ ! -f "${DATA_DIR}/train/index.json" ]]; then
        info "生成 train/index.json"
        python3 - <<PYEOF
import json, os, torch
from pathlib import Path

d = Path("${DATA_DIR}/train")
idx = {}
for f in sorted(d.glob("*.torch")):
    try:
        chunk = torch.load(f, weights_only=True, map_location="cpu")
        for item in chunk:
            idx[item["key"]] = f.name
    except Exception as e:
        print(f"  [WARN] {f.name}: {e}")
(d / "index.json").write_text(json.dumps(idx))
print(f"  train/index.json: {len(idx)} scenes")
PYEOF
    fi

    ok "数据就位: train=${train_n}, test=${test_n}"
    df -h "${DATA_DIR}" 2>/dev/null | tail -1 | awk '{print "    磁盘: "$3"/"$2" ("$5" used)"}'
}

# ============================ 显存监控（后台启动） ============================
start_gpu_monitor() {
    mkdir -p "${MONITOR_DIR}"
    local log_file="${MONITOR_DIR}/gpu_mem_$(date +%Y%m%d_%H%M%S).csv"
    info "启动 GPU 监控 → ${log_file}"
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
    ok "监控 PID: $(cat ${MONITOR_DIR}/monitor.pid)"
}

stop_gpu_monitor() {
    if [[ -f "${MONITOR_DIR}/monitor.pid" ]]; then
        local pid
        pid=$(cat "${MONITOR_DIR}/monitor.pid")
        kill "${pid}" 2>/dev/null && info "已停止 GPU 监控 (pid=${pid})"
        rm -f "${MONITOR_DIR}/monitor.pid"
    fi
}

# ============================ Step 3: 烟雾测试 ============================
step3_smoke() {
    log "Step 3: 烟雾测试（300 step，验证 OOM 修复）"
    if [[ "${SKIP_SMOKE}" == "1" ]]; then
        info "SKIP_SMOKE=1，跳过"
        return 0
    fi

    local out_path="${OUT_ROOT}/_smoke"
    rm -rf "${out_path}"
    mkdir -p "${out_path}"

    info "配置：单卡 GPU0 · batch_size=4 · max_steps=300 · val_check_interval=100"
    info "  预期：第 100 step 首次 val 前后，单卡显存增长 < 500MB（修复后）"

    cd "${WORK_DIR}"

    CUDA_VISIBLE_DEVICES=0 \
    timeout 1800 python3 -m src.main \
        +experiment="${EXPERIMENT}" \
        wandb.mode=disabled \
        wandb.name="smoke_${WANDB_NAME}" \
        "dataset.re10k.roots=[${DATA_DIR}]" \
        data_loader.train.batch_size=4 \
        data_loader.train.num_workers=4 \
        trainer.max_steps=300 \
        trainer.val_check_interval=100 \
        +trainer.num_sanity_val_steps=0 \
        checkpointing.every_n_train_steps=500 \
        train.print_log_every_n_steps=20 \
        hydra.run.dir="${out_path}" \
        2>&1 | tee "${out_path}/smoke.log"

    local rc=${PIPESTATUS[0]}

    if [[ ${rc} -ne 0 ]]; then
        err "烟雾测试失败 (rc=${rc})，日志末尾 40 行："
        echo "------------------------------------------------"
        tail -n 40 "${out_path}/smoke.log"
        echo "------------------------------------------------"
        err "请先修复上述错误再正式训练。"
        return 1
    fi

    # 简单解析 step 数
    local last_step
    last_step=$(grep -oE "step=[0-9]+" "${out_path}/smoke.log" 2>/dev/null | tail -1 | cut -d= -f2)
    [[ -z "${last_step}" ]] && last_step=$(grep -oE "global_step=[0-9]+" "${out_path}/smoke.log" 2>/dev/null | tail -1 | cut -d= -f2)

    ok "烟雾测试通过（last step=${last_step:-'?'}）"

    # 显存增长报告
    if [[ -d "${MONITOR_DIR}" ]]; then
        local latest_csv
        latest_csv=$(ls -t "${MONITOR_DIR}"/gpu_mem_*.csv 2>/dev/null | head -1)
        if [[ -n "${latest_csv}" ]]; then
            info "GPU0 显存增长报告："
            python3 - "${latest_csv}" <<'PYEOF'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
rows = [r for r in rows if r['gpu_idx'] == '0']
if len(rows) >= 2:
    first = int(rows[0]['mem_used_mb'])
    last = int(rows[-1]['mem_used_mb'])
    peak = max(int(r['mem_used_mb']) for r in rows)
    print(f"    初始 {first} MB → 末 {last} MB → 峰值 {peak} MB  (增长 {last-first:+d} MB)")
    if last - first > 2000:
        print("    ⚠ 增长 >2GB，可能仍有泄漏")
    else:
        print("    ✓ 显存增长正常")
PYEOF
        fi
    fi
}

# ============================ Step 4: 正式训练 ============================
step4_train() {
    log "Step 4: 正式 8 卡训练"

    local NUM_GPUS
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    info "可用 GPU 数: ${NUM_GPUS}"

    local out_path="${OUT_ROOT}/${WANDB_NAME}"
    mkdir -p "${out_path}"

    # 防崩 checkpoint 策略（可通过环境变量覆盖）
    local CKPT_EVERY="${CKPT_EVERY:-500}"   # 每 500 step 存一次
    local CKPT_KEEP="${CKPT_KEEP:-3}"       # 保留最新 3 份
    local VAL_INTERVAL="${VAL_INTERVAL:-5000}"  # 降低 val 频率（原来 10000）
    # 设 DISABLE_VAL=1 可以完全关掉 validation，用于快速抢救。
    if [[ "${DISABLE_VAL:-0}" == "1" ]]; then
        VAL_INTERVAL=999999999
        warn "DISABLE_VAL=1: 本次训练跳过所有 validation（只跑 train + 存 ckpt）"
    fi

    # ============================ 自动 resume ============================
    # 这是之前踩过的坑：cfg.checkpointing.resume=true 本身 不 会 自动扫描
    # checkpoints/，必须显式把 cfg.checkpointing.load 设成 ckpt 路径。
    # 所以这里主动去找最新一份 .ckpt：优先找本次实验目录，否则兜底全局最新。
    #
    # 覆盖方式：
    #   RESUME_CKPT=/path/to/foo.ckpt    手动指定
    #   NO_RESUME=1                      禁用 resume（强制从头训）
    local RESUME_ARG=()
    if [[ "${NO_RESUME:-0}" == "1" ]]; then
        warn "NO_RESUME=1: 从头开始训练（忽略已有 checkpoint）"
    elif [[ -n "${RESUME_CKPT:-}" ]]; then
        if [[ -f "${RESUME_CKPT}" ]]; then
            info "从指定 ckpt resume: ${RESUME_CKPT}"
            RESUME_ARG=( "checkpointing.load=${RESUME_CKPT}" )
        else
            err "RESUME_CKPT 不存在: ${RESUME_CKPT}"
            return 1
        fi
    else
        # 自动找：1) 本次 WANDB_NAME 对应的目录  2) outputs/exp_*/ 下全局最新
        local latest_ckpt=""
        # 先找本次实验目录（严格匹配）
        latest_ckpt=$(find "${WORK_DIR}/outputs/exp_${WANDB_NAME}" -name "*.ckpt" 2>/dev/null | sort -V | tail -1)
        # 没找到 → 找所有 exp_* 目录里最新的一个 ckpt
        if [[ -z "${latest_ckpt}" ]]; then
            latest_ckpt=$(find "${WORK_DIR}/outputs" -path "*/checkpoints/*.ckpt" -printf '%T@ %p\n' 2>/dev/null \
                | sort -n | tail -1 | awk '{print $2}')
        fi
        if [[ -n "${latest_ckpt}" && -f "${latest_ckpt}" ]]; then
            info "自动 resume 从最新 ckpt: ${latest_ckpt}"
            info "  （设 NO_RESUME=1 可强制从头训；或 RESUME_CKPT=/path 指定其他 ckpt）"
            RESUME_ARG=( "checkpointing.load=${latest_ckpt}" )
        else
            info "未找到已有 ckpt，从头开始训练"
        fi
    fi

    info "配置：experiment=${EXPERIMENT} · ${NUM_GPUS}×GPU · batch_size=${BATCH_SIZE}/GPU · global_batch=$((NUM_GPUS * BATCH_SIZE))"
    info "  checkpoint: 每 ${CKPT_EVERY} step 存一次, 保留最新 ${CKPT_KEEP} 份"
    info "  validation: 每 ${VAL_INTERVAL} step 跑一次"
    info "  日志 → ${out_path}/train.log"
    info "  checkpoints → ${WORK_DIR}/outputs/exp_${WANDB_NAME}/checkpoints/"

    cd "${WORK_DIR}"

    # ========== 自动重启循环 ==========
    # 背景：NCCL 在长时间训练（10-13h）后偶发 AllReduce 卡死。既然是偶发的，
    # 就接受它：每次崩溃后，自动找最新 ckpt，重启 torchrun，继续训练。
    # 设 NO_AUTORESTART=1 可禁用（仅跑一次）。
    local MAX_RESTARTS="${MAX_RESTARTS:-20}"
    local attempt=0
    local rc=0
    local run_resume_arg=( "${RESUME_ARG[@]}" )

    while true; do
        attempt=$((attempt + 1))
        info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        info "训练启动第 ${attempt} 次"
        [[ ${#run_resume_arg[@]} -gt 0 ]] && info "  ${run_resume_arg[0]}"
        info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        local per_run_log="${out_path}/train_attempt${attempt}.log"

        if [[ "${NUM_GPUS}" -ge 2 ]]; then
            torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
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
                "${run_resume_arg[@]}" \
                2>&1 | tee "${per_run_log}"
            rc=${PIPESTATUS[0]}
        else
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
                "${run_resume_arg[@]}" \
                2>&1 | tee "${per_run_log}"
            rc=${PIPESTATUS[0]}
        fi

        # 合并日志
        cat "${per_run_log}" >> "${out_path}/train.log"

        # 0 = 正常完成（max_steps 到了），退出循环
        if [[ ${rc} -eq 0 ]]; then
            ok "训练正常完成 (attempt=${attempt})"
            break
        fi

        # 禁用自动重启 → 退出
        if [[ "${NO_AUTORESTART:-0}" == "1" ]]; then
            err "训练退出 (rc=${rc})，NO_AUTORESTART=1 不自动重启"
            break
        fi

        # 到达最大重启次数 → 放弃
        if [[ ${attempt} -ge ${MAX_RESTARTS} ]]; then
            err "训练崩溃 ${attempt} 次已达上限 (MAX_RESTARTS=${MAX_RESTARTS})，放弃"
            break
        fi

        warn "训练崩溃 (rc=${rc})，尝试自动重启（第 $((attempt+1)) 次）..."

        # 清理残留 torchrun/python 进程
        pkill -9 -f "src.main" 2>/dev/null || true
        pkill -9 -f "torchrun" 2>/dev/null || true
        sleep 15  # 等 GPU 彻底释放

        # 找最新 ckpt
        local new_ckpt
        new_ckpt=$(find "${WORK_DIR}/outputs" -path "*/checkpoints/*.ckpt" 2>/dev/null \
            | xargs -I{} stat -c '%Y {}' {} 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
        if [[ -z "${new_ckpt}" || ! -f "${new_ckpt}" ]]; then
            # xargs stat 方案失败，用修改时间的备选
            new_ckpt=$(ls -t "${WORK_DIR}"/outputs/*/*/checkpoints/*.ckpt 2>/dev/null \
                | head -1)
        fi
        if [[ -z "${new_ckpt}" || ! -f "${new_ckpt}" ]]; then
            err "找不到 ckpt 可以 resume，放弃自动重启"
            break
        fi
        info "从最新 ckpt 恢复: ${new_ckpt}"
        run_resume_arg=( "checkpointing.load=${new_ckpt}" )
    done

    [[ ${rc} -eq 0 ]] && ok "训练完成" || err "训练退出 (rc=${rc})"
    return ${rc}
}

# ============================ 主流程 ============================
main() {
    local t0=$(date +%s)

    step0_code     || exit 1
    step1_install  || exit 1
    mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUT_ROOT}" "${MONITOR_DIR}"
    step2_data     || warn "数据准备有警告，继续"

    # 启动 GPU 监控（后台）
    start_gpu_monitor

    # 清理陷阱：退出时停止监控
    trap stop_gpu_monitor EXIT

    step3_smoke    || { err "烟雾测试失败，请先修复再继续"; exit 1; }

    if [[ "${SMOKE_ONLY}" == "1" ]]; then
        ok "SMOKE_ONLY=1，烟雾测试通过后退出"
        exit 0
    fi

    step4_train    || warn "训练提前结束"

    local dt=$(( $(date +%s) - t0 ))
    log "总耗时: $((dt/3600))h $((dt%3600/60))m $((dt%60))s"
}

main "$@"

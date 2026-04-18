#!/usr/bin/env bash
# =============================================================================
# NAS3R 全自动评测脚本 - 8×H100 80GB 单节点
#
# 从零开始做完整流程：
#   Step 0: clone / 同步代码仓库 + 初始化 submodules
#   Step 1: 安装 Python 依赖（复用已装的 torch）
#   Step 2: 并行下载 4 个 checkpoint（~19.6GB）
#   Step 3: 下载评测所需的 re10k test 数据子集
#   Step 4: 4 卡并行评测 4 个 checkpoint（单卡一个，保证指标准确）
#   Step 5: 汇总指标
#
# 使用方法：
#     # 放到任意路径执行，默认会 clone 到 /opt/tiger/mmfinetune/nas3r
#     bash auto_eval.sh
#
#     # 或者直接 curl 下来执行
#     curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/auto_eval.sh \
#          | bash
#
# 可调环境变量：
#   PARENT_DIR=/opt/tiger/mmfinetune   仓库父目录
#   REPO_URL=https://github.com/yhyblog/NAS3R.git
#   REPO_BRANCH=main
#   MAX_EVAL_SCENES=-1       评测 scene 数；-1=全量(5601) / 50=smoke test
#   SAVE_IMAGE=false         是否保存渲染图
#   CKPT_DIR=<repo>/checkpoints
#   DATA_DIR=<repo>/datasets/re10k
#   SKIP_INSTALL=0           1 则跳过 pip install
#   SKIP_CLONE=0             1 则跳过 clone（代码已经在位）
#   ONLY_DOWNLOAD=0          1 则只下载 ckpt+数据，不跑评测
# =============================================================================

set -euo pipefail

# ============================ 基础配置 ============================
PARENT_DIR="${PARENT_DIR:-/opt/tiger/mmfinetune}"
REPO_URL="${REPO_URL:-https://github.com/yhyblog/NAS3R.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_NAME="nas3r"
WORK_DIR="${PARENT_DIR}/${REPO_NAME}"

MAX_EVAL_SCENES="${MAX_EVAL_SCENES:--1}"   # 默认全量
SAVE_IMAGE="${SAVE_IMAGE:-false}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_CLONE="${SKIP_CLONE:-0}"
ONLY_DOWNLOAD="${ONLY_DOWNLOAD:-0}"

CKPT_DIR="${CKPT_DIR:-${WORK_DIR}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/datasets/re10k}"
OUT_ROOT="${OUT_ROOT:-${WORK_DIR}/outputs/eval}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"

log() { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
err() { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; }

# ============================ Step 0: clone / 同步代码 ============================
log "Step 0: 准备代码仓库 → ${WORK_DIR}"

mkdir -p "${PARENT_DIR}"
cd "${PARENT_DIR}"

if [[ "${SKIP_CLONE}" == "1" ]]; then
    log "SKIP_CLONE=1，跳过 clone"
    [[ -d "${WORK_DIR}/.git" ]] || { err "${WORK_DIR} 不是 git 仓库"; exit 1; }
else
    if [[ -d "${WORK_DIR}/.git" ]]; then
        log "仓库已存在，执行 git pull"
        cd "${WORK_DIR}"
        git fetch origin "${REPO_BRANCH}" || true
        git checkout "${REPO_BRANCH}" || true
        git pull --ff-only origin "${REPO_BRANCH}" || warn "git pull 失败（有本地改动？），继续使用当前版本"
    else
        log "clone ${REPO_URL} (branch=${REPO_BRANCH})"
        # 尝试多种 URL 兜底
        git clone --recurse-submodules -b "${REPO_BRANCH}" "${REPO_URL}" "${REPO_NAME}" || \
        git clone --recurse-submodules -b "${REPO_BRANCH}" \
            "https://ghproxy.com/${REPO_URL}" "${REPO_NAME}" || \
        { err "代码 clone 失败，请检查网络"; exit 1; }
    fi
fi

cd "${WORK_DIR}"

# 确保 submodule 就位（带兜底）
log "初始化/更新 submodules"
git submodule sync --recursive || true
git submodule update --init --recursive || warn "git submodule update 失败，稍后会用手动 clone 兜底"

DGR_DIR="submodules/diff-gaussian-rasterization"
DGR_URL="https://github.com/ranrhuang/diff-gaussian-rasterization.git"
if [[ ! -f "${DGR_DIR}/setup.py" ]]; then
    warn "${DGR_DIR} 未就绪，手动 clone ..."
    rm -rf "${DGR_DIR}"
    mkdir -p submodules
    git clone --recurse-submodules -b camera "${DGR_URL}" "${DGR_DIR}" || \
    git clone --recurse-submodules -b camera \
        "https://ghproxy.com/${DGR_URL}" "${DGR_DIR}" || \
    { err "diff-gaussian-rasterization 手动 clone 失败，请检查网络"; exit 1; }
fi

# GLM 头文件兜底
GLM_DIR="${DGR_DIR}/third_party/glm"
if [[ ! -f "${GLM_DIR}/glm/glm.hpp" ]]; then
    warn "GLM 头文件缺失，下载 ..."
    rm -rf "${GLM_DIR}"
    git clone --depth 1 https://github.com/g-truc/glm.git "${GLM_DIR}" || \
    {
        mkdir -p "${GLM_DIR}"
        curl -fsSL https://github.com/g-truc/glm/releases/download/1.0.1/glm-1.0.1-light.zip \
            -o /tmp/glm.zip && \
        unzip -qo /tmp/glm.zip -d "$(dirname "${GLM_DIR}")" && rm -f /tmp/glm.zip
    } || { err "GLM 下载失败"; exit 1; }
fi
echo "  [ok] ${DGR_DIR} 就绪"

# ============================ 全局 CUDA 编译参数（丁老师建议：H100=sm_90）============================
# 为所有后续 CUDA 扩展编译（pytorch3d / diff-gaussian-rasterization）统一指定目标架构。
# 探测失败时强制 9.0（H100）；可通过 TORCH_CUDA_ARCH_LIST 环境变量覆盖。
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    DETECTED_ARCH=$(python3 -c "import torch; c=torch.cuda.get_device_capability(0); print(f'{c[0]}.{c[1]}')" 2>/dev/null || echo "9.0")
    export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH}"
fi
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
echo "  [cuda] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}  MAX_JOBS=${MAX_JOBS}"

# ============================ Step 1: 安装依赖 ============================
if [[ "${SKIP_INSTALL}" != "1" ]]; then
    log "Step 1: 安装 Python 依赖"

    echo "当前环境："
    python3 --version
    python3 -c "import torch; print('  torch :', torch.__version__, '| cuda:', torch.version.cuda)" 2>/dev/null || \
        warn "当前 python 没有 torch，请先安装 torch==2.5.1+cu121 再重跑"

    python3 -m pip install --upgrade pip -q

    # 过滤掉与镜像冲突的包，保留 torch/scikit-video/pytorch3d 的特殊处理
    grep -v -E "^torch==|^torchvision==|^torchaudio==|scikit-video|pytorch3d" requirements.txt \
        > /tmp/requirements_fixed.txt
    pip install -r /tmp/requirements_fixed.txt --no-build-isolation

    # scikit-video 社区 fork（原版不兼容 Py3.11+）
    pip install sk-video 2>/dev/null || pip install scikit-video --no-deps 2>/dev/null || true

    # pytorch3d：源码编译（使用全局 TORCH_CUDA_ARCH_LIST）
    if ! python3 -c "import pytorch3d" 2>/dev/null; then
        log "编译安装 pytorch3d (arch=${TORCH_CUDA_ARCH_LIST})"
        pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation || \
            warn "pytorch3d 编译失败（评测本身不强依赖，可继续）"
    fi

    # diff-gaussian-rasterization：splatting_cuda decoder 必需
    # submodule + GLM 在脚本顶部已确保就位
    log "编译 diff-gaussian-rasterization (arch=${TORCH_CUDA_ARCH_LIST})"
    pip install -e "${DGR_DIR}" --no-build-isolation || \
        { err "diff-gaussian-rasterization 编译失败（评测必需，终止）"; exit 1; }

    pip install -q "huggingface_hub[cli,hf_transfer]" colorama
    log "依赖安装完成 ✓"
else
    log "SKIP_INSTALL=1，跳过依赖安装"
fi

mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUT_ROOT}"

# ============================ Step 2: 并行下载 4 个 checkpoint ============================
log "Step 2: 下载 4 个 checkpoint 到 ${CKPT_DIR}"

CKPTS=(
    re10k_nas3r.ckpt
    re10k_nas3r_multiview.ckpt
    re10k_nas3r_pretrained.ckpt
    re10k_nas3r_pretrained-I.ckpt
)

export HF_HUB_ENABLE_HF_TRANSFER=1   # 多线程高速下载

download_one_ckpt() {
    local fname="$1"
    local dst="${CKPT_DIR}/${fname}"
    if [[ -f "${dst}" ]]; then
        echo "  [skip] ${fname} 已存在"
        return 0
    fi
    echo "  [get ] ${fname}"
    # 优先 huggingface-cli
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download RanranHuang/NAS3R "${fname}" \
            --local-dir "${CKPT_DIR}" --local-dir-use-symlinks False \
            >/dev/null 2>&1
    else
        curl -fL --retry 5 --retry-delay 5 -C - \
            -o "${dst}.part" \
            "https://huggingface.co/RanranHuang/NAS3R/resolve/main/${fname}"
        mv "${dst}.part" "${dst}"
    fi
    echo "  [done] ${fname} ($(du -h "${dst}" | awk '{print $1}'))"
}

CKPT_PIDS=()
for f in "${CKPTS[@]}"; do
    download_one_ckpt "$f" &
    CKPT_PIDS+=("$!")
done

fail=0
for pid in "${CKPT_PIDS[@]}"; do
    wait "${pid}" || fail=$((fail+1))
done
[[ "${fail}" -eq 0 ]] || { err "${fail} 个 checkpoint 下载失败"; exit 1; }

log "所有 checkpoint 就位 ✓"
ls -lh "${CKPT_DIR}"/*.ckpt

# ============================ Step 3: 下载评测数据子集 ============================
log "Step 3: 准备 re10k test 数据（MAX_EVAL_SCENES=${MAX_EVAL_SCENES}）"

SUBSET_EVAL_INDEX="${WORK_DIR}/assets/evaluation_index_re10k.subset.json"

MAX_FLAG=()
if [[ "${MAX_EVAL_SCENES}" != "-1" ]]; then
    MAX_FLAG=(--max-scenes "${MAX_EVAL_SCENES}")
fi

python3 -m src.scripts.prepare_re10k_test_subset \
    --data-dir "${DATA_DIR}" \
    --eval-index "${WORK_DIR}/assets/evaluation_index_re10k.json" \
    --subset-out "${SUBSET_EVAL_INDEX}" \
    "${MAX_FLAG[@]}"

log "数据准备完成 ✓"
df -h "${DATA_DIR}" | head -2

if [[ "${ONLY_DOWNLOAD}" == "1" ]]; then
    log "ONLY_DOWNLOAD=1，退出（不执行评测）"
    exit 0
fi

# ============================ Step 4: 并行评测 4 个 checkpoint ============================
log "Step 4: 4 卡并行评测（每卡一个 checkpoint）"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
echo "  可用 GPU 数: ${NUM_GPUS}"
if [[ "${NUM_GPUS}" -lt 4 ]]; then
    warn "GPU 数量 < 4，部分任务会排队在同一卡上"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -8

# key => "<ckpt_filename>|<experiment>|<extra_hydra_overrides>"
declare -A MODELS=(
    [nas3r]="re10k_nas3r.ckpt|nas3r/random/re10k|"
    [multiview]="re10k_nas3r_multiview.ckpt|nas3r/random/re10k|dataset.re10k.view_sampler.num_context_views=10"
    [pretrained]="re10k_nas3r_pretrained.ckpt|nas3r/pretrained/re10k|"
    [pretrained-I]="re10k_nas3r_pretrained-I.ckpt|nas3r/pretrained/re10k-I|"
)
KEY_ORDER=(nas3r multiview pretrained pretrained-I)
DATA_PARENT="$(dirname "${DATA_DIR}")"

run_eval_job() {
    local key="$1"
    local gpu_id="$2"
    IFS='|' read -r fname exp extra <<<"${MODELS[${key}]}"
    local ckpt_path="${CKPT_DIR}/${fname}"
    local out_path="${OUT_ROOT}/${key}"
    local logfile="${out_path}/eval.log"
    mkdir -p "${out_path}"

    local EXTRA_ARGS=()
    # shellcheck disable=SC2206
    [[ -n "${extra}" ]] && EXTRA_ARGS=( ${extra} )

    echo ""
    echo ">>> [GPU ${gpu_id}] 评测 ${key}"
    echo "    ckpt:   ${ckpt_path}"
    echo "    exp:    ${exp}"
    echo "    log:    ${logfile}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python3 -m src.main \
        +experiment="${exp}" \
        mode=test \
        wandb.mode=disabled \
        wandb.name="eval_${key}" \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path="${SUBSET_EVAL_INDEX}" \
        "dataset.re10k.roots=[${DATA_PARENT}/re10k]" \
        checkpointing.load="${ckpt_path}" \
        test.save_image="${SAVE_IMAGE}" \
        test.compute_scores=true \
        test.output_path="${out_path}" \
        "${EXTRA_ARGS[@]}" \
        >"${logfile}" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo ""
        echo "===== 评测 ${key} 失败，eval.log 末尾 40 行如下 ====="
        tail -n 40 "${logfile}" || true
        echo "===== ${key} 日志结束 ====="
    fi
    return ${rc}
}

EVAL_PIDS=()
for i in "${!KEY_ORDER[@]}"; do
    key="${KEY_ORDER[$i]}"
    gpu_id=$(( i % (NUM_GPUS > 0 ? NUM_GPUS : 1) ))
    (
        run_eval_job "${key}" "${gpu_id}" || {
            err "评测 ${key} 失败（GPU ${gpu_id}）"
            exit 1
        }
    ) &
    EVAL_PIDS+=("$!")
    echo "  [spawn] ${key} → GPU${gpu_id} (pid=$!)"
    sleep 5   # 错开初始化：避免同时 import/加载冲突
done

fail=0
for i in "${!EVAL_PIDS[@]}"; do
    if wait "${EVAL_PIDS[$i]}"; then
        echo "  [done] ${KEY_ORDER[$i]} ✓"
    else
        err "${KEY_ORDER[$i]} 进程退出失败，详见 ${OUT_ROOT}/${KEY_ORDER[$i]}/eval.log"
        fail=$((fail+1))
    fi
done

# ============================ Step 5: 汇总指标 ============================
log "Step 5: 汇总指标"

export OUT_ROOT
python3 - <<'PYEOF'
import json, os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    # 实际目录结构: <OUT_ROOT>/<key>/<wandb.name>/scores_all_avg.json
    cand = list(d.rglob("scores_all_avg.json"))
    if not cand:
        print(f"[{d.name}] 未找到 scores_all_avg.json (评测可能失败)")
        continue
    m = json.load(open(cand[0]))
    flat = {}
    def _walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v, f"{prefix}{k}/")
            elif isinstance(v, (int, float)):
                flat[f"{prefix}{k}"] = v
    _walk(m)
    rows.append((d.name, flat))
    print(f"\n=== {d.name} ===   ({cand[0]})")
    for k, v in flat.items():
        print(f"  {k:<30s} = {v:.4f}")

# 生成 Markdown 汇总表
if rows:
    all_keys = sorted({k for _, fl in rows for k in fl})
    out_md = root / "summary.md"
    with open(out_md, "w") as f:
        f.write("| model | " + " | ".join(all_keys) + " |\n")
        f.write("|---" * (len(all_keys) + 1) + "|\n")
        for name, fl in rows:
            vals = [f"{fl.get(k, float('nan')):.4f}" for k in all_keys]
            f.write(f"| {name} | " + " | ".join(vals) + " |\n")
    print(f"\n汇总表 → {out_md}")
PYEOF

if [[ "${fail}" -eq 0 ]]; then
    log "全部评测完成 ✓"
    echo ""
    echo "结果目录: ${OUT_ROOT}"
    echo "  ├── nas3r/         ← 2-view 随机初始化"
    echo "  ├── multiview/     ← 10-view"
    echo "  ├── pretrained/    ← VGGT 初始化"
    echo "  ├── pretrained-I/  ← VGGT + GT intrinsics"
    echo "  └── summary.md     ← 指标汇总表"
else
    err "${fail} 个模型评测失败，请检查对应 eval.log"
    exit 1
fi

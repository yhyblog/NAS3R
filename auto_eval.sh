#!/usr/bin/env bash
# =============================================================================
# NAS3R 全自动评测脚本 · v2（可靠版） · 8×H100 80GB 单节点
#
# 本脚本目标：一次跑完，不再踩坑。
#
# 流程：
#   Step 0: clone / pull 代码 + 手动兜底 submodule + GLM 头文件
#   Step 1: pip 依赖 + pytorch3d + diff-gaussian-rasterization 编译（TORCH_CUDA_ARCH_LIST=9.0）
#   Step 2: 并行下载 4 个 checkpoint（~19.6GB，已存在则跳过）
#   Step 3: 下载 re10k test 数据子集（自动把 shard 搬到正确位置）
#   Step 4: sanity check —— 先用 1 个模型跑 3 个 scene 验证，失败立刻停止
#   Step 5: 并行评测 4 个 checkpoint（每张 H100 绑一个，test.batch_size=1 硬限制）
#   Step 6: 失败重试（单个模型最多 1 次）；仍失败则降级串行跑
#   Step 7: 聚合 4 个模型的 scores_all_avg.json 成 summary.md / summary.json / stdout
#
# 使用：
#   # 在远程机器上：
#   mkdir -p /opt/tiger/mmfinetune && cd /opt/tiger/mmfinetune
#   curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/auto_eval.sh -o auto_eval.sh
#   bash auto_eval.sh
#
#   # 关键环境变量：
#   PARENT_DIR=/opt/tiger/mmfinetune   # 仓库父目录
#   MAX_EVAL_SCENES=-1                 # 评测 scene 数；-1=全量 / 50=smoke
#   SKIP_CLONE=0 / SKIP_INSTALL=0 / SKIP_DOWNLOAD=0 / SKIP_SANITY=0
#   FORCE_SERIAL=0                     # 1 强制串行评测（调试用）
# =============================================================================

set -uo pipefail   # 注意：不用 -e，让每个 step 的错误自己处理

# ============================ 配置 ============================
PARENT_DIR="${PARENT_DIR:-/opt/tiger/mmfinetune}"
REPO_URL="${REPO_URL:-https://github.com/yhyblog/NAS3R.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_NAME="nas3r"
WORK_DIR="${PARENT_DIR}/${REPO_NAME}"

MAX_EVAL_SCENES="${MAX_EVAL_SCENES:--1}"
SAVE_IMAGE="${SAVE_IMAGE:-false}"
SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_SANITY="${SKIP_SANITY:-0}"
FORCE_SERIAL="${FORCE_SERIAL:-0}"

CKPT_DIR="${CKPT_DIR:-${WORK_DIR}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/datasets/re10k}"
OUT_ROOT="${OUT_ROOT:-${WORK_DIR}/outputs/eval}"

# H100 默认 sm_90，可手动覆盖
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HYDRA_FULL_ERROR=1   # 出错时打印完整栈，便于调试

# ============================ 日志工具 ============================
log()   { echo -e "\n\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
info()  { echo -e "  \033[0;34m→\033[0m $*"; }
ok()    { echo -e "  \033[1;32m✓\033[0m $*"; }
warn()  { echo -e "  \033[1;33m⚠\033[0m $*"; }
err()   { echo -e "\033[1;31m✗ ERROR: $*\033[0m" >&2; }

trap 'err "脚本在第 $LINENO 行意外退出（exit=$?）"' ERR

# ============================ Step 0: 代码仓库 ============================
step0_code() {
    log "Step 0: 准备代码仓库 → ${WORK_DIR}"
    mkdir -p "${PARENT_DIR}"
    cd "${PARENT_DIR}"

    if [[ "${SKIP_CLONE}" == "1" ]]; then
        info "SKIP_CLONE=1，跳过 clone"
        [[ -d "${WORK_DIR}/.git" ]] || { err "${WORK_DIR} 不是 git 仓库"; return 1; }
    elif [[ -d "${WORK_DIR}/.git" ]]; then
        info "仓库已存在，git pull"
        cd "${WORK_DIR}"
        git fetch origin "${REPO_BRANCH}" >/dev/null 2>&1 || true
        git checkout "${REPO_BRANCH}" >/dev/null 2>&1 || true
        if ! git pull --ff-only origin "${REPO_BRANCH}" 2>&1 | tail -3; then
            warn "git pull 失败（可能有本地改动），继续用当前版本"
        fi
    else
        info "clone ${REPO_URL}"
        git clone --recurse-submodules -b "${REPO_BRANCH}" "${REPO_URL}" "${REPO_NAME}" || \
        git clone --recurse-submodules -b "${REPO_BRANCH}" \
            "https://ghproxy.com/${REPO_URL}" "${REPO_NAME}" || \
        { err "代码 clone 失败"; return 1; }
    fi

    cd "${WORK_DIR}"

    # submodule 兜底 —— 之前多次失败在这里
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
        git clone --recurse-submodules -b camera \
            "https://ghproxy.com/${dgr_url}" "${dgr_dir}" || \
        { err "diff-gaussian-rasterization clone 失败"; return 1; }
    fi

    local glm_dir="${dgr_dir}/third_party/glm"
    if [[ ! -f "${glm_dir}/glm/glm.hpp" ]]; then
        info "下载 GLM 头文件"
        rm -rf "${glm_dir}"
        git clone --depth 1 https://github.com/g-truc/glm.git "${glm_dir}" || \
        {
            mkdir -p "${glm_dir}"
            curl -fsSL https://github.com/g-truc/glm/releases/download/1.0.1/glm-1.0.1-light.zip \
                -o /tmp/glm.zip && \
            unzip -qo /tmp/glm.zip -d "$(dirname "${glm_dir}")" && rm -f /tmp/glm.zip
        } || { err "GLM 下载失败"; return 1; }
    fi

    ok "代码就位  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
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
        err "torch 未安装，请先装 torch==2.5.1+cu121"
        return 1
    }

    python3 -m pip install --upgrade pip -q
    grep -v -E "^torch==|^torchvision==|^torchaudio==|scikit-video|pytorch3d" requirements.txt \
        > /tmp/requirements_fixed.txt
    pip install -r /tmp/requirements_fixed.txt --no-build-isolation 2>&1 | tail -10

    pip install sk-video 2>/dev/null || pip install scikit-video --no-deps 2>/dev/null || true

    if ! python3 -c "import pytorch3d" 2>/dev/null; then
        info "编译 pytorch3d (arch=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS})"
        pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation \
            2>&1 | tail -5 || warn "pytorch3d 编译失败（评测可能不依赖，继续）"
    else
        info "pytorch3d 已安装"
    fi

    info "编译 diff-gaussian-rasterization (arch=${TORCH_CUDA_ARCH_LIST})"
    pip install -e submodules/diff-gaussian-rasterization --no-build-isolation 2>&1 | tail -10 || \
        { err "diff-gaussian-rasterization 编译失败（评测必需）"; return 1; }

    # 验证关键扩展能 import。注意：ranrhuang/diff-gaussian-rasterization 的 camera 分支
    # 把包名改成了 diff_gauss_camera（src/model/decoder/cuda_splatting.py 也是这么 import 的）
    python3 -c "import diff_gauss_camera; print('  [ok] diff_gauss_camera import 成功')" || \
        { err "diff_gauss_camera import 失败"; return 1; }

    pip install -q "huggingface_hub[cli,hf_transfer]" colorama
    ok "依赖安装完成"
}

# ============================ Step 2: 下载 checkpoint ============================
CKPTS=(
    re10k_nas3r.ckpt
    re10k_nas3r_multiview.ckpt
    re10k_nas3r_pretrained.ckpt
    re10k_nas3r_pretrained-I.ckpt
)

download_one_ckpt() {
    local fname="$1"
    local dst="${CKPT_DIR}/${fname}"
    if [[ -f "${dst}" ]]; then
        # 基础完整性校验：文件应 >4GB
        local sz_mb
        sz_mb=$(du -m "${dst}" | awk '{print $1}')
        if [[ "${sz_mb}" -gt 4000 ]]; then
            echo "  [skip] ${fname} 已存在 (${sz_mb} MB)"
            return 0
        else
            warn "${fname} 大小异常 (${sz_mb}MB)，重下"
            rm -f "${dst}"
        fi
    fi
    echo "  [get ] ${fname}"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download RanranHuang/NAS3R "${fname}" \
            --local-dir "${CKPT_DIR}" --local-dir-use-symlinks False \
            >/dev/null 2>&1
    else
        curl -fL --retry 5 --retry-delay 5 -C - \
            -o "${dst}.part" \
            "https://huggingface.co/RanranHuang/NAS3R/resolve/main/${fname}" \
            >/dev/null 2>&1
        mv "${dst}.part" "${dst}"
    fi
    [[ -f "${dst}" ]] && echo "  [done] ${fname} ($(du -h "${dst}" | awk '{print $1}'))" || \
        { err "${fname} 下载失败"; return 1; }
}

step2_ckpts() {
    log "Step 2: 下载 4 个 checkpoint"
    if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
        info "SKIP_DOWNLOAD=1，跳过"
        return 0
    fi
    mkdir -p "${CKPT_DIR}"
    local pids=() fail=0
    for f in "${CKPTS[@]}"; do
        download_one_ckpt "$f" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || fail=$((fail+1))
    done
    if [[ "${fail}" -gt 0 ]]; then
        err "${fail} 个 checkpoint 下载失败"
        return 1
    fi
    ok "所有 checkpoint 就位"
    ls -lh "${CKPT_DIR}"/*.ckpt | awk '{print "    "$9" "$5}'
}

# ============================ Step 3: 数据子集 ============================
SUBSET_EVAL_INDEX="${WORK_DIR}/assets/evaluation_index_re10k.subset.json"

step3_data() {
    log "Step 3: 准备 re10k test 数据子集 (MAX_EVAL_SCENES=${MAX_EVAL_SCENES})"
    if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
        info "SKIP_DOWNLOAD=1，跳过下载"
        [[ -f "${SUBSET_EVAL_INDEX}" ]] || SUBSET_EVAL_INDEX="${WORK_DIR}/assets/evaluation_index_re10k.json"
        return 0
    fi

    mkdir -p "${DATA_DIR}"

    # 把可能被上一版错误脚本散落在外的 shard 搬回正确位置
    # （修复已知 bug：旧版 snapshot_download 把 .torch 下到 datasets/test/ 而非 datasets/re10k/test/）
    local need_regroup=0
    for alt in "${DATA_DIR}/../test" "${DATA_DIR}/../re10k/test" "${DATA_DIR}/re10k/test"; do
        if [[ -d "${alt}" && "$(realpath "${alt}")" != "$(realpath "${DATA_DIR}/test" 2>/dev/null || echo /dev/null)" ]]; then
            local n
            n=$(find "${alt}" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
            if [[ "${n}" -gt 0 ]]; then
                warn "检测到 ${n} 个 .torch 散落在 ${alt}，搬运到正确位置"
                mkdir -p "${DATA_DIR}/test"
                find "${alt}" -maxdepth 1 -name "*.torch" -exec mv -n {} "${DATA_DIR}/test/" \;
                rmdir "${alt}" 2>/dev/null || true
                need_regroup=1
            fi
        fi
    done

    # 执行 python 下载脚本
    local MAX_FLAG=()
    [[ "${MAX_EVAL_SCENES}" != "-1" ]] && MAX_FLAG=(--max-scenes "${MAX_EVAL_SCENES}")

    python3 -m src.scripts.prepare_re10k_test_subset \
        --data-dir "${DATA_DIR}" \
        --eval-index "${WORK_DIR}/assets/evaluation_index_re10k.json" \
        --subset-out "${SUBSET_EVAL_INDEX}" \
        "${MAX_FLAG[@]}" || {
            err "数据准备失败"
            return 1
        }

    # 验证：shard 数、index.json 不为空
    local shard_n scene_n
    shard_n=$(find "${DATA_DIR}/test" -maxdepth 1 -name "*.torch" 2>/dev/null | wc -l)
    scene_n=$(python3 -c "import json; print(len(json.load(open('${SUBSET_EVAL_INDEX}'))))" 2>/dev/null || echo 0)
    if [[ "${shard_n}" -eq 0 || "${scene_n}" -eq 0 ]]; then
        err "数据验证失败: shards=${shard_n}, scenes=${scene_n}"
        return 1
    fi
    ok "数据就位: ${shard_n} 个 .torch, ${scene_n} 个评测 scene"
    df -h "${DATA_DIR}" 2>/dev/null | tail -1 | awk '{print "    磁盘: 已用 "$3" / 总 "$2" ("$5" 占用)"}'
}

# ============================ 单个评测 job ============================
declare -A MODELS=(
    [nas3r]="re10k_nas3r.ckpt|nas3r/random/re10k|"
    [multiview]="re10k_nas3r_multiview.ckpt|nas3r/random/re10k|dataset.re10k.view_sampler.num_context_views=10"
    [pretrained]="re10k_nas3r_pretrained.ckpt|nas3r/pretrained/re10k|"
    [pretrained-I]="re10k_nas3r_pretrained-I.ckpt|nas3r/pretrained/re10k-I|"
)
KEY_ORDER=(nas3r multiview pretrained pretrained-I)

# 参数: $1=key  $2=gpu_id  $3=extra_override (sanity 时用来限定 batch 数)
run_eval_job() {
    local key="$1" gpu_id="$2" extra_override="${3:-}"
    IFS='|' read -r fname exp extra <<<"${MODELS[${key}]}"
    local ckpt_path="${CKPT_DIR}/${fname}"
    local out_path="${OUT_ROOT}/${key}"
    local logfile="${out_path}/eval.log"
    mkdir -p "${out_path}"
    local DATA_PARENT
    DATA_PARENT="$(dirname "${DATA_DIR}")"

    local EXTRA=()
    # shellcheck disable=SC2206
    [[ -n "${extra}" ]] && EXTRA+=( ${extra} )
    # shellcheck disable=SC2206
    [[ -n "${extra_override}" ]] && EXTRA+=( ${extra_override} )

    info "[GPU ${gpu_id}] ${key} · exp=${exp} · log=${logfile}"

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
        "${EXTRA[@]}" \
        >"${logfile}" 2>&1
    local rc=$?
    return $rc
}

# ============================ Step 4: sanity check ============================
step4_sanity() {
    log "Step 4: Sanity check - 用 nas3r 跑 1 个 batch 验证"
    if [[ "${SKIP_SANITY}" == "1" ]]; then
        info "SKIP_SANITY=1，跳过"
        return 0
    fi

    local out_path="${OUT_ROOT}/_sanity"
    rm -rf "${out_path}"
    mkdir -p "${out_path}"

    info "跑 sanity（只评 1 个 batch，约 30s）..."
    local logfile="${out_path}/sanity.log"
    local DATA_PARENT
    DATA_PARENT="$(dirname "${DATA_DIR}")"

    CUDA_VISIBLE_DEVICES=0 \
    timeout 300 python3 -m src.main \
        +experiment=nas3r/random/re10k \
        mode=test \
        wandb.mode=disabled \
        wandb.name="sanity" \
        dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
        dataset.re10k.view_sampler.index_path="${SUBSET_EVAL_INDEX}" \
        "dataset.re10k.roots=[${DATA_PARENT}/re10k]" \
        checkpointing.load="${CKPT_DIR}/re10k_nas3r.ckpt" \
        test.save_image=false \
        test.compute_scores=true \
        test.output_path="${out_path}" \
        +trainer.limit_test_batches=1 \
        >"${logfile}" 2>&1
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        err "Sanity check 失败（rc=${rc}），下面是日志末尾 50 行："
        echo "------------------------------------------------"
        tail -n 50 "${logfile}"
        echo "------------------------------------------------"
        err "请先根据上述错误修复环境，再重跑。"
        err "常见问题："
        err "  1. diff-gaussian-rasterization 没编译 → 手动: pip install -e submodules/diff-gaussian-rasterization --no-build-isolation"
        err "  2. 数据位置不对 → 检查 ls ${DATA_DIR}/test/*.torch | head"
        err "  3. torch/cuda 版本不匹配 → python -c 'import torch; print(torch.version.cuda)'"
        return 1
    fi
    ok "Sanity check 通过"
    rm -rf "${out_path}"
}

# ============================ Step 5+6: 评测（并行+重试+串行降级）============================
step5_evaluate() {
    log "Step 5: 并行评测 4 个 checkpoint"

    local NUM_GPUS
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    info "可用 GPU 数: ${NUM_GPUS}"
    [[ "${NUM_GPUS}" -lt 1 ]] && { err "没有 GPU"; return 1; }

    # 清理旧输出（保留 _sanity 已删除）
    for k in "${KEY_ORDER[@]}"; do
        rm -rf "${OUT_ROOT}/${k}"
    done

    local mode="parallel"
    [[ "${FORCE_SERIAL}" == "1" || "${NUM_GPUS}" -lt 4 ]] && mode="serial"

    if [[ "${mode}" == "parallel" ]]; then
        info "模式: parallel (每张 H100 一个模型)"
        local pids=()
        for i in "${!KEY_ORDER[@]}"; do
            local key="${KEY_ORDER[$i]}"
            local gpu_id=$(( i % NUM_GPUS ))
            ( run_eval_job "${key}" "${gpu_id}" ) &
            pids+=("$!")
            echo "    [spawn] ${key} → GPU${gpu_id} (pid=$!)"
            sleep 5   # 错开初始化
        done

        local retry=()
        for i in "${!pids[@]}"; do
            if wait "${pids[$i]}"; then
                ok "${KEY_ORDER[$i]} 完成"
            else
                warn "${KEY_ORDER[$i]} 失败，将降级重试"
                retry+=("${KEY_ORDER[$i]}")
            fi
        done

        # 失败的串行重试（单卡 GPU0，避免资源竞争）
        if [[ "${#retry[@]}" -gt 0 ]]; then
            log "重试失败的 ${#retry[@]} 个模型（串行，GPU0）"
            for k in "${retry[@]}"; do
                info "重试 ${k} ..."
                if run_eval_job "${k}" 0; then
                    ok "${k} 重试成功"
                else
                    err "${k} 重试仍失败，日志末尾："
                    tail -n 30 "${OUT_ROOT}/${k}/eval.log"
                fi
            done
        fi
    else
        info "模式: serial (逐个跑)"
        for i in "${!KEY_ORDER[@]}"; do
            local key="${KEY_ORDER[$i]}"
            local gpu_id=$(( i % NUM_GPUS ))
            if run_eval_job "${key}" "${gpu_id}"; then
                ok "${key} 完成"
            else
                err "${key} 失败，日志末尾："
                tail -n 30 "${OUT_ROOT}/${key}/eval.log"
            fi
        done
    fi
}

# ============================ Step 7: 汇总 ============================
step7_summarize() {
    log "Step 7: 汇总指标"
    export OUT_ROOT KEY_ORDER_STR="${KEY_ORDER[*]}"

    python3 <<'PYEOF'
import json, os, sys
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
key_order = os.environ["KEY_ORDER_STR"].split()

results = {}          # key -> {metric: value}
per_scene = {}        # key -> scene_id -> [overlap, psnr, ssim, lpips]
missing = []

for key in key_order:
    d = root / key
    avg_f = next(d.rglob("scores_all_avg.json"), None) if d.exists() else None
    all_f = next(d.rglob("scores_all.json"), None) if d.exists() else None
    if avg_f is None:
        missing.append(key)
        continue
    m = json.load(open(avg_f))
    flat = {}
    def walk(x, prefix=""):
        for k, v in x.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}/")
            elif isinstance(v, (int, float)):
                flat[f"{prefix}{k}"] = float(v)
            elif isinstance(v, list) and all(isinstance(e, (int, float)) for e in v):
                # pose AUC 等返回 list
                for i, e in enumerate(v):
                    flat[f"{prefix}{k}[{i}]"] = float(e)
    walk(m)
    results[key] = flat
    if all_f:
        try:
            per_scene[key] = json.load(open(all_f))
        except Exception:
            pass

# 想要的核心指标顺序
CORE_METRICS = ["psnr", "ssim", "lpips", "R_auc[0]", "R_auc[1]", "R_auc[2]",
                "t_auc[0]", "t_auc[1]", "t_auc[2]", "pose_auc[0]", "pose_auc[1]", "pose_auc[2]",
                "R_median", "t_median", "pose_median"]

def get(flat, k):
    # 兼容带或不带前缀的写法
    if k in flat: return flat[k]
    for key in flat:
        if key.endswith("/" + k) or key == k:
            return flat[key]
    return None

# stdout 打印
print("\n" + "=" * 70)
print(f"{'Model':<16} " + " ".join(f"{m:>8}" for m in CORE_METRICS[:5]))
print("-" * 70)
for k in key_order:
    if k not in results:
        print(f"{k:<16} {'FAILED':>8}")
        continue
    row = results[k]
    vals = [get(row, m) for m in CORE_METRICS[:5]]
    cells = [f"{v:8.4f}" if isinstance(v, float) else "    -   " for v in vals]
    print(f"{k:<16} " + " ".join(cells))
print("=" * 70)

# 完整版 markdown
out_md = root / "summary.md"
with open(out_md, "w") as f:
    f.write("# NAS3R 评测汇总\n\n")
    f.write(f"评测 scene 数: {len(next(iter(per_scene.values())) if per_scene else [])}\n\n")
    f.write("## 核心指标\n\n")
    # 只保留有数据的 metric
    have = set()
    for r in results.values(): have.update(r.keys())
    cols = [m for m in CORE_METRICS if any(get(r, m) is not None for r in results.values())]
    f.write("| Model | " + " | ".join(cols) + " |\n")
    f.write("|" + "---|" * (len(cols) + 1) + "\n")
    for k in key_order:
        if k not in results:
            f.write(f"| {k} | " + " | ".join(["FAILED"] * len(cols)) + " |\n")
        else:
            r = results[k]
            vals = [get(r, m) for m in cols]
            cells = [f"{v:.4f}" if isinstance(v, float) else "-" for v in vals]
            f.write(f"| {k} | " + " | ".join(cells) + " |\n")
    f.write("\n## 全部 raw 指标（flatten）\n\n")
    all_keys = sorted({k for r in results.values() for k in r})
    f.write("| Model | " + " | ".join(all_keys) + " |\n")
    f.write("|" + "---|" * (len(all_keys) + 1) + "\n")
    for k in key_order:
        if k not in results: continue
        r = results[k]
        vals = [f"{r.get(m, float('nan')):.4f}" for m in all_keys]
        f.write(f"| {k} | " + " | ".join(vals) + " |\n")

# JSON 汇总
out_json = root / "summary.json"
with open(out_json, "w") as f:
    json.dump({
        "key_order": key_order,
        "results": results,
        "missing": missing,
    }, f, indent=2)

print(f"\n结果文件：")
print(f"  {out_md}")
print(f"  {out_json}")
for k in key_order:
    if k in results:
        print(f"  {root}/{k}/eval_{k}/  (scores_all.json 含 per-scene 指标)")

if missing:
    print(f"\n⚠ 以下模型未产出结果: {missing}")
    sys.exit(2)
PYEOF
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        ok "全部评测完成 ✓"
        echo ""
        cat "${OUT_ROOT}/summary.md"
    else
        warn "部分模型失败，请检查对应 eval.log"
        return ${rc}
    fi
}

# ============================ 主流程 ============================
main() {
    local t0=$(date +%s)

    step0_code       || exit 1
    step1_install    || exit 1
    mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUT_ROOT}"
    step2_ckpts      || exit 1
    step3_data       || exit 1
    step4_sanity     || exit 1
    step5_evaluate   || warn "部分评测失败"
    step7_summarize  || warn "汇总发现缺失"

    local dt=$(( $(date +%s) - t0 ))
    log "总耗时: $((dt/60)) 分 $((dt%60)) 秒"
    echo ""
    echo "结果目录结构："
    echo "  ${OUT_ROOT}/"
    echo "  ├── nas3r/         ← 2-view 随机初始化"
    echo "  ├── multiview/     ← 10-view"
    echo "  ├── pretrained/    ← VGGT 初始化"
    echo "  ├── pretrained-I/  ← VGGT + GT intrinsics"
    echo "  ├── summary.md     ← 指标汇总表"
    echo "  └── summary.json   ← 机器可读版"
}

main "$@"

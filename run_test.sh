#!/usr/bin/env bash
# =============================================================================
# NAS3R 评测一键脚本（8×H100 80GB 单节点适配版）
#
# 功能：
#   1. 从 HuggingFace 下载指定的预训练 checkpoint
#   2. 只下载评测真正需要的 re10k test shard
#   3. 评测：按 README 命令跑指标 (PSNR/SSIM/LPIPS + pose AUC)
#
# 用法：
#   bash run_test.sh                 # 默认：re10k_nas3r.ckpt，单卡评测（指标准确）
#   bash run_test.sh nas3r           # 2-view 随机初始化模型
#   bash run_test.sh multiview       # 10-view 模型
#   bash run_test.sh pretrained      # VGGT 初始化
#   bash run_test.sh pretrained-I    # VGGT 初始化 + GT intrinsics
#   bash run_test.sh all             # 4 个 checkpoint 并行跑（每卡一个，4 卡占用）
#   bash run_test.sh all-sharded     # 4 个 checkpoint 串行跑，每个用 2 卡 DDP 切分 scene
#
# 可通过环境变量控制：
#   CKPT_DIR=./checkpoints       checkpoint 存放位置
#   DATA_DIR=./datasets/re10k    数据存放位置
#   MAX_EVAL_SCENES=100          smoke-test 用，只评前 N 个 scene；-1 表示全部
#   SAVE_IMAGE=false             是否保存渲染图（README 默认 false）
#   SKIP_DOWNLOAD=0              1 则跳过 checkpoint/数据下载
#   EVAL_BATCH_SIZE=4            test DataLoader 的 batch_size（80GB 可开大）
#   EVAL_NUM_WORKERS=8           test DataLoader worker 数
#   GPUS_PER_EVAL=1              单个 checkpoint 使用几张卡（>1 会有 scene 分桶风险，
#                                详见 WARN；建议保持 1 以保证指标精确）
# =============================================================================
#
# ⚠️  多卡评测注意事项：
#   src/model/model_wrapper.py 的 on_test_end 里 test_step_outputs / running_metrics
#   没有做 DDP all_gather，多卡 test 时每个 rank 只会看到自己处理过的 scene，
#   且最终 dump 的是 rank-0 的本地 dict，会丢失其他 rank 的结果。
#   为了得到 README 中报告的精确指标：
#     - 默认 GPUS_PER_EVAL=1，一个 checkpoint 用 1 张卡跑（scene 全量、指标准确）；
#     - 模式 "all" 通过同时启动 4 个单卡进程（CUDA_VISIBLE_DEVICES 分别绑 0/1/2/3）
#       并行评 4 个 checkpoint，整体只需单模型的耗时。
# =============================================================================

set -euo pipefail

# --------- 基础配置 ---------
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORK_DIR}"

CKPT_DIR="${CKPT_DIR:-${WORK_DIR}/checkpoints}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/datasets/re10k}"
EVAL_INDEX="${EVAL_INDEX:-${WORK_DIR}/assets/evaluation_index_re10k.json}"
MAX_EVAL_SCENES="${MAX_EVAL_SCENES:-50}"
SAVE_IMAGE="${SAVE_IMAGE:-false}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
OUT_ROOT="${OUT_ROOT:-${WORK_DIR}/outputs/eval}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
GPUS_PER_EVAL="${GPUS_PER_EVAL:-1}"

mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUT_ROOT}"

# --------- checkpoint 元信息 ---------
# key => "<ckpt_filename>|<experiment>|<extra_hydra_overrides>"
declare -A MODELS=(
  [nas3r]="re10k_nas3r.ckpt|nas3r/random/re10k|"
  [multiview]="re10k_nas3r_multiview.ckpt|nas3r/random/re10k|dataset.re10k.view_sampler.num_context_views=10"
  [pretrained]="re10k_nas3r_pretrained.ckpt|nas3r/pretrained/re10k|"
  [pretrained-I]="re10k_nas3r_pretrained-I.ckpt|nas3r/pretrained/re10k-I|"
)
ALL_KEYS=(nas3r multiview pretrained pretrained-I)

TARGET="${1:-nas3r}"
MODE="single"  # single | parallel-all | sharded-all
case "${TARGET}" in
  all)          RUN_KEYS=("${ALL_KEYS[@]}"); MODE="parallel-all" ;;
  all-sharded)  RUN_KEYS=("${ALL_KEYS[@]}"); MODE="sharded-all"; GPUS_PER_EVAL=2 ;;
  *)
    if [[ -z "${MODELS[${TARGET}]+x}" ]]; then
      echo "[ERROR] 未知 target: ${TARGET}"
      echo "可选: ${ALL_KEYS[*]} | all | all-sharded"
      exit 1
    fi
    RUN_KEYS=("${TARGET}") ;;
esac

# --------- GPU 探测 ---------
NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
NUM_GPUS="${NUM_GPUS:-0}"
echo "[env] 可用 GPU 数: ${NUM_GPUS}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[ERROR] 未检测到 GPU"; exit 1
fi

# --------- Step 1: 下载 checkpoint ---------
download_ckpt() {
  local fname="$1"
  local dst="${CKPT_DIR}/${fname}"
  if [[ -f "${dst}" ]]; then
    echo "[ckpt] 已存在: ${dst}"
    return 0
  fi
  local url="https://huggingface.co/RanranHuang/NAS3R/resolve/main/${fname}"
  echo "[ckpt] 下载 ${fname} -> ${dst}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download RanranHuang/NAS3R "${fname}" \
        --local-dir "${CKPT_DIR}" --local-dir-use-symlinks False
  else
    curl -L --retry 3 --retry-delay 5 -o "${dst}.part" "${url}"
    mv "${dst}.part" "${dst}"
  fi
}

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  for key in "${RUN_KEYS[@]}"; do
    IFS='|' read -r fname _ _ <<<"${MODELS[${key}]}"
    download_ckpt "${fname}"
  done
fi

# --------- Step 2: 准备评测数据子集 ---------
SUBSET_EVAL_INDEX="${WORK_DIR}/assets/evaluation_index_re10k.subset.json"

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  echo ""
  echo "============================================"
  echo "[Step 2] 准备 re10k test 数据子集"
  echo "  目标 scene 上限: ${MAX_EVAL_SCENES}"
  echo "============================================"
  MAX_FLAG=()
  if [[ "${MAX_EVAL_SCENES}" != "-1" ]]; then
    MAX_FLAG=(--max-scenes "${MAX_EVAL_SCENES}")
  fi
  python -m src.scripts.prepare_re10k_test_subset \
      --data-dir "${DATA_DIR}" \
      --eval-index "${EVAL_INDEX}" \
      --subset-out "${SUBSET_EVAL_INDEX}" \
      "${MAX_FLAG[@]}"
else
  echo "[Step 2] 跳过数据下载（SKIP_DOWNLOAD=1）"
  [[ -f "${SUBSET_EVAL_INDEX}" ]] || SUBSET_EVAL_INDEX="${EVAL_INDEX}"
fi

DATA_PARENT="$(dirname "${DATA_DIR}")"

# --------- 核心：单次评测函数 ---------
# 参数: $1=key  $2=gpus (CUDA_VISIBLE_DEVICES)  $3=logfile
run_one_eval() {
  local key="$1"
  local gpus="$2"
  local logfile="$3"

  IFS='|' read -r fname exp extra <<<"${MODELS[${key}]}"
  local ckpt_path="${CKPT_DIR}/${fname}"
  local out_path="${OUT_ROOT}/${key}"
  mkdir -p "${out_path}"

  local n_gpus
  n_gpus=$(awk -F, '{print NF}' <<<"${gpus}")

  echo ""
  echo "--- 评测 [${key}] | GPU=${gpus} (${n_gpus} 卡) | ckpt=${fname}"
  echo "    experiment: ${exp}"
  echo "    output:     ${out_path}"
  echo "    log:        ${logfile}"

  local EXTRA_ARGS=()
  if [[ -n "${extra}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${extra} )
  fi

  CUDA_VISIBLE_DEVICES="${gpus}" \
  python -m src.main \
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
      data_loader.test.batch_size="${EVAL_BATCH_SIZE}" \
      data_loader.test.num_workers="${EVAL_NUM_WORKERS}" \
      "${EXTRA_ARGS[@]}" \
      >"${logfile}" 2>&1
}

# --------- Step 3: 执行评测 ---------
echo ""
echo "============================================"
echo "[Step 3] 开始评测  mode=${MODE}"
echo "============================================"

case "${MODE}" in
  single)
    # 单模型评测：默认用 1 张卡（GPU 0）；GPUS_PER_EVAL>1 会启用 DDP（指标会不完整，见顶部 WARN）
    if [[ "${GPUS_PER_EVAL}" -gt 1 ]]; then
      echo "[WARN] GPUS_PER_EVAL=${GPUS_PER_EVAL}>1: 多卡 test 的聚合逻辑不完整，"
      echo "       最终 scores_all.json 只会包含 rank-0 子集。如需精确指标请设为 1。"
    fi
    gpus=$(seq -s, 0 $((GPUS_PER_EVAL-1)))
    run_one_eval "${RUN_KEYS[0]}" "${gpus}" "${OUT_ROOT}/${RUN_KEYS[0]}/eval.log"
    tail -n 40 "${OUT_ROOT}/${RUN_KEYS[0]}/eval.log" || true
    ;;

  parallel-all)
    # 4 个 checkpoint 同时跑，每张 H100 绑一个进程：GPU0->nas3r, 1->multiview, 2->pretrained, 3->pretrained-I
    if [[ "${NUM_GPUS}" -lt 4 ]]; then
      echo "[WARN] 可用 GPU=${NUM_GPUS}<4，部分模型会排队在同一张卡上跑"
    fi
    pids=()
    for i in "${!RUN_KEYS[@]}"; do
      key="${RUN_KEYS[$i]}"
      gpu_id=$((i % NUM_GPUS))
      log="${OUT_ROOT}/${key}/eval.log"
      mkdir -p "$(dirname "${log}")"
      (
        run_one_eval "${key}" "${gpu_id}" "${log}"
      ) &
      pids+=("$!")
      echo "[spawn] ${key} -> GPU${gpu_id} (pid=$!)"
      sleep 3   # 错开初始化，避免同时访问 HF cache / 下载冲突
    done
    echo ""
    echo "[Step 3] 4 个评测进程已启动，等待完成..."
    failed=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        echo "[done] ${RUN_KEYS[$i]} OK"
      else
        echo "[FAIL] ${RUN_KEYS[$i]} 失败，查看 ${OUT_ROOT}/${RUN_KEYS[$i]}/eval.log"
        failed=$((failed+1))
      fi
    done
    [[ "${failed}" -eq 0 ]] || exit 1
    ;;

  sharded-all)
    # 每个 checkpoint 用 GPUS_PER_EVAL 卡（DDP）串行。这种模式下评测结果不完整（见顶部 WARN），
    # 但可用于"快速过一遍"看吞吐。生产评测请勿使用。
    for key in "${RUN_KEYS[@]}"; do
      gpus=$(seq -s, 0 $((GPUS_PER_EVAL-1)))
      run_one_eval "${key}" "${gpus}" "${OUT_ROOT}/${key}/eval.log"
    done
    ;;
esac

# --------- Step 4: 汇总指标 ---------
echo ""
echo "============================================"
echo "[Step 4] 指标汇总"
echo "============================================"

python - <<'PYEOF'
import json, os
from pathlib import Path
root = Path(os.environ.get("OUT_ROOT", "outputs/eval"))
for d in sorted(root.iterdir()) if root.exists() else []:
    f = d / f"eval_{d.name}" / "scores_all_avg.json"
    # output_path 下实际结构是 <output_path>/<wandb.name>/scores_all_avg.json
    if not f.exists():
        # 找一下子目录
        cand = list(d.glob("*/scores_all_avg.json"))
        f = cand[0] if cand else None
    if f and f.exists():
        m = json.load(open(f))
        print(f"[{d.name}] {f}")
        for k, v in m.items():
            if isinstance(v, (int, float)):
                print(f"    {k:20s} = {v:.4f}")
    else:
        print(f"[{d.name}] 未找到 scores_all_avg.json")
PYEOF

echo ""
echo "完成。每个模型的日志与指标位于 ${OUT_ROOT}/<key>/"

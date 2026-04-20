#!/usr/bin/env bash
# =============================================================================
# GPU 实时监控 + 落盘 + 异常告警
#
# 用法：
#   bash monitor_gpu.sh                    # 实时显示 + 记录到 csv
#   bash monitor_gpu.sh --no-watch         # 只记录不显示
#   INTERVAL=5 bash monitor_gpu.sh         # 5 秒间隔（默认 10 秒）
#   OUT_DIR=/tmp/gpu_log bash monitor_gpu.sh
#
# 输出：
#   <OUT_DIR>/gpu_<timestamp>.csv         # 完整历史数据
#   同时 stdout 打印紧凑表格
# =============================================================================

set -uo pipefail

INTERVAL="${INTERVAL:-10}"
OUT_DIR="${OUT_DIR:-./outputs/monitor}"
WATCH="${1:-}"

mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/gpu_$(date +%Y%m%d_%H%M%S).csv"

# 检测 GPU 数
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "[ERROR] 没检测到 GPU" >&2
    exit 1
fi

echo "timestamp,gpu_idx,util_gpu_pct,util_mem_pct,mem_used_mb,mem_total_mb,power_w,temp_c" > "${CSV}"
echo "监控 ${NUM_GPUS} 张 GPU，间隔 ${INTERVAL}s，记录 → ${CSV}"
echo "Ctrl+C 停止"
echo ""

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# 记录上一次的显存，检测泄漏
declare -A LAST_MEM

while true; do
    ts=$(date +'%H:%M:%S')
    query=$(nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null)

    if [[ "${WATCH}" != "--no-watch" ]]; then
        # 清屏 + 表头
        printf "\033[H\033[2J"   # clear
        echo -e "${CYAN}=== GPU Monitor $(date '+%Y-%m-%d %H:%M:%S') (interval=${INTERVAL}s) ===${NC}"
        echo -e "${CYAN}日志文件: ${CSV}${NC}"
        echo ""
        printf "%3s  %-6s  %-6s  %-13s  %-6s  %-6s  %s\n" \
            "GPU" "util%" "mem%" "memory" "pwr_W" "temp" "状态"
        echo "───────────────────────────────────────────────────────────"
    fi

    while IFS=',' read -r idx util_gpu util_mem mem_used mem_total pwr temp; do
        idx=$(echo "${idx}" | tr -d ' ')
        util_gpu=$(echo "${util_gpu}" | tr -d ' ')
        util_mem=$(echo "${util_mem}" | tr -d ' ')
        mem_used=$(echo "${mem_used}" | tr -d ' ')
        mem_total=$(echo "${mem_total}" | tr -d ' ')
        pwr=$(echo "${pwr}" | tr -d ' ' | awk '{printf "%.0f", $1}')
        temp=$(echo "${temp}" | tr -d ' ')

        # 写 csv
        echo "$(date -Iseconds),${idx},${util_gpu},${util_mem},${mem_used},${mem_total},${pwr},${temp}" >> "${CSV}"

        if [[ "${WATCH}" == "--no-watch" ]]; then
            continue
        fi

        # 状态判断
        status=""
        color="${GREEN}"
        if [[ "${util_gpu}" -lt 30 ]]; then
            status="⚠ 利用率低"; color="${YELLOW}"
        elif [[ "${util_gpu}" -lt 70 ]]; then
            status="△ 数据瓶颈?"; color="${YELLOW}"
        elif [[ "${util_gpu}" -ge 85 ]]; then
            status="✓ 高效"; color="${GREEN}"
        else
            status="○ 正常"; color="${GREEN}"
        fi

        # 温度告警
        if [[ "${temp}" -ge 85 ]]; then
            status="${status} 🔥高温"; color="${RED}"
        fi

        # 显存泄漏检测（连续上涨）
        prev="${LAST_MEM[${idx}]:-0}"
        delta=$(( mem_used - prev ))
        LAST_MEM[${idx}]="${mem_used}"
        if [[ "${prev}" -ne 0 && "${delta}" -gt 500 ]]; then
            status="${status} 📈+${delta}MB"
            color="${YELLOW}"
        fi

        # 显存百分比
        mem_pct=$(awk "BEGIN{printf \"%.0f\", ${mem_used}/${mem_total}*100}")

        printf "${color}%3s  %5s%%  %5s%%  %5s/%-5s MB %3s%%  %5sW  %s°C  %s${NC}\n" \
            "${idx}" "${util_gpu}" "${util_mem}" "${mem_used}" "${mem_total}" "${mem_pct}" "${pwr}" "${temp}" "${status}"
    done <<< "${query}"

    if [[ "${WATCH}" != "--no-watch" ]]; then
        echo ""
        echo -e "${CYAN}累计记录数: $(wc -l < "${CSV}")  |  csv 大小: $(du -h "${CSV}" | awk '{print $1}')${NC}"
    fi

    sleep "${INTERVAL}"
done

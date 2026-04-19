#!/usr/bin/env python3
"""
把 outputs/eval/<model>/eval_<model>/ 下产出的 JSON 整理成一个 Markdown 汇总。

用法（在 /opt/tiger/mmfinetune/nas3r 下）:
    python3 summarize_eval.py
    python3 summarize_eval.py --out outputs/eval/summary.md

也支持从 eval.log 里兜底抓指标（在 JSON 未落盘时也能工作）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


MODELS = [
    ("nas3r",        "2-view (random init)",            "re10k_nas3r.ckpt"),
    ("multiview",    "10-view",                         "re10k_nas3r_multiview.ckpt"),
    ("pretrained",   "2-view (VGGT pretrained)",        "re10k_nas3r_pretrained.ckpt"),
    ("pretrained-I", "2-view (VGGT + GT intrinsics)",   "re10k_nas3r_pretrained-I.ckpt"),
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 无法解析 {path}: {e}", file=sys.stderr)
        return None


def find_artifacts(root: Path, key: str) -> dict[str, Any]:
    """按顺序查 scores_all_avg / scores_sub_avg / eval.log。"""
    model_dir = root / key
    art: dict[str, Any] = {"model_dir": model_dir, "log": None}

    if not model_dir.exists():
        return art

    for f in model_dir.rglob("scores_all_avg.json"):
        art["avg"] = load_json(f)
        art["avg_path"] = f
        break
    for f in model_dir.rglob("scores_sub_avg.json"):
        art["sub"] = load_json(f)
        art["sub_path"] = f
        break
    # benchmark (推理速度 / 显存)
    for f in model_dir.rglob("benchmark.json"):
        art["bench"] = load_json(f)
        break
    for f in model_dir.rglob("peak_memory.json"):
        art["mem"] = load_json(f)
        break
    log = model_dir / "eval.log"
    if log.exists():
        art["log"] = log

    return art


# eval.log 兜底解析 -------------------------------------------------------
TABLE_HEADERS = ["lpips", "ssim", "psnr", "tgt_e_R", "tgt_e_t", "tgt_e_pose",
                 "cxt_e_R", "cxt_e_t", "cxt_e_pose"]


def parse_overlap_block(text: str) -> dict[str, dict[str, float]] | None:
    """
    从 eval.log 末尾抓：
        All Pairs:
        Method      lpips    ssim    psnr    tgt_e_R ...
        ours       0.194   0.764  23.135  ...
        Overlap: medium
        ...
    返回 {all: {...}, medium: {...}, small: {...}, large: {...}}
    """
    # 取最后一次出现的"All Pairs:"开始直到文件末尾
    idx = text.rfind("All Pairs:")
    if idx < 0:
        return None
    chunk = text[idx:]
    result: dict[str, dict[str, float]] = {}
    # 每个段由 "All Pairs:" 或 "Overlap: <x>" 起头
    sections = re.split(r"^(All Pairs:|Overlap:\s*\w+)\s*$", chunk, flags=re.M)
    # sections = ["", "All Pairs:", "...block...", "Overlap: medium", "...", ...]
    for i in range(1, len(sections), 2):
        head = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        key = "all" if head.startswith("All Pairs") else head.split(":", 1)[1].strip()
        # 找第一行 "ours  <numbers>"
        m = re.search(r"^\s*ours\s+([\d.\-eE\s]+)$", body, flags=re.M)
        if not m:
            continue
        nums = [float(x) for x in m.group(1).split()]
        row = dict(zip(TABLE_HEADERS, nums))
        result[key] = row
    return result or None


POSE_AUC_RE = re.compile(
    r"Pose AUC ours (?:(\w+)\s+)?of\s+(R|t|pose):\s+\[([\d\.,\s\-e]+)\]\s+median error\s+([\d\.eE\-]+)"
)


def parse_pose_auc(text: str) -> dict[str, dict[str, list[float] | float]]:
    """
    返回嵌套：
        {"all":    {"R":   {"auc":[...], "median":...}, "t":..., "pose":...},
         "medium": {...}, "small":..., "large":...}
    """
    out: dict[str, dict[str, Any]] = {}
    for m in POSE_AUC_RE.finditer(text):
        subset = (m.group(1) or "all").strip()  # '', medium, small, large
        metric = m.group(2)
        auc = [float(x) for x in m.group(3).split(",")]
        median = float(m.group(4))
        out.setdefault(subset, {})[metric] = {"auc_5_10_20": auc, "median": median}
    return out


def has_completed(log_text: str) -> bool:
    """判断 eval.log 是否走到了 on_test_end（打印 Test metric 表格）。"""
    return "Test metric" in log_text and "All Pairs:" in log_text


_BENCH_RE = re.compile(r"^\s*(encoder|decoder):\s*(\d+)\s+calls,\s*avg\.\s*([\d.eE+-]+)\s+seconds", re.M)


def parse_bench_from_log(text: str) -> dict[str, dict[str, float]]:
    """从 eval.log 末尾解析 encoder/decoder 的平均耗时。"""
    result: dict[str, dict[str, float]] = {}
    for m in _BENCH_RE.finditer(text):
        result[m.group(1)] = {"num_calls": int(m.group(2)), "avg": float(m.group(3))}
    return result


# Markdown 生成 -----------------------------------------------------------
def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _get_block(r: dict | None, ov: str) -> dict:
    if not r:
        return {}
    overlap = r.get("overlap") or {}
    return overlap.get(ov) or {}


def _get_pose(r: dict | None, subset: str, metric: str) -> dict | None:
    if not r:
        return None
    pa = (r.get("pose_auc") or {}).get(subset) or {}
    return pa.get(metric)


def write_md(results: dict[str, dict[str, Any]], out_path: Path) -> None:
    lines = []
    lines.append("# NAS3R · RealEstate10K 评测结果汇总\n")
    lines.append("数据集：RealEstate10K test (assets/evaluation_index_re10k.json，共 5601 个有效 scene)\n")
    lines.append("评测命令参考 README，模式：`mode=test`，view_sampler=`evaluation`。\n")

    # === 渲染质量（All Pairs）===
    lines.append("## 1. 新视角合成质量（All Pairs）\n")
    lines.append("| Model | Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
    lines.append("|---|---|---|---|---|")
    for key, desc, ckpt in MODELS:
        r = results.get(key)
        block = _get_block(r, "all")
        lines.append(
            f"| `{key}` | {desc} | {fmt(block.get('psnr'), 3)} "
            f"| {fmt(block.get('ssim'), 4)} | {fmt(block.get('lpips'), 4)} |"
        )

    # === 分 overlap 指标 ===
    lines.append("\n## 2. 按 Overlap 分档的渲染质量\n")
    for ov in ["large", "medium", "small"]:
        lines.append(f"### Overlap: **{ov}**\n")
        lines.append("| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
        lines.append("|---|---|---|---|")
        for key, desc, _ in MODELS:
            r = results.get(key)
            block = _get_block(r, ov)
            lines.append(
                f"| `{key}` | {fmt(block.get('psnr'), 3)} "
                f"| {fmt(block.get('ssim'), 4)} | {fmt(block.get('lpips'), 4)} |"
            )
        lines.append("")

    # === 相机位姿误差 (平均) ===
    lines.append("## 3. 相机位姿误差（平均角度误差，越小越好）\n")
    lines.append("| Model | tgt_e_R | tgt_e_t | tgt_e_pose | cxt_e_R | cxt_e_t | cxt_e_pose |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, desc, _ in MODELS:
        r = results.get(key)
        b = _get_block(r, "all")
        lines.append(
            f"| `{key}` | {fmt(b.get('tgt_e_R'), 3)} | {fmt(b.get('tgt_e_t'), 3)} "
            f"| {fmt(b.get('tgt_e_pose'), 3)} | {fmt(b.get('cxt_e_R'), 3)} "
            f"| {fmt(b.get('cxt_e_t'), 3)} | {fmt(b.get('cxt_e_pose'), 3)} |"
        )

    # === 相机位姿 AUC ===
    lines.append("\n## 4. 相机位姿 AUC（@5° / @10° / @20°，越高越好）\n")
    for metric in ["R", "t", "pose"]:
        label = {"R": "Rotation", "t": "Translation", "pose": "Pose (R+t)"}[metric]
        lines.append(f"### {label} — `{metric}`\n")
        lines.append("| Model | AUC@5° | AUC@10° | AUC@20° | Median err. |")
        lines.append("|---|---|---|---|---|")
        for key, desc, _ in MODELS:
            r = results.get(key)
            pa = _get_pose(r, "all", metric)
            if pa:
                auc = pa["auc_5_10_20"]
                lines.append(
                    f"| `{key}` | {fmt(auc[0])} | {fmt(auc[1])} | {fmt(auc[2])} "
                    f"| {fmt(pa['median'], 4)} |"
                )
            else:
                lines.append(f"| `{key}` | — | — | — | — |")
        lines.append("")

    # === 推理开销 ===
    lines.append("## 5. 推理开销\n")
    lines.append("| Model | Encoder 平均耗时 | Decoder 平均耗时 |")
    lines.append("|---|---|---|")
    for key, desc, _ in MODELS:
        r = results.get(key) or {}
        bench = r.get("bench") or {}
        def _fmt_bench(stage: str) -> str:
            x = bench.get(stage)
            if isinstance(x, (int, float)):
                return f"{x:.4f}s"
            if isinstance(x, dict):
                avg = x.get("avg")
                n = x.get("num_calls") or x.get("count")
                if avg is not None:
                    return f"{avg:.4f}s ({n} calls)" if n else f"{avg:.4f}s"
                if "total_time" in x and "num_calls" in x:
                    a = x["total_time"] / max(x["num_calls"], 1)
                    return f"{a:.4f}s ({x['num_calls']} calls)"
            return "—"
        lines.append(f"| `{key}` | {_fmt_bench('encoder')} | {_fmt_bench('decoder')} |")

    # === 输出路径 ===
    lines.append("\n## 6. 结果文件位置\n")
    for key, desc, ckpt in MODELS:
        r = results.get(key) or {}
        d = r.get("model_dir")
        status = "✓ 已完成" if r.get("overlap") else "✗ 未完成或解析失败"
        lines.append(f"- `{key}` ({desc}) — **{status}**")
        lines.append(f"  - checkpoint: `checkpoints/{ckpt}`")
        if d:
            lines.append(f"  - 输出目录: `{d}`")
            if r.get("avg_path"):
                lines.append(f"  - 平均指标: `{r['avg_path']}`")
            if r.get("sub_path"):
                lines.append(f"  - 分档指标: `{r['sub_path']}`")
            if r.get("log"):
                lines.append(f"  - 日志: `{r['log']}`")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ 汇总已写入 {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/eval"),
                    help="评测输出根目录 (默认 outputs/eval)")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出 md 路径 (默认 <root>/summary.md)")
    args = ap.parse_args()

    root: Path = args.root
    out: Path = args.out or root / "summary.md"

    if not root.exists():
        print(f"[ERROR] 根目录不存在: {root}", file=sys.stderr)
        sys.exit(2)

    print(f"扫描: {root.resolve()}")
    results: dict[str, dict[str, Any]] = {}
    for key, desc, ckpt in MODELS:
        art = find_artifacts(root, key)
        log_path = art.get("log")
        log_text = log_path.read_text(errors="ignore") if log_path and log_path.exists() else ""
        ok = has_completed(log_text) if log_text else False

        # 解析 overlap 分档 + pose AUC + bench
        overlap = parse_overlap_block(log_text) if log_text else None
        pose_auc = parse_pose_auc(log_text) if log_text else None
        log_bench = parse_bench_from_log(log_text) if log_text else {}

        art["overlap"] = overlap
        art["pose_auc"] = pose_auc
        # bench 优先用 benchmark.json，其次 log
        if not art.get("bench") and log_bench:
            art["bench"] = log_bench
        results[key] = art

        print(f"  [{key:<13}] completed={ok}  overlap={'✓' if overlap else '—'}  "
              f"pose_auc={'✓' if pose_auc else '—'}  bench={'✓' if art.get('bench') else '—'}  "
              f"log={log_path}")

    # 也把结构化数据 dump 一份 JSON
    json_out = out.with_suffix(".json")
    clean = {}
    for k, v in results.items():
        clean[k] = {
            "overlap": v.get("overlap"),
            "pose_auc": v.get("pose_auc"),
            "model_dir": str(v.get("model_dir")) if v.get("model_dir") else None,
            "avg": v.get("avg"),
            "sub": v.get("sub"),
        }
    with open(json_out, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"✓ 结构化 JSON: {json_out}")

    write_md(results, out)


if __name__ == "__main__":
    main()

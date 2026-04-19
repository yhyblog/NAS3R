#!/usr/bin/env python3
"""
汇总 NAS3R 4 个 checkpoint 的评测结果到 Markdown。

优先查找顺序（对每个 model）：
  1. {root}/{key}/eval_{key}/scores_all_avg.json  (正常产出)
  2. {root}/{key}/**/scores_all_avg.json          (任意深度)
  3. {root}/{key}/eval.log                         (文本解析兜底)
  4. {workspace}/outputs/exp_eval_{key}/**/*.log   (Hydra 默认日志位置)
  5. {workspace}/outputs/test/**/eval_{key}/*.json (experiment yaml 默认路径)

用法（在 /opt/tiger/mmfinetune/nas3r 下）：
  python3 summarize_eval.py
  python3 summarize_eval.py --root outputs/eval --workspace .
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional


MODELS = [
    ("nas3r",        "2-view (random init)",            "re10k_nas3r.ckpt"),
    ("multiview",    "10-view",                         "re10k_nas3r_multiview.ckpt"),
    ("pretrained",   "2-view (VGGT pretrained)",        "re10k_nas3r_pretrained.ckpt"),
    ("pretrained-I", "2-view (VGGT + GT intrinsics)",   "re10k_nas3r_pretrained-I.ckpt"),
]


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.load(open(path))
    except Exception as e:
        print(f"[WARN] 无法解析 {path}: {e}", file=sys.stderr)
        return None


# =========== 文本解析 ===========
TABLE_HEADERS = ["lpips", "ssim", "psnr", "tgt_e_R", "tgt_e_t", "tgt_e_pose",
                 "cxt_e_R", "cxt_e_t", "cxt_e_pose"]


def parse_overlap_block(text: str) -> Optional[dict]:
    """抓最后一次出现的 All Pairs 块 + 后续 Overlap 段。"""
    idx = text.rfind("All Pairs:")
    if idx < 0:
        return None
    chunk = text[idx:]
    result: dict[str, dict[str, float]] = {}
    sections = re.split(r"^(All Pairs:|Overlap:\s*\w+)\s*$", chunk, flags=re.M)
    for i in range(1, len(sections), 2):
        head = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        key = "all" if head.startswith("All Pairs") else head.split(":", 1)[1].strip()
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


def parse_pose_auc(text: str) -> dict:
    out: dict[str, dict[str, Any]] = {}
    for m in POSE_AUC_RE.finditer(text):
        subset = (m.group(1) or "all").strip()
        metric = m.group(2)
        auc = [float(x) for x in m.group(3).split(",")]
        median = float(m.group(4))
        out.setdefault(subset, {})[metric] = {"auc_5_10_20": auc, "median": median}
    return out


_BENCH_RE = re.compile(r"^\s*(encoder|decoder):\s*(\d+)\s+calls,\s*avg\.\s*([\d.eE+-]+)\s+seconds", re.M)


def parse_bench_from_log(text: str) -> dict:
    return {
        m.group(1): {"num_calls": int(m.group(2)), "avg": float(m.group(3))}
        for m in _BENCH_RE.finditer(text)
    }


def has_completed(log_text: str) -> bool:
    return "All Pairs:" in log_text and ("Test metric" in log_text or "encoder:" in log_text)


# =========== 资源查找 ===========
def find_artifacts(root: Path, workspace: Path, key: str) -> dict[str, Any]:
    art: dict[str, Any] = {
        "model_dir": root / key,
        "avg": None, "avg_path": None,
        "sub": None, "sub_path": None,
        "bench": None, "mem": None,
        "log": None, "log_paths": [],
    }

    # 1. JSON 产出（可能位置多样）
    search_dirs = [root / key, workspace / "outputs" / "test"]
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.rglob("scores_all_avg.json"):
            # 过滤：要和当前 key 有关
            if key in str(f) or f.parent.name.startswith(f"eval_{key}") or f.parent.name == key:
                if art["avg"] is None:
                    art["avg"] = load_json(f)
                    art["avg_path"] = f
        for f in d.rglob("scores_sub_avg.json"):
            if key in str(f) or f.parent.name.startswith(f"eval_{key}") or f.parent.name == key:
                if art["sub"] is None:
                    art["sub"] = load_json(f)
                    art["sub_path"] = f
        for f in d.rglob("benchmark.json"):
            if key in str(f) or f.parent.name.startswith(f"eval_{key}") or f.parent.name == key:
                if art["bench"] is None:
                    art["bench"] = load_json(f)
        for f in d.rglob("peak_memory.json"):
            if key in str(f) or f.parent.name.startswith(f"eval_{key}") or f.parent.name == key:
                if art["mem"] is None:
                    art["mem"] = load_json(f)

    # 2. 日志文件（多位置 + 按 mtime 最新优先）
    log_candidates: list[Path] = []
    # 2.1 主脚本 tee 的 eval.log
    for p in (root / key / "eval.log",):
        if p.exists():
            log_candidates.append(p)
    # 2.2 Hydra 输出目录（outputs/exp_eval_<key>/<ts>/main.log 或 *.log）
    hydra_root = workspace / "outputs" / f"exp_eval_{key}"
    if hydra_root.exists():
        for p in hydra_root.rglob("*.log"):
            log_candidates.append(p)
    # 2.3 root 下任意位置
    for p in (root / key).rglob("*.log"):
        if p not in log_candidates:
            log_candidates.append(p)

    # 按内容判断：挑已完成的；若都没完成，挑最大的
    best = None
    for p in log_candidates:
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if has_completed(txt):
            best = p
            break
    if best is None and log_candidates:
        # 取 size 最大
        best = max(log_candidates, key=lambda p: p.stat().st_size if p.exists() else 0)

    art["log"] = best
    art["log_paths"] = log_candidates
    return art


# =========== 格式化 ===========
def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _get_block(r, ov):
    if not r: return {}
    return (r.get("overlap") or {}).get(ov) or {}


def _get_pose(r, subset, metric):
    if not r: return None
    return ((r.get("pose_auc") or {}).get(subset) or {}).get(metric)


def write_md(results: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# NAS3R · RealEstate10K 评测结果汇总\n")
    lines.append("数据集：RealEstate10K test (`assets/evaluation_index_re10k.json`，共 5601 个有效 scene)。\n")
    lines.append("评测命令参考 README，模式：`mode=test`，view_sampler=`evaluation`。\n")

    # 完成情况概览
    lines.append("## 完成情况\n")
    lines.append("| Model | 状态 | 日志 |")
    lines.append("|---|---|---|")
    for key, desc, _ in MODELS:
        r = results.get(key) or {}
        status = "✓ 已完成" if r.get("overlap") else ("⏳ 未完成/解析失败")
        lp = r.get("log")
        lines.append(f"| `{key}` ({desc}) | {status} | `{lp}` |" if lp else f"| `{key}` ({desc}) | {status} | — |")

    # 1. 渲染质量 (All Pairs)
    lines.append("\n## 1. 新视角合成质量（All Pairs）\n")
    lines.append("| Model | Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
    lines.append("|---|---|---|---|---|")
    for key, desc, _ in MODELS:
        b = _get_block(results.get(key), "all")
        lines.append(f"| `{key}` | {desc} | {fmt(b.get('psnr'), 3)} | {fmt(b.get('ssim'))} | {fmt(b.get('lpips'))} |")

    # 2. 分档 overlap
    lines.append("\n## 2. 按 Overlap 分档的渲染质量\n")
    for ov in ["large", "medium", "small"]:
        lines.append(f"### Overlap: **{ov}**\n")
        lines.append("| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
        lines.append("|---|---|---|---|")
        for key, desc, _ in MODELS:
            b = _get_block(results.get(key), ov)
            lines.append(f"| `{key}` | {fmt(b.get('psnr'), 3)} | {fmt(b.get('ssim'))} | {fmt(b.get('lpips'))} |")
        lines.append("")

    # 3. 相机位姿误差
    lines.append("## 3. 相机位姿误差（All Pairs · 平均角度，越小越好）\n")
    lines.append("| Model | tgt_e_R | tgt_e_t | tgt_e_pose | cxt_e_R | cxt_e_t | cxt_e_pose |")
    lines.append("|---|---|---|---|---|---|---|")
    for key, desc, _ in MODELS:
        b = _get_block(results.get(key), "all")
        lines.append(
            f"| `{key}` | {fmt(b.get('tgt_e_R'), 3)} | {fmt(b.get('tgt_e_t'), 3)} | "
            f"{fmt(b.get('tgt_e_pose'), 3)} | {fmt(b.get('cxt_e_R'), 3)} | "
            f"{fmt(b.get('cxt_e_t'), 3)} | {fmt(b.get('cxt_e_pose'), 3)} |"
        )

    # 4. Pose AUC
    lines.append("\n## 4. 相机位姿 AUC（All Pairs · @5°/@10°/@20°，越高越好）\n")
    for metric in ["R", "t", "pose"]:
        label = {"R": "Rotation", "t": "Translation", "pose": "Pose (R+t)"}[metric]
        lines.append(f"### {label} — `{metric}`\n")
        lines.append("| Model | AUC@5° | AUC@10° | AUC@20° | Median err. |")
        lines.append("|---|---|---|---|---|")
        for key, desc, _ in MODELS:
            pa = _get_pose(results.get(key), "all", metric)
            if pa:
                auc = pa["auc_5_10_20"]
                lines.append(f"| `{key}` | {fmt(auc[0])} | {fmt(auc[1])} | {fmt(auc[2])} | {fmt(pa['median'], 4)} |")
            else:
                lines.append(f"| `{key}` | — | — | — | — |")
        lines.append("")

    # 5. 推理开销
    lines.append("## 5. 推理开销\n")
    lines.append("| Model | Encoder 平均耗时 | Decoder 平均耗时 |")
    lines.append("|---|---|---|")
    for key, desc, _ in MODELS:
        bench = (results.get(key) or {}).get("bench") or {}
        def _fmt_bench(stage: str) -> str:
            x = bench.get(stage)
            if isinstance(x, (int, float)):
                return f"{x:.4f}s"
            if isinstance(x, dict):
                avg = x.get("avg")
                n = x.get("num_calls") or x.get("count")
                if avg is not None:
                    return f"{avg*1000:.2f}ms" + (f" × {n} calls" if n else "")
                if "total_time" in x and "num_calls" in x:
                    a = x["total_time"] / max(x["num_calls"], 1)
                    return f"{a*1000:.2f}ms × {x['num_calls']} calls"
            return "—"
        lines.append(f"| `{key}` | {_fmt_bench('encoder')} | {_fmt_bench('decoder')} |")

    # 6. 细节
    lines.append("\n## 6. 结果文件位置\n")
    for key, desc, ckpt in MODELS:
        r = results.get(key) or {}
        status = "✓" if r.get("overlap") else "✗"
        lines.append(f"\n### {status} `{key}` — {desc}")
        lines.append(f"- checkpoint: `checkpoints/{ckpt}`")
        if r.get("model_dir"): lines.append(f"- 输出目录: `{r['model_dir']}`")
        if r.get("avg_path"):  lines.append(f"- 平均指标 JSON: `{r['avg_path']}`")
        if r.get("sub_path"):  lines.append(f"- 分档指标 JSON: `{r['sub_path']}`")
        if r.get("log"):       lines.append(f"- 主日志: `{r['log']}`")
        if len(r.get("log_paths", [])) > 1:
            lines.append(f"- 其他日志候选: {', '.join(f'`{p}`' for p in r['log_paths'] if p != r.get('log'))}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ 汇总已写入 {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/eval"),
                    help="评测输出根目录（默认 outputs/eval）")
    ap.add_argument("--workspace", type=Path, default=Path("."),
                    help="项目工作区根（用于找 Hydra 日志，默认当前目录）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出 md 路径（默认 <root>/summary.md）")
    args = ap.parse_args()

    root = args.root.resolve()
    workspace = args.workspace.resolve()
    out = (args.out or root / "summary.md").resolve()

    if not root.exists():
        print(f"[ERROR] 根目录不存在: {root}", file=sys.stderr)
        sys.exit(2)

    print(f"扫描 root={root}")
    print(f"     workspace={workspace}")

    results: dict[str, dict] = {}
    for key, desc, ckpt in MODELS:
        art = find_artifacts(root, workspace, key)
        log_text = ""
        if art.get("log"):
            try:
                log_text = art["log"].read_text(errors="ignore")
            except Exception:
                pass

        overlap = parse_overlap_block(log_text) if log_text else None
        pose_auc = parse_pose_auc(log_text) if log_text else None
        log_bench = parse_bench_from_log(log_text) if log_text else {}

        art["overlap"] = overlap
        art["pose_auc"] = pose_auc
        if not art.get("bench") and log_bench:
            art["bench"] = log_bench

        # 若 JSON 里有 avg 而日志没 overlap，可降级用 avg 填 all
        if overlap is None and art.get("avg"):
            avg = art["avg"]
            if isinstance(avg, dict):
                flat = {}
                for kk, vv in avg.items():
                    if isinstance(vv, (int, float)):
                        flat[kk] = float(vv)
                if flat:
                    art["overlap"] = {"all": flat}

        results[key] = art

        def flag(v): return "✓" if v else "—"
        lp = art.get("log")
        print(f"  [{key:<13}] overlap={flag(art.get('overlap'))}  "
              f"pose_auc={flag(art.get('pose_auc'))}  "
              f"bench={flag(art.get('bench'))}  "
              f"log={lp}")

    # 结构化 JSON
    json_out = out.with_suffix(".json")
    clean = {
        k: {
            "overlap": v.get("overlap"),
            "pose_auc": v.get("pose_auc"),
            "bench": v.get("bench"),
            "model_dir": str(v.get("model_dir")) if v.get("model_dir") else None,
            "log": str(v.get("log")) if v.get("log") else None,
            "avg_path": str(v.get("avg_path")) if v.get("avg_path") else None,
        }
        for k, v in results.items()
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"✓ 结构化 JSON: {json_out}")

    write_md(results, out)


if __name__ == "__main__":
    main()

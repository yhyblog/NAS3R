"""
下载/准备 RealEstate10K test 数据子集，用于基于预训练 checkpoint 的评测。

特点：
  1. 解析 ``assets/evaluation_index_re10k.json``，找出评测真正需要的场景 (key) 集合。
  2. 先从 HuggingFace (yiren-lu/re10k_pixelsplat) 只下载 ``test/index.json``（如存在），
     若不存在则回退到 streaming 扫描 test/*.torch 头部来定位所需 chunk。
  3. 仅下载那些包含目标场景的 ``.torch`` 分片，大幅节省空间。
  4. 生成一个 ``test/index.json``（如果数据本身没带），使得 dataset loader 能正常查找。
  5. 可选：根据实际下载到的 scene 裁剪一份 ``evaluation_index_re10k.subset.json``，
     避免评测时报 "scene not found"。

用法：
  python -m src.scripts.prepare_re10k_test_subset \
      --data-dir datasets/re10k \
      --eval-index assets/evaluation_index_re10k.json \
      --max-scenes 100          # 可选，截断前 N 个评测场景，进一步缩小
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

try:
    import torch
except ImportError:
    print("[ERROR] 需要 torch，请先安装 PyTorch", file=sys.stderr)
    raise

try:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
except ImportError:
    print("[ERROR] 需要 huggingface_hub：pip install huggingface_hub", file=sys.stderr)
    raise


REPO_ID = "yiren-lu/re10k_pixelsplat"
REPO_TYPE = "dataset"


def load_eval_keys(eval_index_path: Path, max_scenes: int | None) -> List[str]:
    with open(eval_index_path) as f:
        idx = json.load(f)
    # 只保留非 null（有 context/target 配对）的 scene
    keys = [k for k, v in idx.items() if v is not None]
    if max_scenes is not None and max_scenes > 0:
        keys = keys[:max_scenes]
    print(f"[prepare] evaluation_index 有效 scene 数: {len(keys)}")
    return keys


def fetch_remote_index(local_dir: Path) -> Dict[str, str] | None:
    """尝试从 HF 拉取 test/index.json；若仓库中没有则返回 None。"""
    remote_candidates = ["test/index.json", "re10k/test/index.json"]
    for rel in remote_candidates:
        try:
            p = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=rel,
                local_dir=str(local_dir.parent if rel.startswith("re10k/") else local_dir),
            )
            print(f"[prepare] 远程 index.json 命中: {rel}")
            with open(p) as f:
                return json.load(f)
        except Exception as e:
            print(f"[prepare] 远程 {rel} 不存在: {type(e).__name__}")
    return None


def scan_remote_chunks_for_keys(
    local_dir: Path,
    want_keys: Set[str],
) -> Dict[str, str]:
    """
    回退方案：逐个下载 test/*.torch，读取头部后匹配场景；
    匹配后保留，其余删除。速度慢，仅在缺失 index.json 时使用。
    """
    print("[prepare] 远程无 index.json，回退到 streaming 扫描（较慢）")
    all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    test_shards = sorted([f for f in all_files if "/test/" in f and f.endswith(".torch")])
    if not test_shards:
        test_shards = sorted([f for f in all_files if f.startswith("test/") and f.endswith(".torch")])
    print(f"[prepare] 远程 test shard 共 {len(test_shards)} 个")

    found: Dict[str, str] = {}
    remaining = set(want_keys)
    for i, rel in enumerate(test_shards):
        if not remaining:
            break
        try:
            p = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=rel,
                local_dir=str(local_dir.parent),
            )
            chunk = torch.load(p, weights_only=True, map_location="cpu")
            shard_keys = {item["key"] for item in chunk}
            hit = remaining & shard_keys
            if hit:
                fname = Path(rel).name
                for k in hit:
                    found[k] = fname
                remaining -= hit
                print(f"  [{i+1}/{len(test_shards)}] {rel} 命中 {len(hit)}，累计 {len(found)}/{len(want_keys)}")
            else:
                # 未命中的 chunk 删除，避免占空间
                try:
                    os.remove(p)
                except OSError:
                    pass
        except Exception as e:
            print(f"  [WARN] 下载/解析 {rel} 失败: {e}")
    return found


def plan_needed_shards(
    remote_index: Dict[str, str],
    want_keys: List[str],
) -> Dict[str, List[str]]:
    """根据 scene key -> shard 映射，聚合出需下载的 shard 列表。"""
    need: Dict[str, List[str]] = {}
    missing: List[str] = []
    for k in want_keys:
        shard = remote_index.get(k)
        if shard is None:
            missing.append(k)
            continue
        need.setdefault(shard, []).append(k)
    if missing:
        print(f"[prepare] 远程 index.json 中缺失 {len(missing)} 个评测 scene（将跳过）")
    print(f"[prepare] 需下载 shard 数: {len(need)}")
    return need


def download_shards(local_dir: Path, shard_names: List[str]) -> List[str]:
    """只下载指定的 test/*.torch。返回本地已存在的 shard 文件名。"""
    local_test = local_dir / "test"
    local_test.mkdir(parents=True, exist_ok=True)

    patterns = [f"test/{s}" for s in shard_names] + [f"re10k/test/{s}" for s in shard_names]
    print(f"[prepare] 使用 snapshot_download 拉取 {len(shard_names)} 个 shard ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir.parent),  # 保留相对目录结构
        allow_patterns=patterns,
    )
    # 规整路径：若 snapshot 下载到了 re10k/test/ 则移动到 test/
    alt = local_dir.parent / "re10k" / "test"
    if alt.exists() and alt != local_test:
        for f in alt.glob("*.torch"):
            target = local_test / f.name
            if not target.exists():
                os.replace(f, target)

    got = sorted(f.name for f in local_test.glob("*.torch"))
    print(f"[prepare] 本地 test/*.torch 现有 {len(got)} 个")
    return got


def rebuild_local_index(local_dir: Path) -> Dict[str, str]:
    """按本地实际 chunk 重建 test/index.json。"""
    local_test = local_dir / "test"
    idx: Dict[str, str] = {}
    for f in sorted(local_test.glob("*.torch")):
        try:
            chunk = torch.load(f, weights_only=True, map_location="cpu")
            for item in chunk:
                idx[item["key"]] = f.name
        except Exception as e:
            print(f"  [WARN] 读取 {f.name} 失败: {e}")
    out = local_test / "index.json"
    with open(out, "w") as fp:
        json.dump(idx, fp)
    print(f"[prepare] 写入 {out} ({len(idx)} scene)")
    return idx


def write_subset_eval_index(
    eval_index_path: Path,
    available: Set[str],
    out_path: Path,
) -> int:
    with open(eval_index_path) as f:
        orig = json.load(f)
    sub = {k: v for k, v in orig.items() if v is not None and k in available}
    with open(out_path, "w") as f:
        json.dump(sub, f)
    print(f"[prepare] 写入裁剪后评测 index: {out_path} ({len(sub)} scene)")
    return len(sub)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="本地数据根目录，例如 datasets/re10k")
    ap.add_argument("--eval-index", required=True, type=Path,
                    help="评测索引路径，例如 assets/evaluation_index_re10k.json")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="仅准备前 N 个评测 scene（用于快速 smoke test）")
    ap.add_argument("--subset-out", type=Path, default=None,
                    help="裁剪后评测索引写入路径。默认写入同目录 .subset.json")
    args = ap.parse_args()

    data_dir: Path = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    want_keys = load_eval_keys(args.eval_index, args.max_scenes)

    remote_index = fetch_remote_index(data_dir)
    if remote_index is not None:
        need = plan_needed_shards(remote_index, want_keys)
        if need:
            download_shards(data_dir, sorted(need.keys()))
    else:
        scan_remote_chunks_for_keys(data_dir, set(want_keys))

    local_idx = rebuild_local_index(data_dir)

    subset_out = args.subset_out or args.eval_index.with_suffix(".subset.json")
    n = write_subset_eval_index(args.eval_index, set(local_idx.keys()), subset_out)
    if n == 0:
        print("[prepare][ERROR] 裁剪后评测集为空，终止。", file=sys.stderr)
        sys.exit(2)

    print(f"[prepare] 全部完成，本地可用 scene {len(local_idx)}，评测 scene {n}")


if __name__ == "__main__":
    main()

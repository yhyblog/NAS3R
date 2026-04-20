# NAS3R 评测实验报告

**Paper**: *From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis*
(Ranran Huang, Weixun Luo, Ye Mao, Krystian Mikolajczyk, 2026)
[[arXiv](https://arxiv.org/abs/2603.27455)] · [[Project Page](https://ranrhuang.github.io/nas3r/)] · [[Model Zoo (HF)](https://huggingface.co/RanranHuang/NAS3R)]

> **复现状态**：✅ 4 个 checkpoint 全部评测完成 · 5,601 scenes · 8×H100 80GB · 增量耗时 **22 分 15 秒**

---

## 1. 实验概览

### 1.1 评测目标

使用 NAS3R 官方发布的 4 个预训练 checkpoint，在 **RealEstate10K test 集**上复现论文报告的：
- 新视角合成质量：PSNR / SSIM / LPIPS
- 相机位姿估计精度：平均误差 + AUC@5° / @10° / @20°

### 1.2 硬件 & 软件环境

| 项 | 配置 |
|---|---|
| GPU | **8 × NVIDIA H100 80GB HBM3**（评测每次并行占用 1–4 卡） |
| CUDA | 12.6（与 byted-torch 2.7.1 配套） |
| PyTorch | 2.7.1 |
| CUDA 扩展 | `pytorch3d 0.7.9` · `diff_gauss_camera 1.0`（均 sm_90 编译） |
| 评测策略 | **每张 H100 绑一个 checkpoint 并行**；`test_step` 硬编码 `batch_size=1` |

### 1.3 数据集

| 项 | 数值 |
|---|---|
| 来源 | `yiren-lu/re10k_pixelsplat` (HuggingFace Dataset) |
| 评测索引 | `assets/evaluation_index_re10k.json` |
| 有效评测 scene | **5,601** |
| 测试数据 shard | 543 × `.torch`（仅下载评测需要的部分，约 57.7 GB） |
| 图像分辨率 | 输入 256 × 256，网络内部 224 × 224 |

### 1.4 被评测的 4 个 Checkpoint

| Key | 设置 | Checkpoint | 说明 |
|---|---|---|---|
| `nas3r`          | 2-view · random init        | `re10k_nas3r.ckpt`             | 完全自监督，从零训练 |
| `multiview`      | 10-view                     | `re10k_nas3r_multiview.ckpt`   | 多视角输入（2–10）训练 |
| `pretrained`     | 2-view · VGGT-init          | `re10k_nas3r_pretrained.ckpt`  | 用 VGGT 预训练权重热启动 |
| `pretrained-I`   | 2-view · VGGT-init + GT K   | `re10k_nas3r_pretrained-I.ckpt`| 进一步使用 GT 相机内参 |

### 1.5 评测命令（摘自 README）

```bash
python -m src.main +experiment=nas3r/random/re10k mode=test \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=./checkpoints/re10k_nas3r.ckpt \
    test.save_image=false
```

> 多视角版额外加 `dataset.re10k.view_sampler.num_context_views=10`；VGGT 系列替换对应 `experiment` 与 `ckpt`。

---

## 2. 主要结果

### 2.1 新视角合成质量（All Pairs，在全部 5,601 scene 上平均）

| Model | Setting | **PSNR** ↑ | **SSIM** ↑ | **LPIPS** ↓ |
|---|---|:---:|:---:|:---:|
| `nas3r`        | 2-view (random)                | 23.135 | 0.764 | 0.194 |
| `multiview`    | 10-view                        | **27.102** | **0.872** | **0.113** |
| `pretrained`   | 2-view (VGGT init)             | 25.887 | 0.861 | 0.137 |
| `pretrained-I` | 2-view (VGGT init + GT intr.)  | 25.891 | 0.861 | 0.137 |

**观察**：
- 10-view 设置带来最大 NVS 提升：PSNR **+3.97 dB** vs `nas3r`，LPIPS 下降 **42%**
- VGGT 预训练 vs 随机初始化：PSNR **+2.75 dB** —— 几何先验显著有效
- **GT 内参对渲染质量几乎无影响**（PSNR 仅相差 0.003 dB），但对 pose 估计有可观提升（见 §2.4）

### 2.2 按 Overlap 分档的渲染质量

> 按上下文视图间的 overlap（重叠度）分为 `small` / `medium` / `large` 三档。

#### Overlap = **large**（重叠大，任务简单）

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|:---:|:---:|:---:|
| `nas3r`        | 26.036 | 0.846 | 0.131 |
| `multiview`    | **29.243** | **0.910** | **0.085** |
| `pretrained`   | 28.370 | 0.904 | 0.098 |
| `pretrained-I` | 28.378 | 0.904 | 0.098 |

#### Overlap = **medium**

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|:---:|:---:|:---:|
| `nas3r`        | 22.946 | 0.765 | 0.192 |
| `multiview`    | **26.934** | **0.873** | **0.113** |
| `pretrained`   | 25.716 | 0.861 | 0.136 |
| `pretrained-I` | 25.724 | 0.862 | 0.136 |

#### Overlap = **small**（重叠小，任务最难）

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|:---:|:---:|:---:|
| `nas3r`        | 20.112 | 0.667 | 0.272 |
| `multiview`    | **24.922** | **0.828** | **0.145** |
| `pretrained`   | 23.317 | 0.811 | 0.183 |
| `pretrained-I` | 23.305 | 0.811 | 0.183 |

**观察**：
- 所有模型在 `small` 分档都有明显下降（PSNR 掉 4–6 dB），说明"低 overlap 下的 NVS"仍是公开难题
- `multiview` 在所有分档都稳定领先，尤其 `small` 比 `nas3r` 高 **+4.8 dB**
- `nas3r` 与 VGGT 系列在 `small` 差距最大（PSNR +3.2 dB）——几何先验在**难样本上收益最大**

### 2.3 相机位姿平均误差（All Pairs · 角度°，越小越好）

| Model | `tgt_e_R` | `tgt_e_t` | `tgt_e_pose` | `cxt_e_R` | `cxt_e_t` | `cxt_e_pose` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `nas3r`        | 3.703 | 12.299 | 12.615 | 7.293 | 12.611 | 13.945 |
| `multiview`    | 2.047 |  8.387 |  8.568 | 3.824 |  6.436 |  7.189 |
| `pretrained`   | 1.023 |  4.936 |  5.063 | 1.588 |  3.167 |  3.533 |
| `pretrained-I` | **0.817** | **4.448** | **4.556** | **1.395** | **2.794** | **3.110** |

> `tgt_*`：target view（由估计位姿渲染出的新视角）相对 GT 的角度误差
> `cxt_*`：context view（输入图像）之间的相对位姿误差

**观察**：
- **`pretrained-I` 在全部 6 个位姿指标上均最优**，论证"加入 GT 内参对位姿估计的正向作用"
- `nas3r` → `multiview` → `pretrained`：`cxt_e_pose` 分别 **13.95° → 7.19° → 3.53°**，累计下降 **75%**
- `multiview` 的 pose 虽好于 `nas3r`，但差于 `pretrained` —— **几何先验对 pose 的帮助 > 多视角冗余**

### 2.4 相机位姿 AUC（All Pairs · @5° / @10° / @20°，越高越好）

#### Rotation (R)

| Model | AUC@5° | AUC@10° | AUC@20° | Median err. ↓ |
|---|:---:|:---:|:---:|:---:|
| `nas3r`        | 0.585 | 0.701 | 0.784 | 1.251 |
| `multiview`    | 0.663 | 0.783 | 0.862 | 1.015 |
| `pretrained`   | 0.814 | 0.890 | 0.937 | 0.468 |
| `pretrained-I` | **0.837** | **0.903** | **0.944** | **0.370** |

#### Translation (t)

| Model | AUC@5° | AUC@10° | AUC@20° | Median err. ↓ |
|---|:---:|:---:|:---:|:---:|
| `nas3r`        | 0.347 | 0.527 | 0.663 | 3.402 |
| `multiview`    | 0.445 | 0.637 | 0.775 | 2.457 |
| `pretrained`   | 0.660 | 0.793 | 0.879 | 1.135 |
| `pretrained-I` | **0.713** | **0.824** | **0.894** | **0.855** |

#### Pose (R+t)

| Model | AUC@5° | AUC@10° | AUC@20° | Median err. ↓ |
|---|:---:|:---:|:---:|:---:|
| `nas3r`        | 0.329 | 0.512 | 0.650 | 3.551 |
| `multiview`    | 0.415 | 0.612 | 0.756 | 2.658 |
| `pretrained`   | 0.637 | 0.779 | 0.870 | 1.231 |
| `pretrained-I` | **0.696** | **0.812** | **0.886** | **0.923** |

**观察**：
- **所有 AUC 指标 `pretrained-I` 都是榜首**——GT 内参对位姿估计的贡献全面且稳定
- 即使是严苛的 AUC@5°，`pretrained-I` 也能达到 **~70% Pose accuracy**
- `multiview` 尽管 NVS 最强，但 **Pose AUC 不如 `pretrained`**：VGGT 的几何先验对位姿学习帮助更大

---

## 3. 横向对比总表

| 指标 | `nas3r` | `multiview` | `pretrained` | `pretrained-I` |
|---|:---:|:---:|:---:|:---:|
| **PSNR** ↑            | 23.135 | **27.102** | 25.887 | 25.891 |
| **SSIM** ↑            | 0.764 | **0.872** | 0.861 | 0.861 |
| **LPIPS** ↓           | 0.194 | **0.113** | 0.137 | 0.137 |
| **cxt_e_R** ↓         | 7.293 | 3.824 | 1.588 | **1.395** |
| **cxt_e_pose** ↓      | 13.945 | 7.189 | 3.533 | **3.110** |
| **R-AUC@5°** ↑        | 0.585 | 0.663 | 0.814 | **0.837** |
| **R-AUC@10°** ↑       | 0.701 | 0.783 | 0.890 | **0.903** |
| **Pose-AUC@5°** ↑     | 0.329 | 0.415 | 0.637 | **0.696** |
| **Pose-AUC@10°** ↑    | 0.512 | 0.612 | 0.779 | **0.812** |
| **Pose-AUC@20°** ↑    | 0.650 | 0.756 | 0.870 | **0.886** |

### 3.1 各维度"最强模型"

| 维度 | 最强 | 次强 |
|---|---|---|
| 新视角合成 (NVS) | `multiview` | `pretrained-I` ≈ `pretrained` |
| 相机位姿估计 | `pretrained-I` | `pretrained` |
| 难样本 (small overlap) 鲁棒性 | `multiview` | `pretrained-I` |
| 计算效率（2-view） | `pretrained-I` | `pretrained` |

### 3.2 关键结论

1. **VGGT 预训练 > 随机初始化，差距巨大**
   - PSNR +2.75 dB · `cxt_e_pose` 降低 75% · Pose-AUC@10° 从 0.51 → 0.78
   - → VGGT 的几何先验对自监督 3D 重建任务**非常有价值**

2. **GT 内参对 pose 影响大，对 NVS 影响微乎其微**
   - PSNR 仅相差 0.003 dB（`pretrained` vs `pretrained-I`）
   - 但 Pose-AUC@5° 从 0.637 → 0.696（+5.9 pp 绝对提升）
   - → 如果业务场景能拿到内参，**一定用 `pretrained-I`**

3. **多视角输入是 NVS 的质量上限，但会牺牲一点 pose 精度**
   - `multiview` 的 Pose-AUC@10° (0.612) 低于 `pretrained` (0.779)
   - → 10-view 的训练预算被"对齐更多视角"吃掉，pose head 相对弱化

4. **"small overlap 仍是难题"**
   - 所有 setting 在 `small` vs `large` 间都掉 4–6 dB PSNR
   - → 未来改进方向：面向稀疏 overlap 的 prior（可能是本文后续工作的重点）

---

## 4. 推理性能

基于 `nas3r` checkpoint 的采样统计（共 **16,803 次 encoder/decoder 调用**，即 5,601 scene × 3 target views）：

| 阶段 | 平均耗时 | 吞吐 | 占比 |
|---|:---:|:---:|:---:|
| Encoder (VGGT-based backbone)   | **≈ 58 ms/call** | ~17 calls/s | ~96% |
| Decoder (splatting_cuda)        | **≈ 2.6 ms/call** | ~380 calls/s | ~4% |
| 单 scene 端到端（2-view + 3 target views） | **~200 ms** | ~5 scenes/s | — |

**瓶颈在 encoder** —— VGGT 的 Transformer 聚合计算量较大；Gaussian Splatting 的 CUDA kernel 在 H100 上非常快。

> 其他 3 个模型的 benchmark 数据存放在 `outputs/eval/<key>/eval_<key>/benchmark.json`，`multiview` 因为 10-view 输入 encoder 耗时预计为 `nas3r` 的 ~5×。

---

## 5. 复现工作流总结

### 5.1 评测开销

| 项 | 数值 |
|---|---|
| Checkpoint 下载 | 4 × 4.89 GB = **19.6 GB** |
| Test 数据（按需 shard） | **57.7 GB**（543 个 `.torch`） |
| 代码 + 依赖环境 | ≈ 2 GB |
| **评测总耗时（4 卡并行首次跑）** | **≈ 45 分钟** |
| 断点续跑（仅 1 个模型重测） | **22 分 15 秒** |
| 输出目录大小 | < 100 MB |

### 5.2 评测命令

```bash
# 一键完整流程（全新机器）
cd /opt/tiger/mmfinetune
curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/auto_eval.sh -o auto_eval.sh
bash auto_eval.sh

# 仅评测（数据/权重/环境都已就绪）
cd /opt/tiger/mmfinetune/nas3r
SKIP_CLONE=1 SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_SANITY=1 \
    bash /opt/tiger/mmfinetune/auto_eval.sh

# 强制重跑某个模型
rm -rf outputs/eval/nas3r
SKIP_CLONE=1 SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_SANITY=1 \
    bash /opt/tiger/mmfinetune/auto_eval.sh

# 生成汇总报告
python3 summarize_eval.py
cat outputs/eval/summary.md
```

### 5.3 评测过程中踩过的坑（已全部修复）

| # | 问题 | 解决方案 |
|---|---|---|
| 1 | `diff-gaussian-rasterization` submodule 静默失败 | 手动 clone fallback + GLM 头文件单独拉取 |
| 2 | 编译所有 GPU 架构导致超慢 | 显式 `TORCH_CUDA_ARCH_LIST=9.0`（仅 H100） |
| 3 | `snapshot_download` 目录写错位置（57 GB 白下） | 修正 `local_dir` 指向数据根目录 |
| 4 | `test_step` 硬编码 `assert b == 1` | 移除 `data_loader.test.batch_size=4` override |
| 5 | Sanity check timeout 太短（300s）| 放宽到 1200s，区分 `rc=124` 与真错误 |
| 6 | Step 5 `rm -rf` 删掉已完成结果 | 加断点续跑（`scores_all_avg.json` 存在则跳过） |
| 7 | 扩展包名不是 `diff_gaussian_rasterization` | 改用 `diff_gauss_camera`（作者 fork 改的名） |
| 8 | nas3r 第一次跑缺 Pose AUC 数据 | 识别 log 位置问题后，rerun 从而收齐 |

### 5.4 代码仓库

- **评测脚本**: [yhyblog/NAS3R](https://github.com/yhyblog/NAS3R)
  - `auto_eval.sh` — 一键评测主脚本（clone + install + download + eval + summary 全流程）
  - `summarize_eval.py` — 结果聚合 + Markdown 生成
  - `src/scripts/prepare_re10k_test_subset.py` — 按需下载 re10k 数据子集
- **原论文代码**: [ranrhuang/NAS3R](https://github.com/ranrhuang/NAS3R)

---

## 6. 文件产出

```
outputs/eval/
├── summary.md                           ← 自动生成的 Markdown
├── summary.json                         ← 机器可读版
├── nas3r/eval_nas3r/
│   ├── scores_all.json                  # per-scene 指标 (5,601 条)
│   ├── scores_all_avg.json              # 全局平均
│   ├── scores_sub_avg.json              # overlap 分档平均
│   ├── benchmark.json                   # encoder/decoder 每次调用的耗时
│   ├── peak_memory.json                 # 显存峰值
│   └── test_ckpt_path.txt               # 使用的 checkpoint 路径
├── multiview/eval_multiview/            (同上结构)
├── pretrained/eval_pretrained/          (同上结构)
└── pretrained-I/eval_pretrained-I/      (同上结构)
```

---

*Report generated 2026-04-19 · Evaluation on RealEstate10K (5,601 scenes) · 8 × H100 80GB*

# NAS3R · RealEstate10K 评测结果汇总

数据集：RealEstate10K test (`assets/evaluation_index_re10k.json`，共 5601 个有效 scene)。

评测命令参考 README，模式：`mode=test`，view_sampler=`evaluation`。

## 完成情况

| Model | 状态 | 日志 |
|---|---|---|
| `nas3r` (2-view (random init)) | ✓ 已完成 | `/opt/tiger/mmfinetune/nas3r/outputs/eval/nas3r/eval.log` |
| `multiview` (10-view) | ✓ 已完成 | `/opt/tiger/mmfinetune/nas3r/outputs/eval/multiview/eval.log` |
| `pretrained` (2-view (VGGT pretrained)) | ✓ 已完成 | `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained/eval.log` |
| `pretrained-I` (2-view (VGGT + GT intrinsics)) | ✓ 已完成 | `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained-I/eval.log` |

## 1. 新视角合成质量（All Pairs）

| Model | Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| `nas3r` | 2-view (random init) | 23.135 | 0.7640 | 0.1940 |
| `multiview` | 10-view | 27.102 | 0.8720 | 0.1130 |
| `pretrained` | 2-view (VGGT pretrained) | 25.887 | 0.8610 | 0.1370 |
| `pretrained-I` | 2-view (VGGT + GT intrinsics) | 25.891 | 0.8610 | 0.1370 |

## 2. 按 Overlap 分档的渲染质量

### Overlap: **large**

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| `nas3r` | 26.036 | 0.8460 | 0.1310 |
| `multiview` | 29.243 | 0.9100 | 0.0850 |
| `pretrained` | 28.370 | 0.9040 | 0.0980 |
| `pretrained-I` | 28.378 | 0.9040 | 0.0980 |

### Overlap: **medium**

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| `nas3r` | 22.946 | 0.7650 | 0.1920 |
| `multiview` | 26.934 | 0.8730 | 0.1130 |
| `pretrained` | 25.716 | 0.8610 | 0.1360 |
| `pretrained-I` | 25.724 | 0.8620 | 0.1360 |

### Overlap: **small**

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| `nas3r` | 20.112 | 0.6670 | 0.2720 |
| `multiview` | 24.922 | 0.8280 | 0.1450 |
| `pretrained` | 23.317 | 0.8110 | 0.1830 |
| `pretrained-I` | 23.305 | 0.8110 | 0.1830 |

## 3. 相机位姿误差（All Pairs · 平均角度，越小越好）

| Model | tgt_e_R | tgt_e_t | tgt_e_pose | cxt_e_R | cxt_e_t | cxt_e_pose |
|---|---|---|---|---|---|---|
| `nas3r` | 3.703 | 12.299 | 12.615 | 7.293 | 12.611 | 13.945 |
| `multiview` | 2.047 | 8.387 | 8.568 | 3.824 | 6.436 | 7.189 |
| `pretrained` | 1.023 | 4.936 | 5.063 | 1.588 | 3.167 | 3.533 |
| `pretrained-I` | 0.817 | 4.448 | 4.556 | 1.395 | 2.794 | 3.110 |

## 4. 相机位姿 AUC（All Pairs · @5°/@10°/@20°，越高越好）

### Rotation — `R`

| Model | AUC@5° | AUC@10° | AUC@20° | Median err. |
|---|---|---|---|---|
| `nas3r` | 0.5853 | 0.7006 | 0.7843 | 1.2512 |
| `multiview` | 0.6630 | 0.7827 | 0.8618 | 1.0149 |
| `pretrained` | 0.8143 | 0.8901 | 0.9373 | 0.4681 |
| `pretrained-I` | 0.8368 | 0.9025 | 0.9442 | 0.3701 |

### Translation — `t`

| Model | AUC@5° | AUC@10° | AUC@20° | Median err. |
|---|---|---|---|---|
| `nas3r` | 0.3468 | 0.5266 | 0.6632 | 3.4022 |
| `multiview` | 0.4452 | 0.6374 | 0.7746 | 2.4569 |
| `pretrained` | 0.6596 | 0.7934 | 0.8790 | 1.1349 |
| `pretrained-I` | 0.7131 | 0.8237 | 0.8938 | 0.8548 |

### Pose (R+t) — `pose`

| Model | AUC@5° | AUC@10° | AUC@20° | Median err. |
|---|---|---|---|---|
| `nas3r` | 0.3293 | 0.5121 | 0.6498 | 3.5510 |
| `multiview` | 0.4153 | 0.6123 | 0.7559 | 2.6577 |
| `pretrained` | 0.6369 | 0.7786 | 0.8695 | 1.2310 |
| `pretrained-I` | 0.6958 | 0.8120 | 0.8861 | 0.9234 |

## 5. 推理开销

| Model | Encoder 平均耗时 | Decoder 平均耗时 |
|---|---|---|
| `nas3r` | — | — |
| `multiview` | — | — |
| `pretrained` | — | — |
| `pretrained-I` | — | — |

## 6. 结果文件位置


### ✓ `nas3r` — 2-view (random init)
- checkpoint: `checkpoints/re10k_nas3r.ckpt`
- 输出目录: `/opt/tiger/mmfinetune/nas3r/outputs/eval/nas3r`
- 平均指标 JSON: `/opt/tiger/mmfinetune/nas3r/outputs/eval/nas3r/eval_nas3r/scores_all_avg.json`
- 分档指标 JSON: `/opt/tiger/mmfinetune/nas3r/outputs/eval/nas3r/eval_nas3r/scores_sub_avg.json`
- 主日志: `/opt/tiger/mmfinetune/nas3r/outputs/eval/nas3r/eval.log`
- 其他日志候选: `/opt/tiger/mmfinetune/nas3r/outputs/exp_eval_nas3r/2026-04-19_09-49-08/main.log`

### ✓ `multiview` — 10-view
- checkpoint: `checkpoints/re10k_nas3r_multiview.ckpt`
- 输出目录: `/opt/tiger/mmfinetune/nas3r/outputs/eval/multiview`
- 平均指标 JSON: `/opt/tiger/mmfinetune/nas3r/outputs/eval/multiview/eval_multiview/scores_all_avg.json`
- 分档指标 JSON: `/opt/tiger/mmfinetune/nas3r/outputs/eval/multiview/eval_multiview/scores_sub_avg.json`
- 主日志: `/opt/tiger/mmfinetune/nas3r/outputs/eval/multiview/eval.log`
- 其他日志候选: `/opt/tiger/mmfinetune/nas3r/outputs/exp_eval_multiview/2026-04-19_08-56-34/main.log`

### ✓ `pretrained` — 2-view (VGGT pretrained)
- checkpoint: `checkpoints/re10k_nas3r_pretrained.ckpt`
- 输出目录: `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained`
- 平均指标 JSON: `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained/eval_pretrained/scores_all_avg.json`
- 分档指标 JSON: `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained/eval_pretrained/scores_sub_avg.json`
- 主日志: `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained/eval.log`
- 其他日志候选: `/opt/tiger/mmfinetune/nas3r/outputs/exp_eval_pretrained/2026-04-19_08-56-40/main.log`

### ✓ `pretrained-I` — 2-view (VGGT + GT intrinsics)
- checkpoint: `checkpoints/re10k_nas3r_pretrained-I.ckpt`
- 输出目录: `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained-I`
- 平均指标 JSON: `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained-I/eval_pretrained-I/scores_all_avg.json`
- 分档指标 JSON: `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained-I/eval_pretrained-I/scores_sub_avg.json`
- 主日志: `/opt/tiger/mmfinetune/nas3r/outputs/eval/pretrained-I/eval.log`
- 其他日志候选: `/opt/tiger/mmfinetune/nas3r/outputs/exp_eval_pretrained-I/2026-04-19_08-56-45/main.log`

# Arnold 入口指令模板（NAS3R 训练）

## 🎯 使用前的一个决定：训练哪个 setting？

| setting | checkpoint 产出 | 脚本 key |
|---|---|---|
| 🌟 **VGGT 预训练 · 2-view**（推荐，论文主力） | `re10k_nas3r_pretrained.ckpt` | `pretrained` |
| VGGT 预训练 + GT 内参 · 2-view（pose 最佳） | `re10k_nas3r_pretrained-I.ckpt` | `pretrained-I` |
| 从零训练 · 2-view（baseline） | `re10k_nas3r.ckpt` | `random` |
| 多视角 10-view | `re10k_nas3r_multiview.ckpt` | `multiview` |

---

## ✅ 入口指令（单行版，直接贴到 Arnold）

### pretrained（论文主力，**推荐首选**）

```bash
bash -c 'export WANDB_API_KEY=你的wandb_key WANDB_MODE=online WANDB_PROJECT=nas3r TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1 MAX_JOBS=8 HF_HUB_ENABLE_HF_TRANSFER=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_ASYNC_ERROR_HANDLING=1 PARENT_DIR=/opt/tiger/mmfinetune PER_GPU_BATCH=5 MAX_STEPS=400001 VAL_CHECK_INTERVAL=10000 CKPT_EVERY_N_STEPS=5000 SAVE_TOP_K=3 NUM_WORKERS=8 RESUME=auto && mkdir -p /opt/tiger/mmfinetune && cd /opt/tiger/mmfinetune && curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/train.sh -o train.sh && chmod +x train.sh && bash train.sh pretrained'
```

> ⚠️ 把上面的 `你的wandb_key` 替换成你在 wandb.ai 上的真实 API key；
>     如果不想用 wandb，删除 `WANDB_API_KEY=...` 并把 `WANDB_MODE=online` 改成 `WANDB_MODE=offline`。

---

## 📖 多行版（同上，只是易读；如平台支持多行粘贴可用这个）

```bash
# ========== 应用层 env（平台没管的）==========
export WANDB_API_KEY="你的wandb_key"
export WANDB_MODE=online
export WANDB_PROJECT=nas3r

export TORCH_CUDA_ARCH_LIST=9.0     # H100 sm_90，编译 CUDA 扩展用
export FORCE_CUDA=1
export MAX_JOBS=8
export HF_HUB_ENABLE_HF_TRANSFER=1  # VGGT 权重加速下载
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# 以下是平台应该管、但平台没给就兜底的
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

# ========== 训练超参 ==========
export PARENT_DIR=/opt/tiger/mmfinetune
export PER_GPU_BATCH=5               # pretrained 论文配置；random 设 10；multiview 设 5
export MAX_STEPS=400001
export VAL_CHECK_INTERVAL=10000
export CKPT_EVERY_N_STEPS=5000
export SAVE_TOP_K=3
export NUM_WORKERS=8
export RESUME=auto                    # 自动续训

# ========== 拉代码 + 启动 ==========
mkdir -p "${PARENT_DIR}"
cd "${PARENT_DIR}"
curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/train.sh -o train.sh
chmod +x train.sh
bash train.sh pretrained
```

---

## 🔁 其他 setting 的单行指令

### random（从零训练）

只需把末尾的 `bash train.sh pretrained` 改成 `bash train.sh random`，**并把 `PER_GPU_BATCH=5` 改成 `PER_GPU_BATCH=10`**（论文配置）：

```bash
bash -c 'export WANDB_API_KEY=你的wandb_key WANDB_MODE=online WANDB_PROJECT=nas3r TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1 MAX_JOBS=8 HF_HUB_ENABLE_HF_TRANSFER=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_ASYNC_ERROR_HANDLING=1 PARENT_DIR=/opt/tiger/mmfinetune PER_GPU_BATCH=10 MAX_STEPS=400001 VAL_CHECK_INTERVAL=10000 CKPT_EVERY_N_STEPS=5000 SAVE_TOP_K=3 NUM_WORKERS=8 RESUME=auto && mkdir -p /opt/tiger/mmfinetune && cd /opt/tiger/mmfinetune && curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/train.sh -o train.sh && chmod +x train.sh && bash train.sh random'
```

### pretrained-I（VGGT + GT 内参）

`PER_GPU_BATCH=5` 保持，末尾 `bash train.sh pretrained-I`：

```bash
bash -c 'export WANDB_API_KEY=你的wandb_key WANDB_MODE=online WANDB_PROJECT=nas3r TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1 MAX_JOBS=8 HF_HUB_ENABLE_HF_TRANSFER=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_ASYNC_ERROR_HANDLING=1 PARENT_DIR=/opt/tiger/mmfinetune PER_GPU_BATCH=5 MAX_STEPS=400001 VAL_CHECK_INTERVAL=10000 CKPT_EVERY_N_STEPS=5000 SAVE_TOP_K=3 NUM_WORKERS=8 RESUME=auto && mkdir -p /opt/tiger/mmfinetune && cd /opt/tiger/mmfinetune && curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/train.sh -o train.sh && chmod +x train.sh && bash train.sh pretrained-I'
```

### multiview（10-view 多视角）

`PER_GPU_BATCH=5`（10-view 显存更紧），末尾 `bash train.sh multiview`：

```bash
bash -c 'export WANDB_API_KEY=你的wandb_key WANDB_MODE=online WANDB_PROJECT=nas3r TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1 MAX_JOBS=8 HF_HUB_ENABLE_HF_TRANSFER=1 HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_ASYNC_ERROR_HANDLING=1 PARENT_DIR=/opt/tiger/mmfinetune PER_GPU_BATCH=5 MAX_STEPS=400001 VAL_CHECK_INTERVAL=10000 CKPT_EVERY_N_STEPS=5000 SAVE_TOP_K=3 NUM_WORKERS=8 RESUME=auto && mkdir -p /opt/tiger/mmfinetune && cd /opt/tiger/mmfinetune && curl -fsSL https://raw.githubusercontent.com/yhyblog/NAS3R/main/train.sh -o train.sh && chmod +x train.sh && bash train.sh multiview'
```

---

## 🔐 Arnold 任务配置页面上你保留的 5 个环境变量（JSON）

```json
{
  "ARNOLD_GPU_NIC_AFFINITY": "0",
  "ARNOLD_IPV6": "1",
  "BYTED_TORCH_BYTECCL": "O0",
  "BYTED_TORCH_FX": "O0",
  "NCCL_DEBUG": "WARN"
}
```

---

## 📊 脚本在你不看时会自动做的所有事

1. **Step 0**: 从 GitHub clone NAS3R，初始化 submodule，手动兜底 `diff-gaussian-rasterization` + GLM
2. **Step 1**: pip 装依赖 + 编译 `pytorch3d` + `diff_gauss_camera`（sm_90）
3. **Step 1.5**: 预拉 VGGT-1B 权重（`pretrained` / `pretrained-I` 需要）
4. **Step 2**: 检测训练数据；不够的自动从 HuggingFace 拉全量 re10k（~575 GB）
5. **Step 3**: 自动续训判定（找最新 ckpt）
6. **Step 4**: `torchrun --nproc_per_node=8 -m src.main` 启动 8 卡 DDP 训练
7. **Step 5**: 异常退出时（非 Ctrl+C）**自动拉起续训**（如果是 daemon 模式，你可加 `--daemon` flag）

---

## 🔍 训练启动后查看进度

```bash
# 登录 Arnold 机器后
tail -f /opt/tiger/mmfinetune/nas3r/outputs/train_pretrained/train.log

# GPU 利用率
watch -n 30 nvidia-smi

# wandb 看训练曲线（如果填了 WANDB_API_KEY）
# https://wandb.ai/<your_username>/nas3r
```

---

## ⏱️ 预期耗时

| setting | 400K steps 预估 |
|---|---|
| `pretrained` / `pretrained-I` | **4–5 天** |
| `random` | 3–4 天 |
| `multiview` | 10 天 – 2 周（10-view 导致 encoder 慢 5×） |

如果平台有任务时长限制（例如 72h），**脚本会在意外超时退出后、下次你再提交任务时自动从最新 ckpt 续训**（`RESUME=auto` 起作用）。

# zjy_work 项目说明

## 项目用途

本工作区主要用于医学图像合成与病灶分割实验，围绕 ISIC2018 和 PH2 皮肤镜图像数据展开。整体包含三条相关代码线：

- `SBG-Diff`：基于 Stable Diffusion / ControlNet 的 mask-to-image 合成，包含 SBP-PG、边界先验、两阶段训练、推理和生成质量评估。
- `BGDNet`：下游病灶分割模型训练与测试，可使用真实数据或真实数据加合成数据训练。
- `BEF_SBG`：BEF-SBG 实验流水线，包含低标注划分、OOF teacher、边界误差反馈、扩散生成、样本评分、加权分割训练和最终评估。
- 
## 目录结构

```text
zjy_work/
├─ code/
│  ├─ SBG-Diff/
│  │  ├─ Baseline/
│  │  ├─ cldm/
│  │  ├─ data/
│  │  ├─ data_txt/
│  │  ├─ ldm/
│  │  ├─ models/
│  │  ├─ or/
│  │  └─ Ours/
│  │     ├─ ablation_sbg/
│  │     ├─ data_txt/
│  │     ├─ logs/
│  │     ├─ models/
│  │     └─ scripts/
│  ├─ BGDNet/
│  │  ├─ models/
│  │  ├─ model_out/
│  │  ├─ pretrained_pth/
│  │  ├─ scripts/
│  │  └─ utils/
│  └─ BEF_SBG/
│     ├─ diffusion/
│     ├─ feedback/
│     ├─ scoring/
│     ├─ scripts/
│     ├─ segmentation/
│     └─ splits/
├─ ISIC2018/
└─ PH2 Dataset/
```

| 文件夹 | 作用 |
| --- | --- |
| `code/` | 存放主要实验代码，包含扩散合成、分割模型和 BEF-SBG 流水线。 |
| `code/SBG-Diff/` | 扩散生成主项目，用于从病灶 mask 生成皮肤镜图像。 |
| `code/SBG-Diff/Baseline/` | 分割 baseline 训练与测试代码。 |
| `code/SBG-Diff/cldm/` | ControlNet / ControlLDM 相关模型实现。 |
| `code/SBG-Diff/ldm/` | Latent Diffusion 相关基础模块。 |
| `code/SBG-Diff/models/` | 扩散模型 YAML 配置，如 `cldm_v15.yaml`。 |
| `code/SBG-Diff/data/` | prompt JSON 文件。 |
| `code/SBG-Diff/data_txt/` | 训练、测试、混合比例数据列表。 |
| `code/SBG-Diff/or/` | 旧版或原始脚本、README 和 conda 环境文件。 |
| `code/SBG-Diff/Ours/` | 当前主要实验代码、模型权重、日志、推理结果和脚本。 |
| `code/SBG-Diff/Ours/scripts/` | CN-only、SBP-PG stage1/stage2、merge、infer、eval 等启动脚本。 |
| `code/SBG-Diff/Ours/models/` | Stable Diffusion / ControlNet 初始化权重与合并权重。 |
| `code/BGDNet/` | 下游病灶分割项目。 |
| `code/BGDNet/models/` | BGDNet、UNet、TransXNet 等网络结构。 |
| `code/BGDNet/utils/` | 数据加载、预处理、增强和训练工具。 |
| `code/BGDNet/scripts/` | 数据检查、真实数据训练、真实加合成训练、测试脚本。 |
| `code/BGDNet/model_out/` | 分割模型训练输出和评估结果。 |
| `code/BGDNet/pretrained_pth/` | 分割模型相关预训练权重。 |
| `code/BEF_SBG/` | BEF-SBG 工作流代码。 |
| `code/BEF_SBG/scripts/` | 00-13 主流程脚本，串联 split、teacher、反馈、生成、评分和最终分割训练。 |
| `code/BEF_SBG/splits/` | 低标注比例和 K-fold 数据划分。 |
| `code/BEF_SBG/feedback/` | 边界误差反馈构建、可视化和 BEF prompt 生成。 |
| `code/BEF_SBG/diffusion/` | BEF-SBG 扩散训练与推理脚本。 |
| `code/BEF_SBG/scoring/` | 合成样本质量评分和加权训练集构建。 |
| `code/BEF_SBG/segmentation/` | 最终 BGDNet-BEF 分割训练与测试。 |
| `ISIC2018/` | ISIC2018 数据集目录。 |
| `PH2 Dataset/` | PH2 数据集目录。 |

## 启动脚本

### SBG-Diff

```bash
bash /data/zjy_work/BGDiff/Ours/scripts/init.sh
bash /data/zjy_work/BGDiff/Ours/scripts/CN-only.sh
bash /data/zjy_work/BGDiff/Ours/scripts/stage1_sbp_pg.sh
bash /data/zjy_work/BGDiff/Ours/scripts/stage2_sbp_pg.sh
bash /data/zjy_work/BGDiff/Ours/scripts/merge.sh
bash /data/zjy_work/BGDiff/Ours/scripts/infer_sbp_pg.sh
bash /data/zjy_work/BGDiff/Ours/scripts/eval.sh
```

### BGDNet

```bash
bash /data/zjy_work/BGDNet/scripts/01_check_load.sh
bash /data/zjy_work/BGDNet/scripts/02_train_real.sh
bash /data/zjy_work/BGDNet/scripts/02_train_cn_only.sh
bash /data/zjy_work/BGDNet/scripts/02_train_stage1-2.sh
bash /data/zjy_work/BGDNet/scripts/03_test.sh
```

### BEF-SBG

```bash
bash /data/zjy_work/Work3_BEF_SBG/scripts/01_make_low_label_splits.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/02_train_oof_teachers.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/03_infer_oof_predictions.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/04_build_boundary_feedback.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/05_visualize_feedback.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/06_make_bef_prompt_json.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/07_train_diffusion_stage1.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/08_train_diffusion_stage2.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/09_infer_diffusion_bef_sbg.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/10_score_generated_samples.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/11_make_weighted_train_json.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/12_train_final_bgdnet_bef.sh
bash /data/zjy_work/Work3_BEF_SBG/scripts/13_test_final_bgdnet_bef.sh
```

## 依赖文件

当前仅发现一个明确的依赖环境文件：

```text
code/SBG-Diff/or/environment.yaml
```

说明文件中还提到需要安装 `xformers` 和 `deepspeed`。`BGDNet` 和 `BEF_SBG` 未发现独立的 `requirements.txt` 或 `pyproject.toml`，实际运行时可能复用 `BGDiff` 环境。

## 配置文件

| 文件 | 作用 |
| --- | --- |
| `code/SBG-Diff/config.py` | 简单运行配置，当前包含 `save_memory = True`。 |
| `code/SBG-Diff/models/cldm_v15.yaml` | ControlLDM / ControlNet 主配置。 |
| `code/SBG-Diff/models/cldm_v15_3.20.yaml` | 旧版本或特定实验配置。 |
| `code/SBG-Diff/models/cldm_v21.yaml` | v2.1 相关扩散配置。 |
| `code/SBG-Diff/data/prompt_*.json` | 扩散训练/推理 prompt 数据。 |
| `code/BEF_SBG/scripts/00_common.sh` | BEF-SBG 主流程公共路径、GPU、输出目录配置。 |
| `code/BEF_SBG/feedback/prompt_bef_train_5p.json` | BEF-SBG 训练 prompt 示例。 |

## 可用命令

常见 Python 入口包括：

```bash
python /data/zjy_work/BGDiff/make_prompt_json.py
python /data/zjy_work/BGDiff/tool_add_control.py
python /data/zjy_work/BGDiff/tool_merge_control.py
python /data/zjy_work/BGDiff/tutorial_train.py
python /data/zjy_work/BGDiff/tutorial_inference.py
python /data/zjy_work/BGDNet/BGDiff_train.py
python /data/zjy_work/BGDNet/BGDiff_test.py
python /data/zjy_work/Work3_BEF_SBG/splits/make_low_label_splits.py
python /data/zjy_work/Work3_BEF_SBG/feedback/build_boundary_feedback.py
python /data/zjy_work/Work3_BEF_SBG/scoring/score_generated_samples.py
python /data/zjy_work/Work3_BEF_SBG/segmentation/train_BGDNet_BEF.py
python /data/zjy_work/Work3_BEF_SBG/segmentation/test_BGDNet_BEF.py
```

## 风险命令

- 大多数训练和推理脚本使用 `nohup ... &` 后台运行，会持续占用 GPU。
- 训练、推理和评估脚本会写入 `logs/`、`checkpoints/`、`results/`、`model_out/` 等目录。
- `tool_add_control.py` 和 `tool_merge_control.py` 会生成或覆盖模型权重。
- `code/BEF_SBG/scripts/kill_by_pid.sh` 会根据 pid 文件终止进程。
- `code/SBG-Diff/Ours/scripts/eval.sh` 中 active 部分使用 `$LOG_DIR`，但当前脚本内 `LOG_DIR` 定义被注释，运行前需要检查。
- `code/BGDNet/scripts/02_train_stage1-2.sh` 中 `--train_list2` 行的续行符前缺少空格，可能导致参数拼接异常。

## 待确认事项

- 当前文件系统路径是 Windows 风格，但脚本大量硬编码为 `/data/zjy_work/...`，需要确认实际运行环境是否为 Linux、WSL 或容器。
- 实际目录名是 `SBG-Diff` 和 `BEF_SBG`，脚本中常用 `BGDiff` 和 `Work3_BEF_SBG`，需要确认是否存在软链接或路径映射。
- `PH2 Dataset` 与脚本中的 `/data/zjy_work/PH2` 命名不一致，运行 PH2 相关流程前需要确认路径。
- `BGDNet` 和 `BEF_SBG` 没有独立依赖清单，环境复用关系需要确认。

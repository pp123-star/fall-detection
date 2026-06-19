# 📋 项目交接文档 (AI Handover Document)

**Target Agent:** 此文档用于 AI 助手接管当前的“基于深度学习的视频动作识别（摔倒检测）”项目。请仔细阅读当前状态，并严格按照 `Next Steps` 推进。

## 1. 项目基础配置 (Project Context)

* **任务目标:** 视频动作二分类（摔倒 vs 非摔倒/困难负样本）。
* **硬件环境:** AutoDL 平台, RTX 4090 (24GB) * 1, 纯 GPU 模式已开启。
* **软件环境:** Ubuntu 22.04, Python 3.10, 虚拟环境名称为 `falldet`。
* **工作目录:** `~/autodl-tmp/fall-detection`

## 2. 环境状态 (Environment Status) - [已完全就绪]

* 已在一个干净的 conda 环境 (`falldet`) 中通过 `bash env/setup_autodl.sh` 完成所有依赖安装。
* **已解决的历史冲突:** * 彻底修复了无卡模式导致的 CUDA 丢失问题。
* 强制锁定了 `numpy<2.0` 解决底层 API 冲突。
* 使用 `--no-build-isolation` 强装了 `chumpy` 解决了 pip 隔离机制导致的编译崩溃。


* **验证结果:** `python env/verify.py` 报告所有包（PyTorch 2.1.0+cu118, MMCV 2.1.0, MMAction2 1.2.0, Ultralytics 等）加载正常，GPU 识别成功。

## 3. 数据状态 (Data Status) - [已完全就绪]

* **数据下载:** 已成功下载 OpenMMLab 官方预提取的 NTU60 2D 骨骼数据集 (`data/ntu60_2d.pkl`)。
* **二分类构建:** 已执行 `build_binary_pkl.py`。成功将 60 类转化为 0/1 二分类，并重点混入了 "sit down", "staggering" 等困难负样本。输出文件为 `data/fall_binary_xsub.pkl`。
* **防泄漏自检:** 已执行 `split_check.py`。确认 X-Sub 划分（按受试者）绝对安全，训练集（20人）与验证集（20人）完全隔离，0 重叠。
* **骨骼可视化核验:** 已生成 COCO 17 点骨骼连线视频，**人工已肉眼确认关键点顺序 100% 正确**（头连头、脚连脚，无乱线）。

## 4. 下一步行动指令 (Actionable Next Steps)

环境与数据均已完美 Ready，尚未开始任何模型训练。请接管系统并**直接从模型训练开始**，严格按以下顺序在 `falldet` 环境下执行命令：

**Step 1: 启动主线模型训练 (Train PoseConv3D)**
请使用单卡启动训练流程（预计耗时 1.5 - 2 小时）：

```bash
conda activate falldet
cd ~/autodl-tmp/fall-detection
bash tools/train.sh configs/posec3d_fall_binary.py 1

```

**Step 2: Checkpoint 完整性自检 (Verify Weights)**
训练结束后，必须运行此脚本以验证最佳权重文件是否正确保存，防止发生保存逻辑 Bug：

```bash
python tools/verify_best_ckpt.py work_dirs/posec3d_fall_binary

```

**Step 3: 测试与二分类指标评估 (Evaluation)**
使用跑出的最佳 checkpoint 进行测试，并输出精准的二分类指标（F1, Recall, 混淆矩阵）：

```bash
# 请将 <BEST_CKPT_PATH> 替换为 Step 2 中确认的最佳权重路径
bash tools/test.sh configs/posec3d_fall_binary.py <BEST_CKPT_PATH> work_dirs/posec3d_fall_binary/pred.pkl

python tools/eval_binary_metrics.py \
    --pred work_dirs/posec3d_fall_binary/pred.pkl \
    --config configs/posec3d_fall_binary.py \
    --out-dir work_dirs/posec3d_fall_binary/eval \
    --save-errors

```

*(如果以上一切顺利，可以继续按照项目文档执行 `configs/stgcnpp_fall_binary.py` 对比模型的训练，或绘制训练曲线。)*

## 5. 2026-06-20 远端准备度复核 (Remote Readiness Audit)

**执行人:** Codex  
**服务器:** `root@connect.westc.seetacloud.com:50071`  
**远端目录:** `/root/autodl-tmp/fall-detection`

### 5.1 当前平台模式

* 当前实例处于**无卡模式**，`nvidia-smi` 输出 `No devices were found`。
* 这与用户说明一致，不视为环境故障；正式训练前需要在平台切换到有卡/GPU 模式后再次确认 `nvidia-smi` 和 `torch.cuda.is_available()`。

### 5.2 已确认就绪项

* 远端仓库存在，当前提交为 `5d10513`，分支为 `main...origin/main`。
* 核心文件均存在：`README.md`、两份模型配置、`tools/train.sh`、`tools/test.sh`、`tools/verify_best_ckpt.py`、`tools/eval_binary_metrics.py`、`mmaction2_src/tools/train.py`。
* `tools/train.sh`、`tools/test.sh`、`env/setup_autodl.sh` 均通过 `bash -n` 语法检查。
* `falldet` 环境存在，包导入检查通过：
  * Python 3.10.20
  * PyTorch 2.1.0+cu118
  * NumPy 1.26.4
  * MMCV 2.1.0
  * MMEngine 0.10.7
  * MMAction2 1.2.0
  * MMDetection 3.2.0
  * MMPose 1.3.2
  * Ultralytics 8.4.71
  * OpenCV 4.8.1
  * Decord 0.6.0
* 数据文件存在：
  * `data/ntu60_2d.pkl`，约 673 MB
  * `data/ntu120_2d.pkl`，约 1.2 GB
  * `data/fall_binary_xsub.pkl`，约 38 MB
* `data/fall_binary_xsub.pkl` 数据结构检查通过：
  * annotations: 4538
  * xsub_train: 2013，label=0: 1342，label=1: 671
  * xsub_val: 825，label=0: 550，label=1: 275
  * xview_train: 3036，label=0: 2406，label=1: 630
  * xview_val: 1502，label=0: 1186，label=1: 316
  * sample keypoint shape: `(1, 69, 17, 2)`，keypoint_score shape: `(1, 69, 17)`
* 两份训练配置均可被 MMEngine 加载：
  * `configs/posec3d_fall_binary.py`: `ann_file=data/fall_binary_xsub.pkl`，`train=xsub_train`，`val=xsub_val`，`work_dir=work_dirs/posec3d_fall_binary`
  * `configs/stgcnpp_fall_binary.py`: `ann_file=data/fall_binary_xsub.pkl`，`train=xsub_train`，`val=xsub_val`，`work_dir=work_dirs/stgcnpp_fall_binary`
* `vis/` 下已有 5 个骨骼可视化 mp4。
* `work_dirs/` 当前没有训练产物，确认尚未开始训练。
* 服务器未安装 `tmux`，但已安装 `screen 4.09.00`，后续建议用 `screen` 共享训练终端。

### 5.3 需要注意的核查结果

* `data_prep/split_check.py --src data/fall_binary_xsub.pkl` 返回非零，原因是脚本把同一数据包内的 X-Sub 与 X-View 两套官方划分互相比，发现跨划分样本重叠。
* 训练配置实际只使用 `xsub_train/xsub_val`；这两者检查结果为 0 重叠，且受试者完全隔离。因此主训练路径没有发现 X-Sub 泄漏问题。
* 后续如果希望 `split_check.py` 用作自动化门禁，建议新增“只检查 xsub_train/xsub_val”的模式或参数；在未改脚本前，不应把该脚本的整体返回码等同于训练数据不可用。

### 5.4 下一步

1. 用户在平台切换到有卡/GPU 模式。
2. 重新确认：

```bash
nvidia-smi
conda run -n falldet python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

3. 使用 `screen` 启动训练共享会话：

```bash
cd /root/autodl-tmp/fall-detection
screen -S falldet-train
conda activate falldet
bash tools/train.sh configs/posec3d_fall_binary.py 1
```

用户可在 JupyterLab 终端中运行：

```bash
screen -x falldet-train
```

如只想看最新训练日志，可在训练产生日志后运行：

```bash
tail -f "$(find work_dirs/posec3d_fall_binary -name '*.log' -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
```

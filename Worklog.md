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

## 6. 2026-06-20 有卡模式环境修复与训练启动记录

**执行人:** Codex  
**服务器:** `root@connect.westc.seetacloud.com:50071`  
**远端目录:** `/root/autodl-tmp/fall-detection`

### 6.1 GPU 与环境复核

* 用户切换到有卡模式后重新登录，AutoDL 显示：
  * CPU: 12 核
  * 内存: 90 GB
  * GPU: NVIDIA GeForce RTX 4090
* `nvidia-smi` 通过，显卡为 RTX 4090，显存 24564 MiB。
* PyTorch CUDA 检查通过：

```text
2.1.0+cu118 True NVIDIA GeForce RTX 4090
```

### 6.2 发现并处理的环境问题

* `env/verify.py` 报 `MMAction MODELS registry size: 0`。根因不是 CUDA，而是验证脚本直接读取 registry，没有按训练入口完整初始化 MMAction2 scope/modules。
* 用更接近训练入口的方式检查时，发现 `mmaction2_src` 导入模型模块缺少 `importlib_metadata`：

```text
ModuleNotFoundError: No module named 'importlib_metadata'
```

* 已在 `falldet` 环境补装：

```bash
pip install importlib-metadata
```

* 补装后 `import mmaction.models` 可正常注册模型，`Recognizer3D` 存在。
* 当前 shell 原本 `OMP_NUM_THREADS=0`，会触发 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`。训练启动时已显式设置：

```bash
export OMP_NUM_THREADS=1
```

### 6.3 训练前验证

* `Runner.from_cfg(Config.fromfile('configs/posec3d_fall_binary.py'))` 成功，模型为 `Recognizer3D`。
* 取一个训练 batch 成功，数据管线可跑通：

```text
inputs len: 2
first shape: (1, 17, 48, 56, 56)
data_samples: 2
```

* 临时打印 label 时出现的 `TypeError` 是检查脚本中的打印写法问题，不影响训练数据管线。

### 6.4 启动训练时遇到的 deterministic 问题

* 按 `tools/train.sh` 启动时，脚本会传入 `--deterministic`。
* 第一次失败：PyTorch deterministic + CuBLAS 需要 `CUBLAS_WORKSPACE_CONFIG`。
* 第二次设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 后仍失败，因为 PoseConv3D 的 `max_pool3d_with_indices_backward_cuda` 没有 deterministic CUDA 实现：

```text
RuntimeError: max_pool3d_with_indices_backward_cuda does not have a deterministic implementation
```

* 处理方式：不修改训练配置和模型逻辑，保留 `--seed 42`，去掉 `--deterministic`，直接调用 MMAction2 训练入口。
* 影响：不能保证逐 bit 完全复现，但训练实验本身有效；这属于 PyTorch/CUDA deterministic 限制，不是数据或模型配置错误。

### 6.5 当前训练状态

* 训练已在 `screen` 中启动并进入正常迭代：

```bash
screen -x falldet-train
```

* 实际启动命令等价于：

```bash
cd /root/autodl-tmp/fall-detection
source /root/miniconda3/etc/profile.d/conda.sh
conda activate falldet
export OMP_NUM_THREADS=1
export PYTHONPATH=/root/autodl-tmp/fall-detection/mmaction2_src:$PYTHONPATH
python mmaction2_src/tools/train.py configs/posec3d_fall_binary.py --seed 42
```

* 最新确认进度：

```text
Epoch(train) [1][140/1259]
memory: 9559 MiB
```

* `nvidia-smi` 显示训练进程已占用 GPU，显存约 12 GB，GPU util 约 62%。
* 控制台日志：

```bash
tail -f /root/autodl-tmp/fall-detection/work_dirs/posec3d_fall_binary_console.log
```

* MMAction2 最新日志：

```bash
tail -f /root/autodl-tmp/fall-detection/work_dirs/posec3d_fall_binary/20260620_005241/20260620_005241.log
```

### 6.6 split_check.py 说明

* `split_check.py` 整体返回失败的原因是它把同一数据包内的 X-Sub 与 X-View 两套官方划分互相比，发现跨协议重叠。
* 当前训练配置只使用 `xsub_train/xsub_val`，这两个 split 的样本名和受试者均已确认隔离，所以不阻塞当前训练。
* 后续如果要把检查脚本作为 CI/门禁，应给 `split_check.py` 增加只检查指定协议的参数，例如只检查 `xsub_train/xsub_val`。

### 6.7 技术文档对照结果

本次执行原则：先按技术文档走，技术文档命令走不通时只做运行层面的临时修正，不修改技术文档正文。

* 与文档一致的部分：
  * `README.md` 和 `docs/03_model_training.md` 要求训练主线模型 `configs/posec3d_fall_binary.py`，本次确实先启动 PoseConv3D。
  * `docs/01_environment_setup.md` 要求使用 `falldet` 环境、`mmaction2_src/tools/train.py` 训练入口和 `data/fall_binary_xsub.pkl`，本次均按该方向执行。
  * 训练配置仍使用 `xsub_train/xsub_val`，没有切换数据协议，也没有改模型结构、数据文件或训练配置。

* 文档命令在当前环境下暴露的问题：
  * `docs/01_environment_setup.md` 建议 `python env/verify.py` 作为总体环境验证；当前 `env/verify.py` 直接读取 `mmaction.registry.MODELS`，没有按训练入口导入 `mmaction.models`，因此出现 `MMAction MODELS registry size: 0`。实际用 `Runner.from_cfg(...)` 验证训练入口后，模型可正常构建。
  * 文档/脚本路径默认通过 `tools/train.sh` 启动；该脚本会传入 `--deterministic`。在当前 PyTorch 2.1 + CUDA 环境下，PoseConv3D 训练会因 `max_pool3d_with_indices_backward_cuda` 无 deterministic CUDA 实现而失败。
  * `mmaction2_src` 导入模型模块时缺少 `importlib_metadata`，`setup_autodl.sh` 当前没有显式安装该包。已通过 `pip install importlib-metadata` 修复。
  * 当前 shell 中 `OMP_NUM_THREADS=0` 会触发 `libgomp` 警告。训练启动时已设置 `OMP_NUM_THREADS=1`。

* 临时修正与结果：
  * 未修改技术文档、未修改模型配置、未修改数据集构建逻辑。
  * 已补装 `importlib-metadata`，`import mmaction.models` 后 registry 正常，`Recognizer3D` 可注册。
  * 已用 `Runner.from_cfg(Config.fromfile('configs/posec3d_fall_binary.py'))` 验证训练入口可构建。
  * 已保留 `--seed 42`，去掉 `--deterministic` 直接调用 `mmaction2_src/tools/train.py` 启动训练。
  * 训练已经正常进入 Epoch 1 迭代并持续使用 RTX 4090。

* 后续文档修正建议（暂不执行）：
  * 在 `env/requirements_extra.txt` 或 `setup_autodl.sh` 中补充 `importlib-metadata`。
  * 修改 `env/verify.py` 的 registry 检查方式，先导入 `mmaction.models` 或改为 `Runner.from_cfg` 级别验证。
  * 在 `tools/train.sh` 中提供可选 deterministic 开关，默认不要强制对 PoseConv3D 开启 deterministic。
  * 给 `split_check.py` 增加 `--protocol xsub` 或类似参数，避免 X-Sub/X-View 跨协议重叠导致整体返回失败。

## 7. 2026-06-20 多目标实时检测文件接入记录

**执行人:** Codex  
**本地目录:** `D:\AAA\基于深度学习的视频动作识别技术研究\fall-detection`

### 7.1 新增文件放置

用户新增了两个由 Claude 生成的文件，原始位置均在仓库根目录：

* `06_多目标实时检测.md`
* `multitarget_realtime_demo.py`

已按仓库结构移动并改为英文文件名：

* `docs/06_multitarget_realtime_detection.md`
* `inference/multitarget_realtime_demo.py`

同时在 `README.md` 的目录结构中补充了这两个入口，便于 GitHub 浏览。

### 7.2 内容核查

* `docs/06_multitarget_realtime_detection.md` 说明新增的多目标实时检测流程、参数、事件日志、验证方式，以及“多人识别/摄像头输入不需要重训”的原因。
* `inference/multitarget_realtime_demo.py` 新增多人实时摔倒检测推理脚本，核心设计为：
  * 每个 `track_id` 独立维护 `TrackState`
  * 每个 track 独立 `deque` 缓冲 `clip_len=48` 帧骨骼
  * 使用 YOLO Pose + ByteTrack 获取多人关键点和 ID
  * 复用现有 `load_pose_model`、`_extract_one_frame`、`load_action_model`、`build_sample`
  * 通过 `CachedClipPredictor` 缓存 MMAction2 pipeline
  * 支持摄像头、本地视频、RTSP/HTTP 流、可视化 mp4、JSONL 事件日志、报警快照

### 7.3 检查结果

* 已将 `inference/multitarget_realtime_demo.py` 内模糊引用 `docs/06` 修正为 `docs/06_multitarget_realtime_detection.md`。
* 本地使用 Codex bundled Python 执行语法检查通过：

```powershell
python -m py_compile .\fall-detection\inference\multitarget_realtime_demo.py
```

* 语法检查产生的本地临时目录 `inference/__pycache__` 已删除，避免提交无关产物。

### 7.4 关于 checkpoint 保存策略

当前配置中：

```python
checkpoint=dict(
    interval=1,
    save_best="acc/top1",
    rule="greater",
    max_keep_ckpts=3,
    save_last=True,
)
```

含义：

* `max_keep_ckpts=3` 只限制普通 `epoch_N.pth` 最近保留 3 个。
* `best_acc_top1_epoch_*.pth` 是按验证集 `acc/top1` 单独保存的最优模型。
* 如果后续 epoch 出现过拟合，普通最近 3 个 checkpoint 可能较差，但之前保存的 best checkpoint 不会因此消失，除非出现新的验证集分数更高的 best。
* 后续正式测试、推理和部署应优先使用 `best_acc_top1_epoch_*.pth`，不是盲目使用最后一轮模型。

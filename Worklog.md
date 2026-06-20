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

### 7.5 GitHub 同步与本地代理诊断

* 本地 Git 代理原配置为 `http://127.0.0.1:7890`，但 `127.0.0.1:7890` 当前没有监听进程。
* 本机实际监听到 `127.0.0.1:7892`，进程为 `ShanHaiCore.exe`，因此判断为 Git 代理端口配置不匹配，不是典型端口冲突。
* 已使用一次性代理参数推送到 GitHub，未修改全局 Git 代理配置：

```powershell
git -c http.proxy=http://127.0.0.1:7892 -c https.proxy=http://127.0.0.1:7892 push origin main
```

* 推送结果：`main -> main`，GitHub 已包含本次新增的多目标实时检测文档和推理脚本。

## 8. 2026-06-20 PoseConv3D 训练完成与测试评估

**执行人:** Codex  
**服务器目录:** `/root/autodl-tmp/fall-detection`

### 8.1 训练完成状态

* 主线模型 `configs/posec3d_fall_binary.py` 已完成 24 epoch 训练。
* 训练未修改技术文档、模型配置、数据文件或训练逻辑。
* 训练完成时 GPU 已空闲，`nvidia-smi` 显示显存占用约 1 MiB。
* 最后一轮验证结果：

```text
Epoch(val) [24][52/52] acc/top1: 1.0000 acc/mean1: 1.0000
```

* 保留 checkpoint：

```text
work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth
work_dirs/posec3d_fall_binary/epoch_22.pth
work_dirs/posec3d_fall_binary/epoch_23.pth
work_dirs/posec3d_fall_binary/epoch_24.pth
```

### 8.2 checkpoint 一致性检查

已执行：

```bash
python tools/verify_best_ckpt.py work_dirs/posec3d_fall_binary
```

结果：

* 找到 1 个 `best_*.pth` 和最近 3 个普通 epoch checkpoint。
* 日志中最高验证集 `acc/top1=1.0000`，首次出现在 epoch 5。
* `best_acc_top1_epoch_5.pth` 与日志最高验证结果一致。
* epoch 22、23、24 在验证日志中也达到过 `acc/top1=1.0000`，但由于 checkpoint hook 使用 `rule="greater"`，与历史 best 打平不会覆盖 epoch 5 的 best 文件。

### 8.3 测试集评估结果

测试集为当前配置的 `xsub_val`，样本数 825，其中摔倒 275、非摔倒 550。已对 `best` 和最近三轮 checkpoint 分别执行：

```bash
bash tools/test.sh configs/posec3d_fall_binary.py <checkpoint>
python tools/eval_binary_metrics.py --pred <pred.pkl> --config configs/posec3d_fall_binary.py --out-dir <eval_dir> --save-errors
```

汇总结果：

| Checkpoint | Accuracy | Precision | Recall | Specificity | F1 | ROC AUC | PR AUC | 默认阈值误判 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `best_acc_top1_epoch_5.pth` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `epoch_22.pth` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `epoch_23.pth` | 0.9988 | 1.0000 | 0.9964 | 1.0000 | 0.9982 | 1.0000 | 1.0000 | 1 个 FN |
| `epoch_24.pth` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |

`epoch_23.pth` 默认阈值 0.5 下漏报 1 个摔倒样本：

```text
sample_130, gt=1, pred=0, fall_score=0.4759, FN
```

但阈值搜索显示 `epoch_23.pth` 也可通过调整阈值达到 F1=1.0000。

### 8.4 当前结论

* 本轮 PoseConv3D 训练产物可用，暂时没有必要因为 epoch 5 作为 best 而立即重训。
* 推荐优先保留并用于后续推理联调的 checkpoint：`work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth`。
* 同时建议保留 `epoch_24.pth` 作为末轮对照模型，因为它在默认阈值下测试指标同样全为 1.0000。
* 需要注意：当前测试仍使用项目配置内的 `xsub_val`，结果非常高，后续若要确认泛化能力，应补充新的真实视频、摄像头采样或独立场景数据做外部测试。

### 8.5 服务器产物路径

```text
work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5_pred.pkl
work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5_metrics.txt
work_dirs/posec3d_fall_binary/eval_best_acc_top1_epoch_5/

work_dirs/posec3d_fall_binary/epoch_22_pred.pkl
work_dirs/posec3d_fall_binary/epoch_22_metrics.txt
work_dirs/posec3d_fall_binary/eval_epoch_22/

work_dirs/posec3d_fall_binary/epoch_23_pred.pkl
work_dirs/posec3d_fall_binary/epoch_23_metrics.txt
work_dirs/posec3d_fall_binary/eval_epoch_23/

work_dirs/posec3d_fall_binary/epoch_24_pred.pkl
work_dirs/posec3d_fall_binary/epoch_24_metrics.txt
work_dirs/posec3d_fall_binary/eval_epoch_24/
```

## 9. 2026-06-20 真实视频叠加推理排查记录

**执行人:** Codex  
**服务器目录:** `/root/autodl-tmp/fall-detection`

### 9.1 用户上传视频

用户上传了 7 个真实视频到服务器：

```text
data/real_test/test1.mp4
data/real_test/test2.mp4
data/real_test/test3.mp4
data/real_test/test4.mp4
data/real_test/test5.mp4
data/real_test/test6.mp4
data/real_test/test7.mp4
```

检查结果：

* 7 个视频均可被 OpenCV 打开。
* 分辨率均为 `1440x3200` 竖屏。
* 帧率约 `60fps`。
* 视频长度从约 3.4 秒到约 56.9 秒不等。

### 9.2 已执行的服务器同步

服务器上的 `inference/multitarget_realtime_demo.py` 原先仍显示旧标签：

```text
id:<track_id> P:<prob>
```

已同步为本地/GitHub 当前显示方式：

```text
id:<track_id> NORMAL/FALL P(fall):<prob>
```

该修改只影响可视化标签，不改变模型、阈值、跟踪或分类逻辑。

### 9.3 无效输出删除

曾生成过两批真实视频 overlay 输出，但均判定为无效：

```text
outputs/real_test_overlay_20260620_042341
outputs/real_test_overlay_test4567_20260620_043330
```

原因是动作分类阶段未正常完成，`P(fall)` 没有可靠更新。上述输出目录已从服务器删除，避免后续误用。

### 9.4 推理脚本问题定位

第一处问题：

```text
cannot import name 'Compose' from 'mmaction.datasets.transforms'
```

原因：当前服务器的 MMAction2/MMEngine 版本中，`Compose` 应从 `mmengine.dataset` 导入。

修正：

```python
from mmengine.dataset import Compose
```

第二处问题：

```text
Expected input type to be list, but got <class 'torch.Tensor'>
```

原因：`model.test_step(...)` 需要经过 dataloader collate 后的输入格式。旧脚本手动对 `inputs` 做 `unsqueeze(0)` 后传入 tensor，导致 MMAction2 data preprocessor 拒绝。

修正：

```python
from mmengine.dataset import Compose, pseudo_collate

pipeline = Compose(val_pipeline_cfg)
data = pseudo_collate([pipeline(clip_sample.copy())])
result = model.test_step(data)[0]
```

同类修正已同步到：

```text
inference/batch_predict.py
inference/multitarget_realtime_demo.py
```

### 9.5 最小闭环验证

在服务器上用 `data/fall_binary_xsub.pkl` 的一个验证集摔倒样本做最小动作分类测试：

```text
sample: S001C001P003R001A043
label: 1
```

验证结果：

```text
predict_clip Pfall:       0.9993390440940857
cached_predictor Pfall:   0.9993390440940857
```

结论：

* 训练好的 checkpoint 可以正常返回摔倒概率。
* 问题发生在真实视频推理脚本的数据打包层，不是模型文件无法加载。
* 下一次重跑真实视频前，应先跑单个短 clip/单个视频片段 smoke test，确认日志中没有 `推理异常`，再批量处理完整视频。

### 9.6 下一次真实视频处理要求

下次运行时必须采用分阶段检查：

1. 先跑最小动作分类闭环，确认 `P(fall)` 可返回。
2. 再只跑一个短视频或短片段。
3. 频繁检查日志中是否出现 `推理异常`、`Traceback`、`Expected input`、`cannot import`。
4. 一旦出现异常，立即停止 `screen` 任务，修复代码后重跑。
5. 确认 smoke test 无异常后，再处理 `test4.mp4`、`test5.mp4`、`test6.mp4`、`test7.mp4`。

### 9.7 test4-test7 有效重跑结果

按用户要求删除旧的 `test4-test7` 输出后，重新处理：

```text
data/real_test/test4.mp4
data/real_test/test5.mp4
data/real_test/test6.mp4
data/real_test/test7.mp4
```

本次输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_test_overlay_test4567_20260620_044418
```

运行参数要点：

```text
checkpoint: work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth
pose weights: yolo26x-pose.pt
threshold: 0.5
infer-every: 2
max-persons: 5
imgsz: 640
```

日志检查结果：

* 没有 `推理异常`。
* 没有 `Traceback`。
* 没有 `Expected input`。
* 没有 `cannot import`。
* 没有后台 `screen` 任务残留。

汇总：

| Video | Status | Alerts | Event |
| --- | --- | ---: | --- |
| `test4.mp4` | ok | 0 | 无报警 |
| `test5.mp4` | ok | 1 | frame 186, track 2, `P(fall)=0.7350` |
| `test6.mp4` | ok | 1 | frame 223, track 1, `P(fall)=0.6244` |
| `test7.mp4` | ok | 0 | 无报警 |

事件日志：

```json
{"source": "data/real_test/test5.mp4", "frame_idx": 186, "track_id": 2, "fall_prob": 0.735, "event": "onset"}
{"source": "data/real_test/test6.mp4", "frame_idx": 223, "track_id": 1, "fall_prob": 0.6244, "event": "onset"}
```

输出文件：

```text
outputs/real_test_overlay_test4567_20260620_044418/videos/test4_overlay.mp4
outputs/real_test_overlay_test4567_20260620_044418/videos/test5_overlay.mp4
outputs/real_test_overlay_test4567_20260620_044418/videos/test6_overlay.mp4
outputs/real_test_overlay_test4567_20260620_044418/videos/test7_overlay.mp4
outputs/real_test_overlay_test4567_20260620_044418/summary.tsv
outputs/real_test_overlay_test4567_20260620_044418/events/
outputs/real_test_overlay_test4567_20260620_044418/snapshots/
outputs/real_test_overlay_test4567_20260620_044418/logs/
```

当前结论：

* 修复后的推理链路已能在真实视频上触发摔倒报警。
* 用户确认 `test4`、`test5`、`test6`、`test7` 都包含摔倒行为。
* 因此本次真实视频小样本初测中，`test5`、`test6` 检出摔倒，`test4`、`test7` 属于漏检。
* 该结果说明当前模型在项目验证集上表现很好，但真实手机竖屏短视频仍存在泛化和部署链路问题。

可能原因：

* 这批视频约 `60fps`，而模型使用 `clip_len=48`，在 60fps 下 48 帧只覆盖约 0.8 秒；如果训练数据更接近 25/30fps，真实时间窗口会明显变短。
* 部分摔倒动作持续时间可能太短，稳定落入 48 帧 track 缓冲区的有效骨骼序列不足。
* 竖屏高分辨率视频中人体姿态、遮挡、倒地后关键点质量和跟踪 ID 稳定性可能影响分类输入。
* 当前阈值 `0.5`、`infer-every=2` 和报警去抖策略可能对短促摔倒不够敏感。

后续改进空间：

* 对真实输入视频做时间归一化，例如先重采样到 30fps 后再推理，或在推理侧按真实时间窗口采样而不是固定原始帧数。
* 尝试更敏感的阈值和报警策略，例如降低阈值到 0.3-0.4、调整 `alert-k`、保留短时峰值概率。
* 将 `test4/test7` 及相似短促摔倒视频加入真实困难样本集，用于后续微调或重训。
* 在下次重跑前先做单视频 smoke test，重点观察骨骼质量、track 是否断裂、`P(fall)` 是否在摔倒瞬间出现短时峰值。

### 9.8 视频抽帧分析准备

为便于后续人工检查 `test4-test7` 的动作过程，已在服务器生成每个视频的 16 帧 contact sheet：

```text
/root/autodl-tmp/fall-detection/outputs/video_contact_sheets_20260620/test4_sheet.jpg
/root/autodl-tmp/fall-detection/outputs/video_contact_sheets_20260620/test5_sheet.jpg
/root/autodl-tmp/fall-detection/outputs/video_contact_sheets_20260620/test6_sheet.jpg
/root/autodl-tmp/fall-detection/outputs/video_contact_sheets_20260620/test7_sheet.jpg
```

当前尚未完成视觉分析，原因是从服务器拉取图片到本地查看时文件传输审批未通过。后续继续处理时，应优先查看这些 contact sheet 和 overlay 视频，判断 `test4/test7` 漏检是否来自：

* 摔倒动作持续时间过短。
* 60fps 导致 48 帧窗口覆盖真实时间过短。
* 姿态关键点质量不足或倒地后关键点丢失。
* track 断裂或未稳定累积到足够帧数。
* 当前报警阈值/去抖策略过保守。

### 9.9 contact sheet 人工观察结论

用户手动提供了 `test4-test7` 的 contact sheet 后，完成初步视觉观察：

* `test5.mp4`：检出成功。人物正面/斜正面入镜，摔倒过程从跑动、失衡到坐倒较清楚；人体框和关键点预期较稳定。报警发生在 `frame 186`，与画面中坐倒/接触地面的阶段一致。
* `test6.mp4`：检出成功。人物全身长期可见，摔倒后趴倒姿态持续较长，倒地后仍在画面内；报警发生在 `frame 223`，与画面中侧向摔倒/趴倒阶段一致。
* `test4.mp4`：漏检。视频是真摔倒，但属于冰面滑倒场景，人物穿黑色厚外套且戴帽，头颈和四肢关键点不明显；摔倒过程包含大量手撑地、半蹲、坐倒、翻身/趴倒动作，与训练集中标准“跌倒”动作形态可能差异较大。冰面高反光和黑衣服也可能降低姿态估计稳定性。
* `test7.mp4`：漏检。视频是真摔倒，但时长只有约 3.37 秒，夜间雪地场景，人物背对镜头且穿厚外套；摔倒发生很快，倒地时画面有明显运动模糊，后续人快速离开主要画面或只剩远处/模糊区域。该样本对 60fps + `clip_len=48` 的时间窗口尤其不友好。

人工观察后的判断：

* 这不是单纯“模型无效”，因为 `test5/test6` 已经能在真实视频上触发报警。
* `test4/test7` 是有价值的困难正样本，适合作为后续真实场景改进用例。
* 需要优先改推理侧的真实时间窗口和概率日志：当前 summary 只记录报警事件，未报警视频没有每次推理的 `P(fall)` 峰值，因此 `test4/test7` 的 `max_pfall=NA` 不能解释为模型概率一定为 0。
* 下一步应增加每次分类的 `P(fall)` 日志，按视频输出 max/mean/top-k 概率，再比较 60fps 原视频和重采样 30fps 后的结果。

## 10. 2026-06-20 真实视频推理策略升级合并记录

**执行人:** Codex
**本地目录:** `D:\AAA\基于深度学习的视频动作识别技术研究\fall-detection`

### 10.1 合并来源

用户提供了 Claude 生成的升级代码与说明，放在本地未跟踪目录 `add/` 中：

```text
add/08_真实视频推理与诊断.md
add/batch_predict.py
add/multitarget_realtime_demo.py
add/plot_prob_curves.py
add/realtime_core.py
add/run_real_video_eval.py
```

本次将有价值改动合并到正式项目路径，`add/` 目录本身仍保持未跟踪状态，不作为项目源码提交。

### 10.2 本次新增/修改文件

新增：

```text
docs/08_real_video_inference_diagnostics.md
inference/realtime_core.py
tools/run_real_video_eval.py
tools/plot_prob_curves.py
```

修改：

```text
README.md
inference/batch_predict.py
inference/multitarget_realtime_demo.py
Worklog.md
```

### 10.3 功能变化

推理侧新增真实视频诊断与策略能力：

* `TimeAwareBuffer`：支持按真实时间窗口缓存原始帧，再均匀采样到 `clip_len=48`，用于处理 60fps 手机视频中 48 帧只覆盖约 0.8 秒的问题。
* `TrackMerger`：支持在 ByteTrack ID 短时切换时，将刚消失 track 的 buffer 和概率状态继承给空间接近的新 track。
* `AlertPolicy`：新增单次高分 `high_single`、连续中分 `consec_mid`、最近窗口 `topk_mean` 三种报警策略并联。
* `ProbabilityLogger`：支持 `--prob-log`，每次动作分类都记录 raw/smoothed `P(fall)`，未报警视频也能看到概率峰值。
* `VideoSummaryBuilder`：支持 `--summary-json`，输出 max/mean/top-k 概率、报警事件、ID 合并次数和自动诊断标签。
* `tools/run_real_video_eval.py`：一键批量跑真实视频，收集 overlay、events、probs、summaries、summary.csv、failure_cases.csv 和 metrics.json。
* `tools/plot_prob_curves.py`：将概率日志画成曲线，便于诊断 test4/test7 这类漏检到底是阈值问题、ID 切换问题，还是模型全程低分。

### 10.4 合并时保留/修正的关键点

Claude 版本基于较早代码生成，本次合并时保留了上一次已经验证过的 MMAction2 修复：

```python
from mmengine.dataset import Compose, pseudo_collate
data = pseudo_collate([pipeline(clip_sample.copy())])
result = model.test_step(data)[0]
```

同时修正了 Claude 版本中的几个语义问题：

* `multitarget_realtime_demo.py` 的 `--time-window-sec` 默认改回 `0.0`，确保不传新参数时仍保持旧版 48 帧滚动缓冲行为。
* 新增 `--target-fps` 只用于在未显式给 `--time-window-sec` 时推导 `clip_len / target_fps` 的训练等效窗口，不再覆盖真实源视频 fps。
* `tools/run_real_video_eval.py` 不再把 `--target-fps 30` 错传为 `--source-fps 30`，避免 60fps 视频的 1.6 秒窗口被错误压回 48 原始帧。
* `batch_predict.py` 的 time-window 切窗按真实 `source_fps * time_window_sec` 计算窗口原始帧数，`target_fps` 不再用于缩短真实时间窗口。
* overlay 标签保留当前项目已修正样式：`id:<track_id> NORMAL/FALL P(fall):<prob>`。
* 实时脚本主循环发生异常时，写完 summary 后返回非零，方便批量评估识别失败视频。

### 10.5 本地验证

由于本机 PATH 中没有 `python`，使用 Codex 桌面内置 Python 做内存语法编译检查，避免写入 `__pycache__`：

```powershell
C:\Users\zz162\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -c "..."
```

验证文件：

```text
inference/batch_predict.py
inference/multitarget_realtime_demo.py
inference/realtime_core.py
tools/run_real_video_eval.py
tools/plot_prob_curves.py
```

结果：

```text
syntax ok: 5
```

### 10.6 后续建议

服务器同步后优先跑真实视频小实验：

```bash
python tools/run_real_video_eval.py \
    --video-dir data/real_test \
    --labels-csv data/real_test/labels.csv \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --out-dir outputs/real_eval/A_baseline
```

然后跑推荐组合：

```bash
python tools/run_real_video_eval.py \
    --video-dir data/real_test \
    --labels-csv data/real_test/labels.csv \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --out-dir outputs/real_eval/C_recommended \
    --time-window-sec 1.6 \
    --track-merge \
    --threshold 0.45 \
    --high-thr 0.7 \
    --topk-mean-thr 0.5
```

跑完后用 `tools/plot_prob_curves.py` 查看 `test4/test7` 的概率曲线，再决定是否需要把困难正样本加入微调/重训。

### 10.7 服务器首次运行暴露的 MMAction2 导入修复

服务器同步到 `079b5dc` 后首次运行 `tools/run_real_video_eval.py`，4 个视频均在加载动作模型阶段失败：

```text
ModuleNotFoundError: No module named 'mmaction.models.localizers.drn'
```

原因是评估脚本启动子进程时没有优先使用仓库内的 `mmaction2_src`，导致 Python 导入了环境里不完整的 pip 版 `mmaction`。修复方式：

* `inference/batch_predict.py` 的 `load_action_model()` 在导入 `mmaction.apis` 前，将仓库内 `mmaction2_src` 插入 `sys.path`。
* `tools/run_real_video_eval.py` 在 `subprocess.run(...)` 时显式给子进程传入包含 `mmaction2_src` 的 `PYTHONPATH`。

该修复只影响代码导入路径，不删除、不移动、不覆盖服务器上的 `work_dirs/`、checkpoint、`data/` 或已有 `outputs/`。

### 10.8 test4-test7 推荐策略服务器复测结果

服务器已同步到：

```text
e56cccd Use local MMAction2 source for inference
```

本次运行未删除服务器训练产物，未删除 `work_dirs/`、checkpoint、`data/` 或旧 `outputs/`。之前阻塞 `git pull` 的服务器本地代码冲突文件已保存为 stash：

```text
stash@{0}: On main: before-079b5dc-server-conflicts
```

本次输入目录使用软链接，只包含 `test4-test7`：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval_inputs/test4567
```

运行命令要点：

```bash
python tools/run_real_video_eval.py \
    --video-dir outputs/real_eval_inputs/test4567 \
    --labels-csv outputs/real_eval_inputs/test4567/labels.csv \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --out-dir outputs/real_eval/test4567_recommended_20260620_154749 \
    --time-window-sec 1.6 \
    --track-merge \
    --threshold 0.45 \
    --high-thr 0.7 \
    --topk-mean-thr 0.5 \
    --infer-every 2 \
    --max-persons 5
```

输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/test4567_recommended_20260620_154749
```

该目录约 `95 MB`，主要文件包括：

```text
summary.csv
failure_cases.csv
metrics.json
overlays/test4_overlay.mp4
overlays/test5_overlay.mp4
overlays/test6_overlay.mp4
overlays/test7_overlay.mp4
probs/test4_prob.jsonl
probs/test5_prob.jsonl
probs/test6_prob.jsonl
probs/test7_prob.jsonl
curves/test4.png
curves/test5.png
curves/test6.png
curves/test7.png
summaries/test4_summary.json
summaries/test5_summary.json
summaries/test6_summary.json
summaries/test7_summary.json
```

汇总指标：

```text
num_with_gt: 4
TP: 2
FP: 0
TN: 0
FN: 2
accuracy: 0.5
precision: 1.0
recall: 0.5
f1: 0.6667
```

逐视频结果：

| Video | GT | Diagnosis | Alerts | Max P(fall) | Mean top5 P(fall) | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `test4.mp4` | 1 | `model_unaware` | 0 | 0.1134 | 0.0496 | 全程低分，推荐作为困难正样本微调 |
| `test5.mp4` | 1 | `detected` | 1 | 0.9987 | 0.9979 | frame 182, track 2, event prob 0.7711 |
| `test6.mp4` | 1 | `detected` | 1 | 0.9998 | 0.9997 | frame 213, track 1, event prob 0.5675 |
| `test7.mp4` | 1 | `model_unaware` | 0 | 0.0569 | 0.0504 | 全程低分，推荐作为困难正样本微调 |

事件日志：

```json
{"source": "outputs/real_eval_inputs/test4567/test5.mp4", "frame_idx": 182, "track_id": 2, "fall_prob": 0.7711, "event": "onset"}
{"source": "outputs/real_eval_inputs/test4567/test6.mp4", "frame_idx": 213, "track_id": 1, "fall_prob": 0.5675, "event": "onset"}
```

概率曲线已生成：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/test4567_recommended_20260620_154749/curves/test4.png
/root/autodl-tmp/fall-detection/outputs/real_eval/test4567_recommended_20260620_154749/curves/test5.png
/root/autodl-tmp/fall-detection/outputs/real_eval/test4567_recommended_20260620_154749/curves/test6.png
/root/autodl-tmp/fall-detection/outputs/real_eval/test4567_recommended_20260620_154749/curves/test7.png
```

结论：

* 推理升级后的推荐策略仍检出 `test5/test6`，且概率峰值接近 1.0。
* `test4/test7` 即使使用 1.6 秒时间窗口、track merge、低阈值和 top-k 策略，最高 `P(fall)` 仍低于 0.12，属于模型不识别而不是阈值/去抖问题。
* 后续应把 `test4/test7` 作为真实困难正样本，配合真实困难负样本做微调；不建议继续只靠阈值扫描解决这两个样本。
* 下次启动训练时必须使用 `screen`，例如 `screen -S falldet-finetune`，方便网页端通过 `screen -x falldet-finetune` 查看训练进度。

# 项目交接摘要 (AI Handover)

本文档顶部是当前状态摘要；后面保留历史记录，只在需要追溯细节时阅读。

## 当前项目状态

* 项目路径：`D:\AAA\基于深度学习的视频动作识别技术研究\fall-detection`
* 服务器路径：`/root/autodl-tmp/fall-detection`
* 服务器环境：AutoDL / RTX 4090 / conda env `falldet`
* 主模型：PoseConv3D 二分类摔倒检测
* 推荐 checkpoint：`work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth`
* 训练状态：PoseConv3D 已完成 24 epoch；不要在未明确要求时重新训练。
* 重要保护：不要删除 `work_dirs/`、checkpoint、训练产物、服务器数据和用户未跟踪文件。

## 当前推理策略

主入口：

```text
inference/multitarget_realtime_demo.py
tools/run_real_video_eval.py
inference/realtime_core.py
```

当前真实视频推理已包含：

* `TimeAwareBuffer`：真实视频用 `--time-window-sec 1.6` 覆盖更完整动作。
* `AlertPolicy`：`--high-thr`、`--topk-mean-thr`、连续中阈值三类报警。
* `PoseHeuristicScorer`：紫框逻辑兜底，明确区分模型判断和工程逻辑。
* `lost_track_alert`：跌倒姿态后 pose/track 消失时报警。
* `track_merge` + `track_merge_same_frame`：处理 ID 切换和同帧拆分，尽量保持动作 clip 连续。
* stale overlay suppression：内部保留长 track 用于接力，但 overlay 只短暂绘制旧 track，避免旧框污染后续画面。
* `FallTrendDetector`：新增趋势/几何/消失/清理前复合检测，用于救 `elder_fall_7` 这类临界漏检。
* `PoseInterpolator` + `SimpleKalmanBoxTracker`：显式 `--pose-interp` 开关，短时跟丢时外推 COCO-17 骨架并衰减 keypoint score，目标是让 PoseConv3D buffer 不在 5-8 帧短缺口内断流。

颜色语义：

```text
green  = normal
red    = model fall, including model+logic/model+trend
orange = fall_trend/autopsy engineering fallback
purple = logic-only fallback
```

## 最近有效评估

上一轮服务器输出：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/elder_fall_falltrend_20260622_002414
```

结果：

```text
12 videos processed
failure_cases.csv is empty
metrics.json = {} because no labels CSV was supplied
elder_fall_7.mp4 was rescued by fall_trend at frame 131, max_pfall=0.361
```

`elder_fall_7` 的主要问题不是单纯骨架识别失败，而是摔倒过程短、ID/pose 连续性弱、模型 raw/heuristic 信号临界。`FallTrendDetector` 可以救这类临界漏检，但它是工程兜底，不是纯 PoseConv3D 模型识别能力。

## 分布式摄像头方案

新增 `deploy/`：本地电脑采集摄像头并跑轻量 YOLO Pose，只上传骨架 JSON 到服务器；服务器跑 PoseConv3D 和报警策略，只返回结果 JSON；本地合成 overlay。暂时只作为代码和文档同步，不需要现在部署。

## 2026-06-21 add/ 补充代码合并

从 `D:\AAA\基于深度学习的视频动作识别技术研究\add` 合并两组补充内容：

* `files-2` 检测策略：合入 `FallTrendDetector`、`recent_heuristics`、`recent_bboxes`、lost-track raw 概率修复、`--fall-trend` CLI 透传和 `tools/replay_fall_trend.py`。目标是救 `elder_fall_7` 这类 raw/heuristic 上升但绝对阈值临界的漏检。
* `files-1` 分布式部署：新增 `deploy/` WebSocket 服务端/客户端和 `docs/10_分布式部署_本地相机_远端推理.md`。这部分暂时只同步代码，不启动服务、不跑实时摄像头。
* 新增文档：`docs/09_fall_7漏检诊断与新策略.md`。

本地只读语法检查已通过：

```text
inference/multitarget_realtime_demo.py
inference/realtime_core.py
tools/run_real_video_eval.py
tools/replay_fall_trend.py
deploy/server.py
deploy/client.py
deploy/protocol.py
```

后续需要在服务器 `falldet` 环境做正式 `py_compile`，然后用 `--fall-trend` 重跑 `data/real_test/elder_fall`。

## 后续工作原则

* 不要把逻辑/趋势兜底说成模型能力提升；输出中必须保留红框/橙框/紫框语义。
* 不要为了旧框连续性恢复 `detector.snapshot()` 全量绘制，否则拼接视频会出现旧骨架长期残留。
* 今天不要直接更换 YOLO pose/track 模型；后续若评估更强检测/跟踪模型，必须先验证 COCO 17 点顺序、坐标格式、置信度分布和当前训练输入一致，避免破坏 PoseConv3D 的输入分布。
* 长任务和训练必须用 `screen`。
* 没有 labels CSV 时 `metrics.json={}` 是正常的；不能据此说 P/R/F1 不可算，只是缺少显式标签输入。
* 当前最严重的问题是 YOLO pose/track 在顽固视频里容易跟丢，导致 PoseConv3D 拿不到连续骨架；逻辑检测现阶段不作为主要增强方向。若后续重训，只在用户明确要求并准备困难样本后进行。
* `--pose-interp` 是实验性跟踪连续性增强，默认关闭；跑对比时必须在输出目录名里标明是否启用。

## 历史详细记录

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

### 10.9 test1-test3 与额外真实视频服务器复测结果

按用户要求，暂不重训；继续处理 `data/real_test` 中除 `test4-test7` 之外的真实视频。用户确认这 4 个视频也全部为摔倒正样本。

本次先删除旧 test4567 输出中的 overlay mp4，释放空间并避免混淆，但保留诊断证据文件：

* 已删除：
  * `/root/autodl-tmp/fall-detection/outputs/real_eval/test4567_recommended_20260620_154749/overlays/*.mp4`
  * `/root/autodl-tmp/fall-detection/outputs/real_test_overlay_test4567_20260620_044418/videos/*.mp4`
* 已保留：
  * `summary.csv`
  * `failure_cases.csv`
  * `metrics.json`
  * `probs/*.jsonl`
  * `curves/*.png`
  * `summaries/*.json`
  * snapshots 和事件日志

本次输入视频：

```text
data/real_test/2026-06-20 035837.mp4
data/real_test/test1.mp4
data/real_test/test2.mp4
data/real_test/test3.mp4
```

运行参数沿用推荐策略：

```bash
python tools/run_real_video_eval.py \
    --video-dir /tmp/falldet_other_inputs \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --out-dir outputs/real_eval/other_tests_recommended_20260620_160056 \
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
/root/autodl-tmp/fall-detection/outputs/real_eval/other_tests_recommended_20260620_160056
```

该目录约 `465 MB`，包含 overlay、events、probs、curves、summaries、`summary.csv`、`failure_cases.csv`、`metrics_all_fall.json` 和 `summary_all_fall.csv`。

按用户确认的全部正样本计算指标：

```text
TP=3
FP=0
TN=0
FN=1
accuracy=0.75
precision=1.0
recall=0.75
f1=0.8571
```

逐视频结果：

| Video | GT | Diagnosis | Alerts | Max P(fall) | Mean top5 P(fall) | Notes |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `2026-06-20 035837.mp4` | 1 | `detected` | 8 | 0.9966 | 0.9959 | 检出 |
| `test1.mp4` | 1 | `partial_signal` | 0 | 0.4460 | 0.3473 | 漏检，接近阈值但未触发 |
| `test2.mp4` | 1 | `detected` | 5 | 0.9996 | 0.9993 | 检出 |
| `test3.mp4` | 1 | `detected` | 11 | 0.9999 | 0.9999 | 检出 |

概率曲线已生成：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/other_tests_recommended_20260620_160056/curves/2026-06-20 035837.png
/root/autodl-tmp/fall-detection/outputs/real_eval/other_tests_recommended_20260620_160056/curves/test1.png
/root/autodl-tmp/fall-detection/outputs/real_eval/other_tests_recommended_20260620_160056/curves/test2.png
/root/autodl-tmp/fall-detection/outputs/real_eval/other_tests_recommended_20260620_160056/curves/test3.png
```

结论：

* 除 `test1` 外，其余 3 个额外正样本均被检出。
* `test1` 与 `test4/test7` 不同，不是完全 `model_unaware`，而是 `partial_signal`：最高 `P(fall)=0.4460`，接近当前 `threshold=0.45`，后续可通过更低阈值、top-k 策略或微调进一步处理。
* 当前真实正样本合并看：`test4-test7` 推荐策略检出 2/4，本轮额外视频检出 3/4；合计 5/8。

### 10.10 `2026-06-20 035837.mp4` 拼接视频诊断与第三版推理

用户补充说明：`2026-06-20 035837.mp4` 是多个视频片段拼接成的一分钟视频，因此跨片段出现不同人、不同视角、不同 track id 是合理现象；不能把所有跨片段 ID 跳变都归因于 ByteTrack 失败，也不应继续盲目增强跨片段 ID 合并。

第二版稳定 display id 复测输出：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/single_035837_stableid_20260620_162704
```

结果摘要：

```text
diagnosis: detected
num_alerts: 9
max_pfall: 0.9995
mean_pfall: 0.2546
num_unique_tracks: 13
num_id_switches_handled: 3
```

用户手动截图显示：香蕉皮式快摔片段中 `id:7` 基本稳定，骨架识别清楚，但模型 `P(fall)` 仍低（截图中约 0.00、0.03、0.08、0.01）。因此该片段的主要瓶颈不是骨架失败，也不是 ID 切换，而是 PoseConv3D 对这种快摔/翻倒姿态的泛化不足。

第三版代码升级：

* 新增 `PoseHeuristicScorer`，默认关闭，仅在显式传入 `--pose-heuristic-alert` 时启用。
* 启发式分数独立于模型 `P(fall)`，主要使用 COCO17 骨架几何信号：躯干倾斜、躯干倾斜变化、骨架宽高比/宽高比变化、髋部下落、腿部抬高。
* `ProbabilityLogger` 新增 `heuristic_score` 和 `heuristic_reason` 字段。
* `VideoSummaryBuilder` 新增 `max_pose_heuristic`、`topK_pose_heuristic`、`per_track_max_pose_heuristic`。
* `tools/run_real_video_eval.py` 支持透传 `--pose-heuristic-alert`、`--pose-heuristic-thr`、`--pose-heuristic-min-frames`。
* 事件日志同步记录触发 `reason`，便于区分 `high_single` 模型报警和 `pose_heuristic` 规则兜底。

第三版服务器单视频复测输出：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/single_035837_poseheur_v3_final_20260620_165510
```

运行参数：

```bash
python inference/multitarget_realtime_demo.py \
  --source "data/real_test/2026-06-20 035837.mp4" \
  --config configs/posec3d_fall_binary.py \
  --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
  --time-window-sec 1.6 \
  --track-merge \
  --threshold 0.45 \
  --high-thr 0.7 \
  --topk-mean-thr 0.5 \
  --infer-every 2 \
  --max-persons 5 \
  --pose-heuristic-alert \
  --pose-heuristic-thr 0.62 \
  --ground-truth 1 \
  --no-show
```

结果摘要：

```text
total_frames: 1812
total_inferences: 580
num_unique_tracks: 13
num_id_switches_handled: 3
num_alerts: 12
diagnosis: detected
max_pfall: 0.9995
mean_pfall: 0.2546
max_pose_heuristic: 1.0
mean_top5_pose_heuristic: 1.0
```

关键发现：

* 香蕉皮片段对应的 `track_id=7` 在 `frame=451` 被第三版启发式兜底触发，触发分数 `0.6808`，原因 `pose_heuristic:wide_delta=0.64,leg_raised=0.50`。
* 同一片段模型原始概率仍很低：`frame=451 raw_prob=0.025365, smoothed_prob=0.030157`，后续多帧 raw 仍低于 0.15。
* 第三版没有“修好模型认知”，而是在模型不认识该动作时，通过明确的姿态几何信号补上快速报警。
* 因为启发式可能增加误报风险，默认仍关闭；后续需要用非摔倒负样本（坐下、弯腰、跳跃、抬腿、运动、翻身等）专门评估 FP。

重要输出文件：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/single_035837_poseheur_v3_final_20260620_165510/overlays/2026-06-20_035837_overlay.mp4
/root/autodl-tmp/fall-detection/outputs/real_eval/single_035837_poseheur_v3_final_20260620_165510/events/2026-06-20_035837_events.jsonl
/root/autodl-tmp/fall-detection/outputs/real_eval/single_035837_poseheur_v3_final_20260620_165510/probs/2026-06-20_035837_prob.jsonl
/root/autodl-tmp/fall-detection/outputs/real_eval/single_035837_poseheur_v3_final_20260620_165510/summaries/2026-06-20_035837_summary.json
/root/autodl-tmp/fall-detection/outputs/real_eval/single_035837_poseheur_v3_final_20260620_165510/snapshots/2026-06-20_035837/fall_t7_f451.jpg
```

后续建议：

* 不再把该拼接视频的跨片段 ID 变化视作主要问题。
* 第三版启发式适合先作为可选部署兜底策略，不应替代模型训练指标。
* 若要把第三版变成默认策略，必须补充真实非摔倒负样本并统计 FP；否则保持 `--pose-heuristic-alert` 手动开启。
* 长期模型方案仍是把 `test4/test7/test1` 以及香蕉皮快摔片段作为真实困难正样本，配合困难负样本微调。

### 10.11 Overlay 颜色语义区分

按用户要求，更新多人叠加视频的框、骨架和标签语义：

* 绿色：`NORMAL`，未触发摔倒。
* 红色：`MODEL FALL`，由模型概率/模型报警策略触发，例如 `high_single`、`consec_mid`、`topk_mean`。
* 紫色：`LOGIC FALL`，由 `pose_heuristic` 姿态几何兜底触发。
* 主标签显示 `P`（模型概率）和必要时的 `H`（逻辑启发式分数）。
* 紫色逻辑报警时，额外放大显示 `LOGIC FALL H:<score>`，并在下一行显示触发参数名，例如 `wide_delta/leg_raised`，避免把规则兜底误读成纯模型输出。

该改动只影响可视化和控制台提示，不改变模型 checkpoint，不改变默认是否启用启发式兜底。

### 10.12 outputs 整理与彩色 overlay 全量重跑

服务器已将旧 `outputs` 内容整体归档，未删除任何训练产物、checkpoint、数据集或旧输出：

```text
/root/autodl-tmp/fall-detection/outputs/_archive_before_color_overlay_20260620_171257
```

随后用颜色语义区分后的 overlay 版本重跑 `data/real_test` 下 8 个测试视频（`test1.mp4` 到 `test8.mp4`）。本轮是 `model + pose_heuristic logic fallback` 部署效果，不是纯模型效果。

输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/all_tests_color_overlay_20260620_171342
```

运行参数：

```bash
python tools/run_real_video_eval.py \
  --video-dir data/real_test \
  --labels-csv data/real_test/labels_all_fall.csv \
  --config configs/posec3d_fall_binary.py \
  --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
  --out-dir outputs/real_eval/all_tests_color_overlay_20260620_171342 \
  --time-window-sec 1.6 \
  --track-merge \
  --threshold 0.45 \
  --high-thr 0.7 \
  --topk-mean-thr 0.5 \
  --infer-every 2 \
  --max-persons 5 \
  --pose-heuristic-alert \
  --pose-heuristic-thr 0.62
```

结果摘要（全部视频均按用户确认的摔倒正样本计算）：

```text
num_with_gt: 8
TP: 8
FP: 0
TN: 0
FN: 0
accuracy: 1.0
precision: 1.0
recall: 1.0
f1: 1.0
```

重要解释：

* 上述 8/8 是部署版本 `模型 + 逻辑兜底` 的结果，不能作为纯 PoseConv3D 模型效果。
* 例如 `test4/test7` 这类视频，模型原始 `P(fall)` 仍低，主要靠紫色 `LOGIC FALL` 兜底。
* 后续论文/汇报中应同时保留 `model-only` 和 `model+logic` 两套指标，避免把工程规则误写成模型泛化能力。

### 10.13 `50种摔倒方式_fall.MP4` 单视频重跑

按用户要求，单独重跑服务器上的长视频：

```text
/root/autodl-tmp/fall-detection/data/real_test/50种摔倒方式_fall.MP4
```

输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/fifty_fall_color_overlay_20260620_174107
```

输出文件：

```text
overlays/50_fall_overlay.mp4
events/50_fall_events.jsonl
probs/50_fall_prob.jsonl
summaries/50_fall_summary.json
snapshots/50_fall/
```

运行结果摘要：

```text
total_frames: 5586
total_inferences: 1690
num_unique_tracks: 42
num_id_switches_handled: 12
num_alerts: 43
alerted track count: 38
diagnosis: detected
max_pfall: 1.0
mean_pfall: 0.35
max_pose_heuristic: 1.0
mean_top5_pfall: 1.0
mean_top5_pose_heuristic: 1.0
```

解释：

* 这是 `model + pose_heuristic logic fallback` 部署版本结果。
* 该视频包含多段摔倒和多 track；既有红色 `MODEL FALL`，也有紫色 `LOGIC FALL`。
* 后续如需论文中的纯模型对照，应另跑一份不加 `--pose-heuristic-alert` 的 `model-only` 输出。

### 10.14 拼接视频残留骨架显示修复

用户反馈：拼接视频切换片段后，上一段中已经消失的人仍会在原位置残留骨架框，同时新片段里的人会以另一个 ID 出现。

原因定位：多人 track 内部会保留一段 `track_timeout`，用于 ID switch 合并和动作窗口连续性；但 overlay 绘制时直接使用了全部 `detector.snapshot()`，导致“内部保留但当前帧未观测到”的旧 track 也被画出来。

修复：

```text
inference/multitarget_realtime_demo.py
```

* 新增 `MultiTrackFallDetector.visible_snapshot(frame_idx, max_age_frames=0)`。
* 新增 CLI 参数 `--draw-track-max-age`，默认 `0`，表示只绘制当前帧真实观测到的 track。
* 主循环 overlay 改为绘制 `visible_tracks`，HUD 的 active 数也改为当前可见 track 数。
* 内部 stale track 仍保留给 ID 合并和动作缓冲，不影响已有 `--track-merge` 策略。

预期效果：拼接视频切到下一段后，旧片段人物不会继续残留在画面上；若需要更宽松的显示，可把 `--draw-track-max-age` 设为 1 或 2。

### 10.15 训练产物纳入 Git 恢复范围

按用户要求，将服务器上“训练得到的包”同步到本地仓库并准备推送 GitHub。纳入普通 Git 的目录：

```text
work_dirs/posec3d_fall_binary/
```

包含：

```text
best_acc_top1_epoch_5.pth
epoch_22.pth
epoch_23.pth
epoch_24.pth
训练日志 / 测试日志 / 预测 pkl / 评估图 / config 快照
```

未纳入普通 Git：

```text
data/
outputs/
yolo26x-pose.pt
mmaction2_src/
vis/
```

原因：这些是上游可下载/可再生成内容或大体积输出，其中 `yolo26x-pose.pt` 超过 GitHub 普通单文件 100 MB 限制。已新增 `ARTIFACTS_MANIFEST.md` 记录恢复边界和本地备份包位置。

### 10.16 `elder_fall` 新增真实视频批量重跑

按用户要求，服务器新增目录：

```text
/root/autodl-tmp/fall-detection/data/real_test/elder_fall
```

共 11 个视频：

```text
elder_fall_1.mp4
elder_fall_2.mp4
elder_fall_3.mp4
elder_fall_4.mp4
elder_fall_5.mp4
elder_fall_6.mp4
elder_fall_7.mp4
elder_fall_8.mp4
elder_fall_9.mp4
test8.mp4
test9.mp4
```

运行方式：使用 `screen` 启动一次批量任务，`tools/run_real_video_eval.py` 会在一个进程中顺序处理目录内所有视频。没有并行开多个 GPU 任务；单 4090 上并行跑多个视频通常会抢显存和降低吞吐。

正确环境：

```bash
/root/miniconda3/bin/conda run -n falldet python tools/run_real_video_eval.py ...
```

注意：base Python `/root/miniconda3/bin/python` 缺少 `cv2`，不能用于本项目推理。

最终有效输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/elder_fall_color_overlay_20260620_225822
```

本次参数：

```bash
conda run -n falldet python tools/run_real_video_eval.py \
  --video-dir data/real_test/elder_fall \
  --config configs/posec3d_fall_binary.py \
  --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
  --pose-weights yolo26x-pose.pt \
  --device cuda:0 \
  --out-dir outputs/real_eval/elder_fall_color_overlay_20260620_225822 \
  --timeout-sec 1800 \
  --time-window-sec 1.6 \
  --track-merge \
  --threshold 0.45 \
  --high-thr 0.7 \
  --topk-mean-thr 0.5 \
  --infer-every 2 \
  --max-persons 5 \
  --pose-heuristic-alert \
  --pose-heuristic-thr 0.62 \
  --draw-track-max-age 0
```

输出文件：

```text
overlays/*.mp4
events/*.jsonl
probs/*.jsonl
summaries/*.json
summary.csv
failure_cases.csv
metrics.json
```

因为本次未提供 labels.csv，`metrics.json` 为空 `{}`。若按目录名暂时把 11 个视频都当作摔倒正样本观察，则部署版 `model + pose_heuristic` 检出 9/11。

摘要：

```text
detected:
elder_fall_1, elder_fall_2, elder_fall_3, elder_fall_5, elder_fall_6,
elder_fall_8, elder_fall_9, test8, test9

not detected / needs review:
elder_fall_4: just_below_threshold, max_pfall=0.4616
elder_fall_7: model_unaware, max_pfall=0.2654
```

`elder_fall_4` 更像阈值/策略边缘样本；`elder_fall_7` 更像模型泛化不足样本，后续若微调，应优先纳入困难正样本池。

补充代码同步：`tools/run_real_video_eval.py` 已补充 `--draw-track-max-age` 透传，确保批量 overlay 同样使用“只绘制当前帧可见 track”的残影修复策略。

### 10.17 红框/紫框显示策略修正与 `elder_fall` 重跑

用户反馈：新版 overlay 观感变差，担心 `--draw-track-max-age 0` 导致框被漏画。复查代码与同一 `test8.mp4` 的新旧 summary 后确认：

* 模型输入、概率、报警事件没有变化。
* `test8.mp4` 新旧两次的 `max_pfall=0.9995`、`mean_pfall=0.2546`、`num_alerts=12`、track 列表均一致。
* 变差主要来自 overlay 可视化：只画当前帧可见 track 会让短暂漏检帧上的红/紫框消失，肉眼看像“检测变差”。

代码修正：

```text
inference/multitarget_realtime_demo.py
```

* `visible_snapshot()` 改为：普通 track 仍按 `--draw-track-max-age` 短缓冲显示。
* 但满足 fall 显示条件的 track 必须继续画出来：
  * `st.alerted`
  * `st.alert_frames_left > 0`
  * `st.smoothed_prob >= threshold`
  * `pose_heuristic` 分数超过阈值
* 默认 `--draw-track-max-age` 由 `0` 调整为 `3`，用于减少短时漏检闪断，同时避免拼接视频出现长时间普通旧骨架残留。

有效新输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/elder_fall_color_overlay_fallboxes_20260620_232720
```

本次仍处理 11 个 `elder_fall` 视频，summary 与上一版一致：部署版检测 9/11，`elder_fall_4` 为阈值边缘样本，`elder_fall_7` 为模型不敏感样本。区别在 overlay：红框/紫框不再因短暂不可见而被过滤掉。

服务器输出目录整理：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/MANIFEST_20260620.txt
```

保留有效输出：

```text
all_tests_color_overlay_20260620_171342
fifty_fall_color_overlay_20260620_174107
elder_fall_color_overlay_20260620_225822
elder_fall_color_overlay_fallboxes_20260620_232720
```

只把无关的 `.ipynb_checkpoints` 移入：

```text
outputs/real_eval/_archive_misc_20260620_233500
```

未删除任何 `work_dirs/`、checkpoint、data 或有效 overlay 结果。

### 10.18 还原下午 17:20-18:00 版本的 overlay 绘制方式

用户反馈：当前输出效果不如下午 17:20-18:00 左右版本，红框出现/持续显示效果变差。对比提交后定位到影响操作：

```text
368cf45 Track training artifacts and hide stale overlay tracks
```

该提交之后，overlay 绘制从：

```python
draw_multitrack_overlay(frame, detector.snapshot(), ...)
```

改成了：

```python
visible_tracks = detector.visible_snapshot(...)
draw_multitrack_overlay(frame, visible_tracks, ...)
```

这不会改变模型输入、`P(fall)`、报警事件或 summary，但会改变视频上哪些 track 被画出来。由于 `visible_snapshot()` 会过滤短时不可见/未重新观测到的 track，红框/紫框的视觉持续性不如下午版本。

已还原：

```text
inference/multitarget_realtime_demo.py
tools/run_real_video_eval.py
```

恢复为下午版本的绘制路径：

```text
draw_multitrack_overlay(frame, detector.snapshot(), ...)
```

并移除批量脚本对 `--draw-track-max-age` 的透传。后续 overlay 会重新画出 detector 内部保留的全部 track，红框显示效果应回到下午版本。代价是拼接视频中普通旧骨架残留也会回到下午版本的行为。

重跑 `elder_fall` 后的最新输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/elder_fall_color_overlay_legacydraw_20260620_235056
```

输出包含 11 个 overlay：

```text
elder_fall_1_overlay.mp4
elder_fall_2_overlay.mp4
elder_fall_3_overlay.mp4
elder_fall_4_overlay.mp4
elder_fall_5_overlay.mp4
elder_fall_6_overlay.mp4
elder_fall_7_overlay.mp4
elder_fall_8_overlay.mp4
elder_fall_9_overlay.mp4
test8_overlay.mp4
test9_overlay.mp4
```

summary 与前次一致：

```text
detected: 9/11
elder_fall_4.mp4: just_below_threshold, max_pfall=0.4616
elder_fall_7.mp4: model_unaware, max_pfall=0.2654
```

服务器 manifest 已更新，最新版指向：

```text
outputs/real_eval/elder_fall_color_overlay_legacydraw_20260620_235056
```

### 10.19 模型+逻辑同时触发时的 overlay 颜色规则

用户建议：当模型判断摔倒和逻辑兜底判断摔倒同时发生时，框应统一显示红色，并在文字中同时标明模型和逻辑都给出了摔倒结论。

已修改：

```text
inference/multitarget_realtime_demo.py
```

新的 overlay 颜色语义：

```text
green  = NORMAL
red    = MODEL FALL，或 MODEL+LOGIC FALL
purple = LOGIC FALL only
```

具体规则：

* 只要模型信号参与判断，框使用红色。
* 如果模型与逻辑同时成立，标签显示 `MODEL+LOGIC FALL P:<pfall> H:<heuristic>`。
* 只有纯 `pose_heuristic` 逻辑兜底、模型未参与时，才使用紫色 `LOGIC FALL`。
* HUD 右上角报警灯也改为：只要当前报警 track 中有模型信号，即显示红色；否则纯逻辑报警显示紫色。

该修改只影响 overlay 可视化颜色和文字，不改变模型输入、概率、报警事件或 summary。

### 10.20 跟踪丢失兜底策略

用户反馈：若老人转身、跌倒、躺地后 YOLO pose/ByteTrack 跟踪丢失，后续画面中不再有框，模型也可能没有足够连续骨架判断摔倒。

原因拆分：

* 若人还在画面里但换了 ID，这是跟踪关联问题，可通过 `--track-merge`、更长 `--track-merge-gap`、更长 `--track-timeout` 缓解。
* 若老人已经躺倒，YOLO pose 本身不再输出人体骨架，这是检测层丢失。PoseConv3D 没有新骨架输入，无法继续靠模型判断。
* 因此要在工程层增加“跌倒姿态后 track 消失”的显式兜底，并在结果中标明这是逻辑兜底。

代码新增：

```text
inference/multitarget_realtime_demo.py
tools/run_real_video_eval.py
```

新增参数：

```text
--lost-track-alert
--lost-track-min-gap
--lost-track-heuristic-thr
--lost-track-model-thr
```

逻辑：

* 某个 track 已经有模型/骨架逻辑的跌倒前兆；
* 随后连续 `lost_track_min_gap` 帧没有被 pose 检测重新观测到；
* 若消失前 `heuristic_score >= lost_track_heuristic_thr` 或 `smoothed_prob >= lost_track_model_thr`，触发 `track_lost_after_fall_pose` 报警；
* 报警事件、summary、overlay 都记录该原因。

同时修复批量脚本没有透传 `--track-timeout` 的问题，后续可针对老人监控/躺倒场景把 track 保留时间调长，例如：

```bash
--track-timeout 120 \
--lost-track-alert \
--lost-track-min-gap 8 \
--lost-track-heuristic-thr 0.45 \
--lost-track-model-thr 0.35
```

注意：这是工程逻辑兜底，不是模型本体能力提升。若从头到尾都没有检测到人体骨架，仍需换更强 pose detector、调低 `--conf`、提高 `--imgsz`，或增加人体检测/分割兜底。

### 10.21 Overlay 旧框残留修正

用户反馈：当前人物已经离开画面后，识别框和骨架仍会停留很久，尤其在多人或拼接视频中会严重干扰后续画面判断。

原因确认：

* 之前为了恢复 17:20-18:00 版本的红框连续性，把 overlay 绘制恢复为 `detector.snapshot()`，即绘制 detector 内部保留的全部 track。
* 后续批跑又将 `--track-timeout` 提高到 120，用于增强 ID 合并、跌倒后丢跟兜底和跟踪连续性。
* 因此问题不在于当前跟踪能力差，而在于“内部保留 track”和“画面继续显示旧 track”没有拆开。

已修改：

```text
inference/multitarget_realtime_demo.py
tools/run_real_video_eval.py
```

新的处理方式：

* 保留现有跟踪能力：`track_timeout`、`track_merge`、`lost_track_alert`、模型历史窗口和 summary 逻辑不变。
* 新增仅用于 overlay 的 `visible_snapshot(frame_idx, max_age, alert_max_age)`。
* 普通 track 如果连续若干帧未被重新观测到，就不再画在视频上。
* 刚报警的 track 可比普通 track 多保留几帧，避免摔倒红框刚出现就闪断。
* 批量脚本新增透传：

```text
--draw-track-max-age
--draw-alert-max-age
```

默认建议：

```text
--track-timeout 120          # 内部跟踪/合并/丢跟兜底继续保留较长历史
--draw-track-max-age 8       # 普通旧框不长期残留
--draw-alert-max-age 15      # 报警框短暂保留，避免视觉闪烁
```

该修改只影响 overlay 可视化是否继续绘制旧 track，不改变模型输入、P(fall)、报警事件、summary 或训练 checkpoint。

### 10.22 同帧 ID 拆分接力

用户截图确认了另一类失败：同一个老人摔倒过程中，YOLO/ByteTrack 会在同一帧同时给出两个 ID。例如旧 `id:1` 仍覆盖身体一部分，新 `id:5` 覆盖已经躺倒的人体，导致新 ID 没有继承旧 ID 的站立、失衡、下落过程，PoseConv3D 只能看到短片段或躺倒后的片段，`P(fall)` 容易很低。

已有逻辑只能处理“旧 ID 消失后，新 ID 再出现”的接力：

```text
_try_adopt_recent_inactive_track(...)
```

但截图属于“旧 ID 和新 ID 同帧共存”的拆分，因此新增显式开关：

```text
--track-merge-same-frame
```

新增处理：

* 在同一帧中，如果新 track 刚出现，会检查已经处理过的当前帧 track。
* 只有同时满足空间证据和摔倒前兆才合并，避免多人场景误合并：
  * bbox IoU 达到较低但有效的重叠阈值，或小框被大框明显包含，或两框非常近且呈横向/低姿态；
  * 旧 track 已有 `heuristic_score >= 0.35`、`P(fall) >= 0.20`，或新旧 bbox 呈明显横向躺倒形态。
* 合并时，新 ID 继承旧 ID 的 display id、TimeAwareBuffer、概率状态、启发式状态和报警状态。
* 被合并的旧碎片 track 从当前内部 tracks 中移除，避免 overlay 同时画出同一个人的两个框。

这次修改不关闭或削弱现有 `track_timeout=120`、`track_merge`、`lost_track_alert`。它是对“同帧拆分”的补充，目标是让模型拿到更连续的摔倒动作输入，而不是只靠逻辑兜底。

### 10.23 elder_fall 同帧接力版本重跑

服务器已拉取：

```text
aa202df Merge same-frame split tracks
```

已用 `screen` 重跑 `data/real_test/elder_fall` 全目录 12 个视频，输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/elder_fall_sameframe_merge_20260621_014000
```

关键参数：

```bash
--time-window-sec 1.6 \
--track-merge \
--track-merge-same-frame \
--track-merge-gap 45 \
--track-timeout 120 \
--draw-track-max-age 8 \
--draw-alert-max-age 15 \
--threshold 0.45 \
--high-thr 0.7 \
--topk-mean-thr 0.5 \
--infer-every 2 \
--max-persons 5 \
--conf 0.15 \
--imgsz 960 \
--pose-heuristic-alert \
--pose-heuristic-thr 0.62 \
--lost-track-alert \
--lost-track-min-gap 8 \
--lost-track-heuristic-thr 0.45 \
--lost-track-model-thr 0.35
```

运行结果：

```text
12 overlays
12 summaries
summary.csv / failure_cases.csv / metrics.json 均已生成
metrics.json = {}，因为未提供 labels CSV
```

按部署诊断字段看：

```text
detected: 11/12
partial_signal: 1/12
```

唯一问题视频：

```text
elder_fall_7.mp4: partial_signal, num_alerts=0, max_pfall=0.361, max_pose_heuristic=0.4487
```

其他重点结果：

```text
50_fall.MP4: detected, num_alerts=42, max_pfall=1.0, num_id_switches_handled=15
elder_fall_4.mp4: detected, max_pfall=0.6474
elder_fall_8.mp4: detected, max_pfall=0.1033, max_pose_heuristic=0.5756, num_id_switches_handled=1
test8.mp4: detected, num_alerts=9, max_pfall=1.0, num_id_switches_handled=61
test9.mp4: detected, num_alerts=8, max_pfall=1.0, num_id_switches_handled=44
```

解释：

* 同帧接力主要针对“同一个人同一帧被拆成两个 ID”的情况，可以减少动作 clip 被切碎。
* 对 `elder_fall_7` 仍未完全解决，说明该视频更接近模型/骨架信号都偏弱的困难样本；后续若要继续提高，需要单独调低逻辑阈值做部署兜底，或把它作为困难正样本进入微调数据。
* 本次没有训练，也没有删除任何 checkpoint、训练目录或旧输出。

### 10.24 FallTrendDetector 评估与颜色语义修正

服务器已在 `--fall-trend` 策略下重跑 `data/real_test/elder_fall`，输出目录：

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/elder_fall_falltrend_20260622_002414
```

结果摘要：

```text
12 videos processed
failure_cases.csv is empty
metrics.json = {} because no labels CSV was supplied
elder_fall_7.mp4: fall_trend rescued at frame 131, max_pfall=0.361
```

重要解释：

* `fall_trend` / `autopsy` 是工程趋势兜底，不是纯 PoseConv3D 模型输出。
* overlay 颜色语义已修正：红色表示模型触发或模型同时触发；橙色表示 `fall_trend` / `autopsy` 趋势兜底；紫色表示纯 `pose_heuristic` / `track_lost_after_fall_pose` 逻辑兜底；绿色表示正常。
* 若模型判断和趋势/逻辑同时发生，框仍为红色，但标签会写出 `MODEL+TREND FALL` 或 `MODEL+LOGIC FALL`。
* 后续如研究更强 YOLO pose/track 或跟踪接力方案，必须保证输出骨架格式、COCO 17 点顺序、坐标尺度和置信度分布与当前 PoseConv3D 训练输入兼容。
* 当前最严重问题是跟踪模型容易跟丢，逻辑检测现阶段没必要继续加强。

### 10.25 files-3 跟踪连续性增强合并

从 `D:\AAA\基于深度学习的视频动作识别技术研究\add\files-3` 合并跟踪连续性相关内容：

```text
docs/12_跟踪连续性强化.md
inference/realtime_core.py
inference/multitarget_realtime_demo.py
tools/run_real_video_eval.py
deploy/server.py
deploy/client.py
```

本轮改动：

* 新增 `SimpleKalmanBoxTracker`：记录 bbox 中心速度，短时跟丢时预测当前 bbox。
* 新增 `PoseInterpolator`：在 `--pose-interp` 开启时，对短时缺失 track 外推 COCO-17 骨架，keypoint score 按帧衰减，避免把外推帧伪装成高置信真实检测。
* `TrackMerger` 支持用 tombstone 的 Kalman 预测位置匹配新 ID，增强短时丢失后的同人接力。
* `run_real_video_eval.py` 透传 `--pose-interp` 参数，便于批量跑同一目录做对比。
* 分布式 server/client 增加 `alert_source` 和外推状态显示字段。
* 保持 YOLO pose/track 模型不变；`--pose-interp` 默认关闭，批量对比时显式开启。
* 保留红/橙/紫语义：只有模型参与时红框；纯 `fall_trend/autopsy` 橙框；纯传统逻辑兜底紫框。

本地检查：

```text
python -m py_compile inference/realtime_core.py inference/multitarget_realtime_demo.py tools/run_real_video_eval.py deploy/server.py deploy/client.py deploy/protocol.py
PoseInterpolator smoke test: kpts=(17,2), scores=(17,), bbox=(4,), mean_score=0.85
```

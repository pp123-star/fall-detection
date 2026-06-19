# 01 环境搭建(AutoDL / 云GPU)

## 一、选机器

### 1.1 AutoDL 实例规格建议

| 配置项 | 推荐 | 备注 |
|---|---|---|
| GPU | RTX 4090 / RTX 3090 / A40 | 24GB 显存即可 |
| 镜像 | `PyTorch 2.1.0` + `Python 3.10` + `CUDA 11.8` | **关键**,见下文 |
| 系统盘 | 30 GB | 默认 |
| 数据盘 | 50 GB | NTU pickle + 中间产物 |
| 计费 | 按量(¥2-3/h)或包日 | 训练时长约 3-10 小时 |

### 1.2 为什么选 PyTorch 2.1 + CUDA 11.8

- MMCV 2.x 官方有 `cu118 + torch2.1` 的预编译 wheel,**避免源码编译**
- Ultralytics `ultralytics>=8.3` 完美支持 CUDA 11.8
- AutoDL 默认镜像里这套组合最稳定,新建实例直接选这个镜像即可

**不要选**:
- ❌ PyTorch 1.x(MMCV 旧 API,会和 mmaction2 v1.x 冲突)
- ❌ CUDA 12.x(MMCV 预编译 wheel 仍在追赶,可能要源码编译)
- ❌ Python 3.11+(部分 mmcv wheel 暂未发布)

## 二、一键环境脚本

进入实例后,在 `~/autodl-tmp/`(数据盘)目录下:

```bash
cd ~/autodl-tmp
git clone <你的仓库地址> fall-detection  # 或直接 scp 上传
cd fall-detection
bash env/setup_autodl.sh
```

`setup_autodl.sh` 做的事:
1. 创建 conda 环境 `falldet`(Python 3.10)
2. 装 PyTorch 2.1.0 + cu118
3. 用 `openmim` 装齐 mmengine / mmcv / mmaction2 / mmdet / mmpose
4. 装 ultralytics(YOLO26-Pose)
5. 验证安装

执行后会打印一段"全部 OK"或者具体哪一步失败,失败的话看下面排错章节。

## 三、手动安装步骤(setup_autodl.sh 内容拆解,出错时按这个查)

### Step 1. 创建并激活 conda 环境

```bash
# 如果用的是 AutoDL 默认 Miniconda
conda create -n falldet python=3.10 -y
conda activate falldet
```

### Step 2. 安装 PyTorch 2.1.0 + CUDA 11.8

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

验证:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 预期输出: 2.1.0+cu118 True NVIDIA GeForce RTX 4090
```

### Step 3. 装 OpenMMLab 工具链

**这里有个坑**:必须按下面这个顺序,且版本号要对齐。

```bash
# 装 mim,统一管理 openmmlab 包
pip install -U openmim

# 用 mim 装(它会自动选对应 torch+cuda 的 wheel)
mim install mmengine
mim install "mmcv==2.1.0"             # 关键: 不要装 2.2+,有概率和 mmaction2 v1.2 冲突
mim install "mmdet>=3.0.0,<3.3.0"
mim install "mmpose>=1.0.0,<2.0.0"
mim install "mmaction2>=1.2.0,<2.0.0"
```

验证:
```bash
python -c "
import mmcv, mmengine, mmaction
print('mmcv:', mmcv.__version__)
print('mmengine:', mmengine.__version__)
print('mmaction:', mmaction.__version__)
"
# 预期:
# mmcv: 2.1.0
# mmengine: 0.10.x
# mmaction: 1.2.x
```

### Step 4. 装 ultralytics(YOLO26-Pose)

```bash
pip install "ultralytics>=8.3.222"
```

验证:
```bash
python -c "from ultralytics import YOLO; print('ultralytics OK')"
```

第一次用模型时会自动下载权重到 `~/.config/Ultralytics/`,如果实例没外网或速度慢:
```bash
# 手动下到本地
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26x-pose.pt
# 然后 YOLO('yolo26x-pose.pt') 即可
```

### Step 5. 装辅助包

```bash
pip install \
    scikit-learn \
    seaborn \
    matplotlib \
    rich \
    tqdm \
    opencv-python \
    decord \
    pandas
```

## 四、用 mmaction2 的两种方式

### 方式 A:pip 安装(简单,推荐先用这个)

上面 `mim install mmaction2` 就是这种,直接 `import mmaction` 即可。

### 方式 B:源码安装(改 mmaction 内部代码时用)

```bash
cd ~/autodl-tmp
git clone https://github.com/open-mmlab/mmaction2.git -b main
cd mmaction2
pip install -v -e .
```

源码安装的好处是你能直接用 `mmaction2/tools/train.py` 这个脚本,我们的配置文件也是按这个用法写的。

**强烈建议两种都装**:pip 装好包供 `import`,源码 clone 一份用 `mmaction2/tools/train.py` 启动训练。

```bash
# 拉源码(仅用于 tools/ 脚本)
cd ~/autodl-tmp
git clone https://github.com/open-mmlab/mmaction2.git mmaction2_src
# 之后训练命令里的 python 调用 mmaction2_src/tools/train.py
```

## 五、常见安装错误与解决

### 错误 1:`mmcv` import 时报 CUDA op 找不到

原因:你装了纯 Python 版 `mmcv`,但 mmaction2 某些 op 需要 CUDA op。

解决:
```bash
# 卸载重装
pip uninstall mmcv mmcv-full mmcv-lite -y
mim install "mmcv==2.1.0"   # mim 会自动选对版本
```

### 错误 2:`No module named 'mmaction.datasets.pipelines'`

原因:你装了 mmaction 0.x 旧版,API 完全不同。

解决:
```bash
pip uninstall mmaction2 -y
mim install "mmaction2>=1.2.0,<2.0.0"
```

### 错误 3:`torch.cuda.is_available() == False`

原因:CPU 版 PyTorch。

解决:重装明确带 cu118 的版本:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 错误 4:NumPy 2.x 不兼容

PyTorch 2.1 默认会装上 NumPy 2.x,但 mmaction2 1.2 在 NumPy 2.x 上有概率出现 `np.float` 已废弃报错。

解决:固定 NumPy 1.x:
```bash
pip install "numpy<2.0"
```

### 错误 5:`Decord ImportError`(libGL.so.1 找不到)

云镜像缺少图形库依赖。

解决:
```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

### 错误 6:HTTPS 下载 NTU pickle 速度慢

OpenMMLab 的 download.openmmlab.com 在国内速度有时不稳,可用以下加速方案:
```bash
# 方案 1: 用国内镜像源 pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 方案 2: 下载文件用 aria2c 多线程
apt-get install -y aria2
aria2c -x 8 -s 8 https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
```

## 六、验证总体环境

完整跑通这段代码,应该全部 OK 才算环境装好:

```python
# env/verify.py(脚本里也有)
import torch
import mmengine
import mmcv
import mmaction
from ultralytics import YOLO
import numpy as np
import cv2

print("=" * 50)
print(f"PyTorch:     {torch.__version__}")
print(f"CUDA OK:     {torch.cuda.is_available()}")
print(f"GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
print(f"mmengine:    {mmengine.__version__}")
print(f"mmcv:        {mmcv.__version__}")
print(f"mmaction:    {mmaction.__version__}")
print(f"numpy:       {np.__version__}")
print(f"opencv:      {cv2.__version__}")
print("=" * 50)

# 试一下 PoseConv3D 模型能否被构建
from mmengine.registry import MODELS
from mmaction.registry import MODELS as MMA_MODELS
print("MMAction MODELS registry size:", len(MMA_MODELS.module_dict))
print("✓ All imports OK")
```

跑:
```bash
python env/verify.py
```

---

下一篇:`02_data_preparation.md`

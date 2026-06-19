#!/bin/bash
# =============================================================================
# AutoDL / 云GPU 实例一键环境搭建
# 用法: bash env/setup_autodl.sh
# 适用: PyTorch 2.1 + CUDA 11.8 + Python 3.10
# =============================================================================

set -e  # 任一命令失败立即退出

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================
# Step 0. 系统级依赖(图形库,decord 视频读取要用)
# ============================================================
log_info "Step 0: 装系统级图形库"
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq libgl1 libglib2.0-0 wget aria2 git || true
else
    log_warn "非 Debian 系系统,跳过 apt 安装(请手动确保 libGL 已装)"
fi

# ============================================================
# Step 1. 创建 conda 环境
# ============================================================
log_info "Step 1: 创建 conda 环境 falldet (Python 3.10)"

# 检查 conda 是否在 PATH
if ! command -v conda &> /dev/null; then
    log_error "conda 未找到。请先确保 Miniconda/Anaconda 已装并在 PATH 里。"
    exit 1
fi

# 初始化 conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# 如果已存在环境,询问是否重建
if conda env list | grep -q "^falldet"; then
    log_warn "环境 falldet 已存在,跳过创建。如需重建,先执行 conda env remove -n falldet"
else
    conda create -n falldet python=3.10 -y
fi

conda activate falldet
log_info "已激活环境: $(which python)"

# ============================================================
# Step 2. 装 PyTorch 2.1 + CUDA 11.8
# ============================================================
log_info "Step 2: 安装 PyTorch 2.1.0 + cu118"
pip install --quiet \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证 CUDA
python -c "
import torch
assert torch.__version__.startswith('2.1.'), f'PyTorch 版本不对: {torch.__version__}'
assert torch.cuda.is_available(), 'CUDA 不可用'
print(f'  -> PyTorch {torch.__version__}, GPU = {torch.cuda.get_device_name(0)}')
"

# ============================================================
# Step 3. 装 OpenMMLab 工具链
# ============================================================
log_info "Step 3: 装 OpenMMLab 工具链(mim 自动选对应版本)"

pip install --quiet -U openmim
mim install --quiet mmengine
mim install --quiet "mmcv==2.1.0"
mim install --quiet "mmdet>=3.0.0,<3.3.0"
mim install --quiet "mmpose>=1.0.0,<2.0.0"
mim install --quiet "mmaction2>=1.2.0,<2.0.0"

python -c "
import mmcv, mmengine, mmaction, mmdet, mmpose
print(f'  -> mmcv {mmcv.__version__}, mmengine {mmengine.__version__}')
print(f'  -> mmaction {mmaction.__version__}, mmdet {mmdet.__version__}, mmpose {mmpose.__version__}')
"

# ============================================================
# Step 4. 装 ultralytics(YOLO26-Pose 用)
# ============================================================
log_info "Step 4: 装 ultralytics"
pip install --quiet "ultralytics>=8.3.222"
python -c "from ultralytics import YOLO; print('  -> ultralytics OK')"

# ============================================================
# Step 5. 装辅助库
# ============================================================
log_info "Step 5: 装辅助库"
pip install --quiet \
    "numpy<2.0" \
    scikit-learn \
    seaborn \
    matplotlib \
    rich \
    tqdm \
    opencv-python \
    decord \
    pandas

# ============================================================
# Step 6. 克隆 mmaction2 源码(为了用它的 tools/train.py)
# ============================================================
log_info "Step 6: 克隆 mmaction2 源码到 ./mmaction2_src(为了用 tools/train.py)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ ! -d "mmaction2_src" ]; then
    git clone --depth 1 --branch main https://github.com/open-mmlab/mmaction2.git mmaction2_src
    log_info "  -> 已克隆到 $(pwd)/mmaction2_src"
else
    log_warn "  -> mmaction2_src 已存在,跳过"
fi

# ============================================================
# Step 7. 整体验证
# ============================================================
log_info "Step 7: 整体验证"
python "$SCRIPT_DIR/verify.py"

# ============================================================
echo ""
log_info "=================================================="
log_info "✓ 环境搭建完成!"
log_info "  下次进入实例后,执行:"
log_info "    conda activate falldet"
log_info "  即可。"
log_info "=================================================="

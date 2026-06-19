#!/bin/bash
# =============================================================================
# tools/train.sh — 启动训练
# 用法:
#   bash tools/train.sh <CONFIG_PATH> [<NUM_GPUS>]
# 示例:
#   bash tools/train.sh configs/posec3d_fall_binary.py 1     # 单卡
#   bash tools/train.sh configs/stgcnpp_fall_binary.py 2     # 双卡
# =============================================================================

set -e

CONFIG=${1:?"用法: bash tools/train.sh <config> [num_gpus]"}
GPUS=${2:-1}

# 找到 mmaction2 源码路径
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMACTION_DIR="$PROJECT_ROOT/mmaction2_src"

if [ ! -d "$MMACTION_DIR" ]; then
    echo "错误: 未找到 mmaction2 源码 $MMACTION_DIR"
    echo "请先 git clone https://github.com/open-mmlab/mmaction2.git mmaction2_src"
    exit 1
fi

TRAIN_PY="$MMACTION_DIR/tools/train.py"
DIST_PY="$MMACTION_DIR/tools/dist_train.sh"

# 设置 PYTHONPATH(确保用 mmaction2_src 里的代码)
export PYTHONPATH="$MMACTION_DIR:$PYTHONPATH"

# 设置随机种子(可复现)
export PYTHONHASHSEED=42

cd "$PROJECT_ROOT"

if [ "$GPUS" -eq 1 ]; then
    echo "==============================================================="
    echo "[INFO] 单卡训练"
    echo "  config: $CONFIG"
    echo "  workdir: $(grep '^work_dir' $CONFIG | head -1)"
    echo "==============================================================="
    python "$TRAIN_PY" "$CONFIG" \
        --seed 42 \
        --deterministic
else
    echo "==============================================================="
    echo "[INFO] 分布式训练 (GPUs=$GPUS)"
    echo "  config: $CONFIG"
    echo "==============================================================="
    bash "$DIST_PY" "$CONFIG" "$GPUS" \
        --seed 42 \
        --deterministic
fi

echo ""
echo "[INFO] 训练完成,checkpoint 在 work_dirs/ 下"

#!/bin/bash
# =============================================================================
# tools/test.sh — 启动测试(用 best checkpoint)
# 用法:
#   bash tools/test.sh <CONFIG> <CHECKPOINT>
# 示例:
#   bash tools/test.sh configs/posec3d_fall_binary.py work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18.pth
# =============================================================================

set -e

CONFIG=${1:?"用法: bash tools/test.sh <config> <checkpoint>"}
CKPT=${2:?"用法: bash tools/test.sh <config> <checkpoint>"}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMACTION_DIR="$PROJECT_ROOT/mmaction2_src"

if [ ! -d "$MMACTION_DIR" ]; then
    echo "错误: 未找到 mmaction2 源码"; exit 1
fi

export PYTHONPATH="$MMACTION_DIR:$PYTHONPATH"

cd "$PROJECT_ROOT"

# 输出预测结果到 pickle,以便 eval_binary_metrics.py 计算混淆矩阵/F1
OUT_PKL="${CKPT%.pth}_pred.pkl"

python "$MMACTION_DIR/tools/test.py" "$CONFIG" "$CKPT" \
    --dump "$OUT_PKL"

echo ""
echo "[INFO] 测试完成,预测结果在 $OUT_PKL"
echo "[INFO] 接下来执行二分类细致评估:"
echo "  python tools/eval_binary_metrics.py --pred $OUT_PKL --config $CONFIG"

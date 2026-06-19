#!/bin/bash
# scripts/run_all.sh — 一键串联整个流程(从环境到训练到评估)
#
# 用法:
#   bash scripts/run_all.sh                # 跑完整 pipeline
#   bash scripts/run_all.sh --skip-env     # 已搭好环境,跳过环境步骤
#   bash scripts/run_all.sh --quick        # 跑一个最小流程(用 ST-GCN++ 半小时出结果)
#   bash scripts/run_all.sh --posec3d-only # 只跑主线模型,跳过对比模型
#
# 出错就 fail-fast,不会接着糟蹋时间。

set -euo pipefail
trap 'echo -e "\n[run_all] 出错于第 $LINENO 行,中断"; exit 1' ERR

# ============================================================
# 参数解析
# ============================================================
SKIP_ENV=0
QUICK=0
POSEC3D_ONLY=0
STGCN_ONLY=0
GPUS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-env) SKIP_ENV=1; shift;;
        --quick) QUICK=1; shift;;
        --posec3d-only) POSEC3D_ONLY=1; shift;;
        --stgcn-only) STGCN_ONLY=1; shift;;
        --gpus) GPUS="$2"; shift 2;;
        -h|--help)
            grep "^#" "$0" | head -20
            exit 0;;
        *) echo "未知参数: $1"; exit 1;;
    esac
done

# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[run_all] 项目根目录:$PROJECT_ROOT"

POSEC3D_CFG="configs/posec3d_fall_binary.py"
STGCN_CFG="configs/stgcnpp_fall_binary.py"
POSEC3D_DIR="work_dirs/posec3d_fall_binary"
STGCN_DIR="work_dirs/stgcnpp_fall_binary"

# ============================================================
# 工具函数
# ============================================================
section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# 找 best ckpt(配合 default_runtime.py 的 save_best='acc/top1')
find_best_ckpt() {
    local work_dir="$1"
    local ckpt
    ckpt=$(ls -t "${work_dir}"/best_acc_top1_*.pth 2>/dev/null | head -1 || true)
    if [[ -z "$ckpt" ]]; then
        # 兜底:用最后一次 epoch
        ckpt=$(ls -t "${work_dir}"/epoch_*.pth 2>/dev/null | head -1 || true)
    fi
    echo "$ckpt"
}

# ============================================================
# 0. 环境
# ============================================================
if [[ $SKIP_ENV -eq 0 ]]; then
    section "[0/6] 环境搭建"
    bash env/setup_autodl.sh
    python env/verify.py
else
    echo "[0/6] 跳过环境搭建(--skip-env)"
fi

# ============================================================
# 1. 数据
# ============================================================
section "[1/6] 数据下载"
if [[ ! -f "data/ntu60_2d.pkl" ]]; then
    python data_prep/download_pkl.py
else
    echo "[1/6] data/ntu60_2d.pkl 已存在,跳过下载"
fi

section "[1.5/6] 构建二分类数据集"
NEG_STRAT="hard"        # 难负样本:坐下/起身/staggering 等
NEG_RATIO=3             # 负:正 = 3:1
SUBSAMPLE=1.0           # 全量
if [[ $QUICK -eq 1 ]]; then
    SUBSAMPLE=0.25
    echo "[1.5/6] QUICK 模式:子采样 25%"
fi

python data_prep/build_binary_pkl.py \
    --src data/ntu60_2d.pkl \
    --dst data/fall_binary_xsub.pkl \
    --split xsub \
    --neg-strategy "$NEG_STRAT" \
    --neg-pos-ratio "$NEG_RATIO" \
    --subsample-ratio "$SUBSAMPLE"

# 划分泄漏检查
python data_prep/split_check.py \
    --pkl data/fall_binary_xsub.pkl

# 骨骼可视化(只画 3 个,人工抽查)
echo "[1.5/6] 骨骼可视化(3 个样本,人工核验头连头脚连脚)"
python data_prep/visualize_skeleton.py \
    --pkl data/fall_binary_xsub.pkl \
    --num 3 \
    --out-dir vis/skeleton_check || \
    echo "[警告] 骨骼可视化失败,可在训练后手动跑确认"

# ============================================================
# 2. 训练 ST-GCN++(快,先出基线)
# ============================================================
if [[ $POSEC3D_ONLY -eq 0 ]]; then
    section "[2/6] 训练 ST-GCN++(基线,~30min @ 4090)"
    bash tools/train.sh "$STGCN_CFG" "$GPUS"
fi

# ============================================================
# 3. 训练 PoseConv3D(主线,慢,效果好)
# ============================================================
if [[ $STGCN_ONLY -eq 0 ]]; then
    section "[3/6] 训练 PoseConv3D(主线,~1.5-2h @ 4090)"
    bash tools/train.sh "$POSEC3D_CFG" "$GPUS"
fi

# ============================================================
# 4. checkpoint 完整性验证
# ============================================================
section "[4/6] checkpoint 验证(防止上一版 ckpt 保存 bug 复发)"
if [[ $STGCN_ONLY -eq 0 ]]; then
    python tools/verify_best_ckpt.py --work-dir "$POSEC3D_DIR"
fi
if [[ $POSEC3D_ONLY -eq 0 ]]; then
    python tools/verify_best_ckpt.py --work-dir "$STGCN_DIR"
fi

# ============================================================
# 5. 测试 + 二分类精细指标
# ============================================================
section "[5/6] 测试 + 二分类指标"

eval_model() {
    local cfg="$1"; local work_dir="$2"; local tag="$3"
    local ckpt
    ckpt=$(find_best_ckpt "$work_dir")
    if [[ -z "$ckpt" ]]; then
        echo "[警告] $tag 找不到 ckpt,跳过评估"; return
    fi
    echo "[eval] $tag  ckpt=$ckpt"

    # 5.1 跑 test.py dump pred pickle
    local pred_pkl="${work_dir}/pred.pkl"
    bash tools/test.sh "$cfg" "$ckpt" "$pred_pkl"

    # 5.2 跑二分类指标
    python tools/eval_binary_metrics.py \
        --pred "$pred_pkl" \
        --config "$cfg" \
        --out-dir "${work_dir}/eval"
}

if [[ $STGCN_ONLY -eq 0 ]]; then
    eval_model "$POSEC3D_CFG" "$POSEC3D_DIR" "PoseConv3D"
fi
if [[ $POSEC3D_ONLY -eq 0 ]]; then
    eval_model "$STGCN_CFG" "$STGCN_DIR" "ST-GCN++"
fi

# ============================================================
# 6. 训练曲线
# ============================================================
section "[6/6] 训练曲线绘制"

mkdir -p figs
if [[ $POSEC3D_ONLY -eq 1 ]]; then
    python tools/plot_curves.py \
        --work-dirs "$POSEC3D_DIR" \
        --labels PoseConv3D \
        --out figs/curves_posec3d.png
elif [[ $STGCN_ONLY -eq 1 ]]; then
    python tools/plot_curves.py \
        --work-dirs "$STGCN_DIR" \
        --labels ST-GCN++ \
        --out figs/curves_stgcnpp.png
else
    python tools/plot_curves.py \
        --work-dirs "$POSEC3D_DIR" "$STGCN_DIR" \
        --labels PoseConv3D ST-GCN++ \
        --out figs/main_compare.png
fi

# ============================================================
# 完成
# ============================================================
section "全部完成 ✓"
cat <<EOF

接下来:
  1. 看训练曲线:      figs/main_compare.png
  2. 看二分类指标:    work_dirs/*/eval/metrics.json + confusion_matrix.png
  3. 录制 demo:       python inference/realtime_demo.py \\
                          --source 你的测试视频.mp4 \\
                          --config $POSEC3D_CFG \\
                          --ckpt $(find_best_ckpt "$POSEC3D_DIR") \\
                          --save-out demo_output.mp4 --no-show
  4. 跨数据集泛化:    见 docs/04_evaluation_visualization.md "URFD 测试" 章节
  5. 论文消融实验:    见 docs/03_model_training.md "消融实验配置" 章节

EOF

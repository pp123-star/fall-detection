"""
tools/eval_binary_metrics.py — 二分类精细指标计算

MMAction2 默认输出的是 top-1 准确率,对二分类不够细致。
本脚本基于 test.py 输出的预测 pickle,计算:
  - Precision / Recall / F1
  - ROC AUC / PR AUC
  - 不同阈值下的指标曲线(找最佳阈值)
  - 混淆矩阵(可视化为 png)
  - 错误样本列表(便于失败案例分析,写论文用)

用法:
    python tools/eval_binary_metrics.py \
        --pred work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18_pred.pkl \
        --config configs/posec3d_fall_binary.py \
        [--threshold 0.5]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def load_predictions(pred_path):
    """加载 mmaction2 test.py --dump 输出的 pickle。

    格式(mmaction v1.x):
      [
        {'pred_score': tensor(C,), 'gt_label': tensor(scalar), 'frame_dir': str, ...},
        ...
      ]
    """
    with open(pred_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, list) or len(results) == 0:
        raise ValueError(f"预测文件格式不对: {pred_path}")

    # 提取分数和标签
    pred_scores = []
    gt_labels = []
    frame_dirs = []

    for r in results:
        if "pred_score" in r:
            score = r["pred_score"]
        elif "pred_scores" in r:
            score = r["pred_scores"]
        else:
            raise KeyError(f"找不到 pred_score 字段,实际字段: {list(r.keys())}")

        # 转 numpy
        import torch
        if isinstance(score, torch.Tensor):
            score = score.cpu().numpy()
        pred_scores.append(score)

        if "gt_label" in r:
            label = r["gt_label"]
        elif "gt_labels" in r:
            label = r["gt_labels"]
        else:
            raise KeyError(f"找不到 gt_label 字段")
        if isinstance(label, torch.Tensor):
            label = int(label.cpu().numpy())
        gt_labels.append(int(label))

        # 样本名(可选,部分版本可能没有)
        for key in ["frame_dir", "video_id", "sample_idx"]:
            if key in r:
                frame_dirs.append(str(r[key]))
                break
        else:
            frame_dirs.append(f"sample_{len(frame_dirs)}")

    pred_scores = np.stack(pred_scores)  # (N, 2)
    gt_labels = np.array(gt_labels)       # (N,)
    return pred_scores, gt_labels, frame_dirs


def print_confusion_matrix(y_true, y_pred):
    """打印 2x2 混淆矩阵到终端。"""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    print()
    print("           预测=非摔倒   预测=摔倒")
    print(f"实际=非摔倒  {tn:>10d}   {fp:>10d}")
    print(f"实际=摔倒    {fn:>10d}   {tp:>10d}")
    return tp, tn, fp, fn


def plot_confusion_matrix(y_true, y_pred, out_path):
    """画混淆矩阵 png(论文用)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[WARN] 缺少 matplotlib/seaborn,跳过混淆矩阵绘图")
        return

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["non-fall", "fall"],
                yticklabels=["non-fall", "fall"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] 混淆矩阵保存到 {out_path}")


def compute_metrics(y_true, y_score, threshold=0.5):
    """计算给定阈值下的指标。"""
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n_pos = tp + fn
    n_neg = tn + fp

    acc       = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0          # 又叫 sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) else 0        # TNR
    fpr       = fp / (fp + tn) if (fp + tn) else 0          # false positive rate
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return dict(
        threshold=threshold,
        accuracy=acc,
        precision=precision,
        recall=recall,
        specificity=specificity,
        fpr=fpr,
        f1=f1,
        tp=tp, tn=tn, fp=fp, fn=fn,
        n_pos=n_pos, n_neg=n_neg,
    )


def find_best_threshold(y_true, y_score):
    """在 [0, 1] 上扫描阈值,找出 F1 最大的点和 Youden's J 最大的点。"""
    thrs = np.linspace(0.05, 0.95, 91)
    best_f1 = (0, 0)
    best_j = (0, 0)

    for t in thrs:
        m = compute_metrics(y_true, y_score, t)
        if m["f1"] > best_f1[1]:
            best_f1 = (t, m["f1"])
        j = m["recall"] + m["specificity"] - 1
        if j > best_j[1]:
            best_j = (t, j)
    return best_f1, best_j


def plot_pr_roc(y_true, y_score, out_dir):
    """画 PR 曲线和 ROC 曲线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (roc_curve, auc,
                                     precision_recall_curve, average_precision_score)
    except ImportError:
        print("[WARN] 缺少 sklearn,跳过 ROC/PR 绘图")
        return None, None

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", color="C0")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=150)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, label=f"PR (AP = {pr_auc:.3f})", color="C1")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr.png", dpi=150)
    plt.close()

    print(f"[INFO] ROC/PR 曲线保存到 {out_dir}/")
    return roc_auc, pr_auc


def main():
    parser = argparse.ArgumentParser(description="二分类细致评估")
    parser.add_argument("--pred", required=True,
                        help="test.py --dump 输出的 pickle")
    parser.add_argument("--config", default=None,
                        help="(可选)配置文件,只用于打印")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="主要阈值(默认 0.5)")
    parser.add_argument("--out-dir", default=None,
                        help="输出目录(默认与 pred 同目录)")
    parser.add_argument("--save-errors", action="store_true",
                        help="保存误判样本列表(失败案例分析用)")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"预测文件: {pred_path}")
    if args.config:
        print(f"配置: {args.config}")
    print("=" * 70)

    pred_scores, gt_labels, frame_dirs = load_predictions(pred_path)
    print(f"样本数: {len(gt_labels)}")
    print(f"正样本数(摔倒): {int(np.sum(gt_labels == 1))}")
    print(f"负样本数(非摔倒): {int(np.sum(gt_labels == 0))}")
    print(f"模型输出分数 shape: {pred_scores.shape}")

    # 二分类:取类别 1(摔倒)的概率
    y_score = pred_scores[:, 1]
    y_true = gt_labels

    # ============ 默认阈值 0.5 ============
    print()
    print("=" * 70)
    print(f"[Threshold = {args.threshold}] 默认阈值评估")
    print("=" * 70)
    m = compute_metrics(y_true, y_score, args.threshold)
    print(f"Accuracy:    {m['accuracy']:.4f}")
    print(f"Precision:   {m['precision']:.4f}")
    print(f"Recall (摔倒检出率):    {m['recall']:.4f}")
    print(f"Specificity (非摔倒识别率): {m['specificity']:.4f}")
    print(f"FPR (误报率): {m['fpr']:.4f}")
    print(f"F1:          {m['f1']:.4f}")
    y_pred_def = (y_score >= args.threshold).astype(int)
    print_confusion_matrix(y_true, y_pred_def)

    # ============ 最佳阈值搜索 ============
    print()
    print("=" * 70)
    print("[Threshold Search] 寻找最佳阈值")
    print("=" * 70)
    best_f1, best_j = find_best_threshold(y_true, y_score)
    print(f"按 F1 最大: threshold = {best_f1[0]:.3f}, F1 = {best_f1[1]:.4f}")
    print(f"按 Youden's J: threshold = {best_j[0]:.3f}, J = {best_j[1]:.4f}")

    m_best = compute_metrics(y_true, y_score, best_f1[0])
    print(f"\n按最佳 F1 阈值的指标:")
    print(f"  Accuracy: {m_best['accuracy']:.4f}, "
          f"P: {m_best['precision']:.4f}, R: {m_best['recall']:.4f}, "
          f"F1: {m_best['f1']:.4f}, FPR: {m_best['fpr']:.4f}")

    # ============ AUC & 曲线 ============
    print()
    print("=" * 70)
    print("[AUC & 曲线]")
    print("=" * 70)
    roc_auc, pr_auc = plot_pr_roc(y_true, y_score, out_dir)
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC (AP): {pr_auc:.4f}")

    # ============ 混淆矩阵 png ============
    plot_confusion_matrix(y_true, y_pred_def, out_dir / "confusion_matrix.png")

    # ============ 错误样本列表 ============
    if args.save_errors:
        errors = []
        for i in range(len(y_true)):
            if y_pred_def[i] != y_true[i]:
                err_type = "FP (误报)" if y_pred_def[i] == 1 else "FN (漏报)"
                errors.append({
                    "frame_dir": frame_dirs[i],
                    "gt_label": int(y_true[i]),
                    "pred_label": int(y_pred_def[i]),
                    "fall_score": float(y_score[i]),
                    "type": err_type,
                })
        if errors:
            err_path = out_dir / "errors.csv"
            with open(err_path, "w") as f:
                f.write("frame_dir,gt,pred,fall_score,type\n")
                for e in errors:
                    f.write(f"{e['frame_dir']},{e['gt_label']},{e['pred_label']},"
                            f"{e['fall_score']:.4f},{e['type']}\n")
            print(f"\n[INFO] 误判样本列表保存到 {err_path} ({len(errors)} 条)")
            # 打印一些代表性错误
            fp_list = [e for e in errors if e["type"].startswith("FP")]
            fn_list = [e for e in errors if e["type"].startswith("FN")]
            print(f"  FP (误报为摔倒): {len(fp_list)}")
            print(f"  FN (漏报真摔倒): {len(fn_list)}")
            for e in (fp_list[:3] + fn_list[:3]):
                print(f"    {e['type']:>15s}: {e['frame_dir']:<25s} score={e['fall_score']:.3f}")

    # ============ 汇总表(论文用) ============
    print()
    print("=" * 70)
    print("[论文汇总表]")
    print("=" * 70)
    print(f"{'Metric':<15s} {'Value':>10s}")
    print("-" * 30)
    print(f"{'Accuracy':<15s} {m['accuracy']:>10.4f}")
    print(f"{'Precision':<15s} {m['precision']:>10.4f}")
    print(f"{'Recall':<15s} {m['recall']:>10.4f}")
    print(f"{'Specificity':<15s} {m['specificity']:>10.4f}")
    print(f"{'F1':<15s} {m['f1']:>10.4f}")
    if roc_auc:
        print(f"{'ROC AUC':<15s} {roc_auc:>10.4f}")
        print(f"{'PR AUC':<15s} {pr_auc:>10.4f}")
    print(f"{'FPR':<15s} {m['fpr']:>10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
tools/plot_curves.py — 训练曲线绘制(论文图用)

从 MMAction2 work_dir 下的 JSON / scalars.json 日志里抽取
train_loss / val_acc / val_loss / lr,画成单图或多图。

支持两种 MMAction2 v1.x 日志格式:
  1) {work_dir}/{时间戳}/vis_data/scalars.json     ← v1.x 默认
  2) {work_dir}/{时间戳}.json                       ← 老版兜底

支持同时输入多个 work_dir,在同一张图上对比(适合论文里 PoseConv3D vs ST-GCN++)。

用法:
    # 单模型
    python tools/plot_curves.py --work-dirs work_dirs/posec3d_fall_binary

    # 多模型对比(论文里要的图)
    python tools/plot_curves.py \
        --work-dirs work_dirs/posec3d_fall_binary work_dirs/stgcnpp_fall_binary \
        --labels PoseConv3D ST-GCN++ \
        --out figs/main_compare.png

    # 单独画某一项指标
    python tools/plot_curves.py --work-dirs work_dirs/posec3d_fall_binary \
        --metric acc --out figs/acc_curve.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境也能跑
import matplotlib.pyplot as plt


def find_scalar_jsons(work_dir):
    """在 work_dir 下找所有 scalars.json(可能跑了多次,有多个时间戳)。

    返回按 mtime 排序的列表,最后一个是最新的。
    """
    work_dir = Path(work_dir)
    candidates = list(work_dir.rglob("vis_data/scalars.json"))
    # 兜底:老格式
    if not candidates:
        candidates = list(work_dir.glob("*.json"))
        # 过滤掉 best_*.json、configs.json 之类的
        candidates = [
            c for c in candidates
            if c.name not in ("best.json",) and not c.name.startswith("config")
        ]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates


def parse_scalars(json_path):
    """解析 scalars.json,每行是一个 JSON dict。

    返回:
      {
        'train': {'epoch': [...], 'iter': [...], 'loss': [...], 'lr': [...]},
        'val':   {'epoch': [...], 'acc/top1': [...], 'loss': [...]},
      }
    """
    train = {"epoch": [], "iter": [], "loss": [], "lr": []}
    val = {"epoch": [], "acc/top1": [], "loss": []}

    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 启发式判断 train / val:val 行通常有 'acc/top1' 字段;
            # 也可以用 'mode' 字段(部分版本有)区分
            mode = rec.get("mode")
            has_acc = any(k.startswith("acc/") for k in rec.keys())

            if mode == "val" or has_acc:
                if "epoch" in rec:
                    val["epoch"].append(rec["epoch"])
                if "acc/top1" in rec:
                    val["acc/top1"].append(rec["acc/top1"])
                if "loss" in rec and mode == "val":
                    val["loss"].append(rec["loss"])
            else:
                if "loss" in rec:
                    train["loss"].append(rec["loss"])
                    if "epoch" in rec:
                        train["epoch"].append(rec["epoch"])
                    if "iter" in rec:
                        train["iter"].append(rec["iter"])
                if "lr" in rec:
                    train["lr"].append(rec["lr"])

    return {"train": train, "val": val}


def load_run(work_dir):
    """加载一个 work_dir 的最新一次训练日志,聚合多个 scalars.json。"""
    jsons = find_scalar_jsons(work_dir)
    if not jsons:
        raise FileNotFoundError(
            f"在 {work_dir} 下找不到 scalars.json 或日志 JSON。\n"
            f"请确认训练已经至少跑过一个 epoch,或 work_dir 路径是否写错。"
        )
    # 只取最近一次的(避免和之前失败的训练日志混在一起)
    latest = jsons[-1]
    print(f"[load_run] {work_dir} -> {latest}")
    return parse_scalars(latest)


# ============================================================
# 绘图
# ============================================================
PAPER_STYLE = {
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def plot_all(runs, labels, out_path):
    """画 4 合 1 图:train_loss, val_loss, val_acc, lr。"""
    plt.rcParams.update(PAPER_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_tl, ax_vl), (ax_va, ax_lr) = axes

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

    for idx, (run, label) in enumerate(zip(runs, labels)):
        c = colors[idx % len(colors)]

        # train loss(按 iter 展开,可能很密,做下采样)
        tl = run["train"]["loss"]
        if tl:
            stride = max(1, len(tl) // 200)
            ax_tl.plot(range(0, len(tl), stride), tl[::stride], color=c, label=label)

        # val loss
        if run["val"]["loss"]:
            ax_vl.plot(run["val"]["epoch"][:len(run["val"]["loss"])],
                       run["val"]["loss"], color=c, marker="o", label=label)

        # val acc
        if run["val"]["acc/top1"]:
            ax_va.plot(run["val"]["epoch"][:len(run["val"]["acc/top1"])],
                       run["val"]["acc/top1"], color=c, marker="o", label=label)

        # lr
        lrs = run["train"]["lr"]
        if lrs:
            stride = max(1, len(lrs) // 200)
            ax_lr.plot(range(0, len(lrs), stride), lrs[::stride], color=c, label=label)

    ax_tl.set_title("Training Loss")
    ax_tl.set_xlabel("Iteration")
    ax_tl.set_ylabel("Loss")
    ax_tl.legend()

    ax_vl.set_title("Validation Loss")
    ax_vl.set_xlabel("Epoch")
    ax_vl.set_ylabel("Loss")
    ax_vl.legend()

    ax_va.set_title("Validation Accuracy (Top-1)")
    ax_va.set_xlabel("Epoch")
    ax_va.set_ylabel("Accuracy")
    ax_va.set_ylim(0.5, 1.01)
    ax_va.legend()

    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.set_xlabel("Iteration")
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot_all] 已保存 -> {out_path}")
    plt.close()


def plot_single(runs, labels, metric, out_path):
    """画单一指标图(论文 4.x 节用)。"""
    plt.rcParams.update(PAPER_STYLE)
    fig, ax = plt.subplots()

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

    for idx, (run, label) in enumerate(zip(runs, labels)):
        c = colors[idx % len(colors)]

        if metric == "acc":
            vals = run["val"]["acc/top1"]
            epochs = run["val"]["epoch"][:len(vals)]
            if vals:
                ax.plot(epochs, vals, color=c, marker="o", label=label)
            ax.set_ylabel("Validation Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylim(0.5, 1.01)
            ax.set_title("Fall Detection — Validation Accuracy")
        elif metric == "loss":
            vals = run["train"]["loss"]
            stride = max(1, len(vals) // 200)
            ax.plot(range(0, len(vals), stride), vals[::stride], color=c, label=label)
            ax.set_ylabel("Training Loss")
            ax.set_xlabel("Iteration")
            ax.set_title("Fall Detection — Training Loss")
        elif metric == "val_loss":
            vals = run["val"]["loss"]
            epochs = run["val"]["epoch"][:len(vals)]
            if vals:
                ax.plot(epochs, vals, color=c, marker="o", label=label)
            ax.set_ylabel("Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_title("Fall Detection — Validation Loss")
        else:
            raise ValueError(f"未知 metric: {metric}")

    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[plot_single] 已保存 -> {out_path}")
    plt.close()


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="训练曲线绘制(论文图用)")
    parser.add_argument("--work-dirs", nargs="+", required=True,
                        help="一个或多个 work_dir,多个会画在同一张图上对比")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="图例标签,顺序对应 --work-dirs;不填则用目录名")
    parser.add_argument("--out", type=str, default="figs/training_curves.png",
                        help="输出图片路径")
    parser.add_argument("--metric", type=str, default="all",
                        choices=["all", "acc", "loss", "val_loss"],
                        help="all=4合1图,其他=单一指标图")
    args = parser.parse_args()

    # 标签默认值
    if args.labels is None:
        args.labels = [Path(w).name for w in args.work_dirs]
    assert len(args.labels) == len(args.work_dirs), \
        f"--labels 数量 ({len(args.labels)}) 与 --work-dirs ({len(args.work_dirs)}) 不一致"

    # 输出目录
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载
    runs = [load_run(w) for w in args.work_dirs]

    # 绘制
    if args.metric == "all":
        plot_all(runs, args.labels, str(out_path))
    else:
        plot_single(runs, args.labels, args.metric, str(out_path))

    # 顺手打印一份"最佳验证准确率"摘要
    print("\n=== 最佳验证准确率摘要 ===")
    for run, label in zip(runs, args.labels):
        accs = run["val"]["acc/top1"]
        if accs:
            best = max(accs)
            best_epoch = run["val"]["epoch"][accs.index(best)]
            print(f"  {label:20s}  best_acc={best:.4f}  @ epoch {best_epoch}")
        else:
            print(f"  {label:20s}  (无 val acc 记录)")


if __name__ == "__main__":
    main()

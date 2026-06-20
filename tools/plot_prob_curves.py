"""
tools/plot_prob_curves.py — 把 prob log 画成概率曲线

诊断核心问题:test4/test7 漏检时,究竟是
  • "模型给了 0.49 没过 0.5"  → 调阈值/策略即可救
  • "模型从头到尾只给 0.10"   → 必须微调

只看 summary.csv 的 max_pfall 一个数字看不清趋势。这个工具:
  1) 把 prob log JSONL 读成时间序列
  2) 每个 track 画一条 raw + smoothed 曲线
  3) 横线标记阈值,标出报警触发点
  4) 输出 PNG 到指定目录

典型用法:
    # 单视频
    python tools/plot_prob_curves.py \
        --prob-log outputs/real_eval/baseline/probs/test7_prob.jsonl \
        --out outputs/real_eval/baseline/curves/test7.png \
        --threshold 0.5 --high-thr 0.8

    # 整批(配合 run_real_video_eval.py 的输出)
    python tools/plot_prob_curves.py \
        --prob-log-dir outputs/real_eval/baseline/probs \
        --out-dir outputs/real_eval/baseline/curves \
        --threshold 0.5 --high-thr 0.8

输出图:
  x 轴 = 帧号(或时间戳)
  y 轴 = P(fall) ∈ [0, 1]
  每个 track 一条 raw 实线 + smoothed 虚线
  红色三角 = 报警触发点
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load_prob_log(path: Path) -> List[dict]:
    """读 JSONL,返回字典列表。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] 跳过损坏行: {e}")
    return out


def group_by_track(records: List[dict]) -> Dict[int, List[dict]]:
    """按 track_id 分组,组内按 frame_idx 排序。"""
    by_tid = defaultdict(list)
    for r in records:
        by_tid[int(r["track_id"])].append(r)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda x: int(x["frame_idx"]))
    return dict(by_tid)


def plot_one_video(records: List[dict],
                   out_png: Path,
                   threshold: float = 0.5,
                   high_thr: Optional[float] = None,
                   title: Optional[str] = None,
                   ground_truth: Optional[int] = None,
                   summary_json: Optional[Path] = None):
    """画一个视频的概率曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        print(f"  [skip] {out_png.name}: 概率日志为空")
        return

    by_tid = group_by_track(records)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10.colors

    # 摘要数字(可选,叠加在图上)
    all_raw = [r["raw_prob"] for r in records]
    max_p = max(all_raw) if all_raw else 0.0
    top5 = sorted(all_raw, reverse=True)[:5]
    mean_top5 = sum(top5) / len(top5) if top5 else 0.0

    for i, (tid, recs) in enumerate(sorted(by_tid.items())):
        c = colors[i % len(colors)]
        frames = [r["frame_idx"] for r in recs]
        raw = [r["raw_prob"] for r in recs]
        smoothed = [r["smoothed_prob"] for r in recs]

        ax.plot(frames, raw, "-", color=c, linewidth=1.6,
                label=f"track {tid} (raw)", alpha=0.85)
        ax.plot(frames, smoothed, "--", color=c, linewidth=1.0,
                label=f"track {tid} (smoothed)", alpha=0.55)

        # 报警点
        alerted = [(r["frame_idx"], r["raw_prob"]) for r in recs if r.get("alerted")]
        if alerted:
            xs, ys = zip(*alerted)
            ax.scatter(xs, ys, marker="v", color="red", s=120,
                       zorder=5, edgecolors="black", linewidths=0.6,
                       label="ALERT" if i == 0 else None)

    # 阈值线
    ax.axhline(threshold, linestyle=":", color="gray", linewidth=1.0,
               label=f"mid threshold = {threshold}")
    if high_thr is not None and high_thr < 1.0:
        ax.axhline(high_thr, linestyle=":", color="darkred", linewidth=1.0,
                   label=f"high threshold = {high_thr}")
    # max 标记线
    ax.axhline(max_p, linestyle="-", color="purple", linewidth=0.6, alpha=0.4)
    ax.text(ax.get_xlim()[1], max_p + 0.01, f"max={max_p:.3f}",
            color="purple", fontsize=8, ha="right", va="bottom")

    ax.set_xlabel("frame index")
    ax.set_ylabel("P(fall)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    # 标题
    parts = []
    if title:
        parts.append(title)
    parts.append(f"max={max_p:.3f}  mean_top5={mean_top5:.3f}")
    if ground_truth is not None:
        parts.append(f"GT={ground_truth}")
    ax.set_title("  |  ".join(parts), fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  → {out_png}  (max={max_p:.3f}, tracks={len(by_tid)})")


def _try_read_summary(prob_log: Path) -> dict:
    """根据 prob log 路径猜对应 summary.json(便于叠加 GT)。"""
    stem = prob_log.stem.replace("_prob", "")
    parent = prob_log.parent.parent  # 通常 .../probs/
    candidates = [
        parent / "summaries" / f"{stem}_summary.json",
        prob_log.with_suffix(".summary.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.load(open(p, "r", encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    return {}


def main():
    p = argparse.ArgumentParser(description="画 prob log 概率曲线")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prob-log", help="单个 prob log JSONL")
    src.add_argument("--prob-log-dir", help="批量:prob log 目录")

    p.add_argument("--out", default=None,
                   help="单视频模式:输出 PNG 路径")
    p.add_argument("--out-dir", default=None,
                   help="批量模式:输出 PNG 目录")

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--high-thr", type=float, default=0.8)
    args = p.parse_args()

    if args.prob_log:
        log_path = Path(args.prob_log)
        recs = load_prob_log(log_path)
        s = _try_read_summary(log_path)
        out = Path(args.out) if args.out else log_path.with_suffix(".png")
        plot_one_video(
            records=recs, out_png=out,
            threshold=args.threshold, high_thr=args.high_thr,
            title=log_path.stem.replace("_prob", ""),
            ground_truth=s.get("ground_truth"),
        )
    else:
        in_dir = Path(args.prob_log_dir)
        out_dir = Path(args.out_dir) if args.out_dir else (in_dir.parent / "curves")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("*.jsonl"))
        print(f"[scan] {len(files)} 个 prob log 在 {in_dir}")
        for fp in files:
            recs = load_prob_log(fp)
            s = _try_read_summary(fp)
            stem = fp.stem.replace("_prob", "")
            out = out_dir / f"{stem}.png"
            plot_one_video(
                records=recs, out_png=out,
                threshold=args.threshold, high_thr=args.high_thr,
                title=stem,
                ground_truth=s.get("ground_truth"),
            )
        print(f"[done] → {out_dir}")


if __name__ == "__main__":
    main()

"""
tools/replay_fall_trend.py — 用 prob log 离线回放 FallTrendDetector

不需要重跑视频和模型,直接喂历史数据看会不会触发新策略。
用于:
  1. 验证 elder_fall_7 在新策略下能被救回(模拟集成场景)
  2. 反向测试 ADL/正常行走段不会被新策略误报
  3. 阈值快速扫描(改参数不用重跑昂贵的 YOLO+模型推理)

示例:
    python tools/replay_fall_trend.py \
        --prob-log outputs/real_eval/.../probs/elder_fall_7_prob.jsonl \
        --simulate-disappearance-at-end
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.realtime_core import FallTrendDetector


def load_prob_log(path):
    """读 JSONL prob log。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    out.sort(key=lambda r: r["frame_idx"])
    return out


def replay(prob_log_path, args):
    """模拟主循环:逐次推理喂入历史,在每次推理后检查四个策略。"""
    records = load_prob_log(prob_log_path)
    print(f"[load] {prob_log_path}: {len(records)} 次推理")
    if not records:
        return None

    det = FallTrendDetector(
        slope_window=args.slope_window,
        slope_prob_thr=args.slope_prob_thr,
        slope_heur_thr=args.slope_heur_thr,
        slope_min_current=args.slope_min_current,
        geom_window_frames=args.geom_window,
        bbox_h_drop_ratio=args.bbox_h_drop,
        aspect_rise=args.aspect_rise,
        disappear_lookback=args.disappear_lookback,
        disappear_raw_min=args.disappear_raw_min,
        disappear_heur_min=args.disappear_heur_min,
        autopsy_max_raw_thr=args.autopsy_raw_thr,
        autopsy_max_heur_thr=args.autopsy_heur_thr,
    )

    # 模拟主循环
    raw_history = deque(maxlen=30)
    heur_history = deque(maxlen=30)
    bbox_history = deque(maxlen=60)

    triggered = []
    first_alert_frame = None

    last_frame = records[-1]["frame_idx"]

    for i, r in enumerate(records):
        raw_history.append(r["raw_prob"])
        heur_history.append(r["heuristic_score"])
        bbox = [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]]
        bbox_history.append(bbox)

        # 在每次推理后检查策略 B + C
        res_b = det.check_slope(list(raw_history), list(heur_history))
        res_c = det.check_geometric(list(bbox_history), track_age=100)

        if first_alert_frame is None:
            if res_b.alert:
                first_alert_frame = r["frame_idx"]
                triggered.append((r["frame_idx"], "slope", res_b.reason, res_b.score))
            if res_c.alert:
                if first_alert_frame is None:
                    first_alert_frame = r["frame_idx"]
                triggered.append((r["frame_idx"], "geom", res_c.reason, res_c.score))

    # 视频结束时模拟 track 已经丢失,跑策略 A + D
    print("\n=== 推理阶段触发 ===")
    if triggered:
        for f, strat, reason, score in triggered[:5]:
            print(f"  [frame {f}] {strat}: {reason} (score={score:.3f})")
        print(f"  最早触发: frame {first_alert_frame}")
    else:
        print("  无策略 B/C 触发")

    # 模拟 track 丢失若干帧
    print("\n=== 模拟 track 消失后触发 ===")
    if args.simulate_disappearance_at_end:
        for sim_age in [8, 15, 30]:
            res_a = det.check_disappearance(
                list(raw_history), list(heur_history),
                track_age=sim_age, min_lost_gap=8,
            )
            tag = "✓" if res_a.alert else "✗"
            print(f"  [track_age={sim_age}] disappear: {tag} {res_a.strategy if res_a.alert else ''} {res_a.reason} (score={res_a.score:.3f})")

    print("\n=== 模拟 track 永久清理 (autopsy) ===")
    res_d = det.check_autopsy(list(raw_history), list(heur_history))
    tag = "✓" if res_d.alert else "✗"
    print(f"  autopsy: {tag} {res_d.reason} (score={res_d.score:.3f})")

    overall_alert = (first_alert_frame is not None) or res_d.alert
    return {
        "prob_log": str(prob_log_path),
        "n_inferences": len(records),
        "first_alert_frame": first_alert_frame,
        "in_infer_triggers": triggered,
        "autopsy_triggered": res_d.alert,
        "overall_alert": overall_alert,
    }


def main():
    p = argparse.ArgumentParser(description="离线回放 prob log,模拟 FallTrendDetector")
    p.add_argument("--prob-log", required=True, help="prob log JSONL 文件路径")
    p.add_argument("--simulate-disappearance-at-end", action="store_true", default=True,
                   help="末尾模拟 track 丢失,跑策略 A")

    # 阈值参数 (与 FallTrendDetector 一致)
    p.add_argument("--slope-window", type=int, default=5)
    p.add_argument("--slope-prob-thr", type=float, default=0.05)
    p.add_argument("--slope-heur-thr", type=float, default=0.08)
    p.add_argument("--slope-min-current", type=float, default=0.28)
    p.add_argument("--geom-window", type=int, default=15)
    p.add_argument("--bbox-h-drop", type=float, default=0.35)
    p.add_argument("--aspect-rise", type=float, default=0.10)
    p.add_argument("--disappear-lookback", type=int, default=4)
    p.add_argument("--disappear-raw-min", type=float, default=0.28)
    p.add_argument("--disappear-heur-min", type=float, default=0.32)
    p.add_argument("--autopsy-raw-thr", type=float, default=0.30)
    p.add_argument("--autopsy-heur-thr", type=float, default=0.35)
    args = p.parse_args()

    replay(args.prob_log, args)


if __name__ == "__main__":
    main()

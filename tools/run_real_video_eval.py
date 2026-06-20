"""
tools/run_real_video_eval.py — 一键真实视频批量评估

把一个目录(默认 data/real_test)下的所有 mp4 用 multitarget_realtime_demo
跑一遍,统一收集:
  • overlay 可视化 mp4
  • per-video 事件日志 JSONL
  • per-video 概率日志 JSONL(每次推理 raw/smoothed 概率,核心诊断数据)
  • per-video summary JSON
  • 整体 summary.csv(汇总诊断标签 + 指标)
  • failure_cases.csv(漏检 / 误报 / 模型不识别)
  • snapshots/  报警瞬间帧图

对应"明天的实验"模板:
  • 同一组视频,跑多个参数组合,每个组合一个独立 out 目录
  • 直接对比 summary.csv

示例:
    # 基线(原参数):一键跑 data/real_test 所有视频
    python tools/run_real_video_eval.py \
        --video-dir data/real_test \
        --config configs/posec3d_fall_binary.py \
        --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
        --out-dir outputs/real_eval/baseline

    # 时间窗口 + 多策略 + track 合并(推荐)
    python tools/run_real_video_eval.py \
        --video-dir data/real_test --labels-csv data/real_test/labels.csv \
        --config configs/posec3d_fall_binary.py \
        --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
        --out-dir outputs/real_eval/tw16_track_merge \
        --time-window-sec 1.6 \
        --track-merge --high-thr 0.7 --topk-mean-thr 0.5 --threshold 0.4

labels.csv 格式:
    video,label
    test4.mp4,1
    test5.mp4,1
    test6.mp4,1
    test7.mp4,1
    adl_walking.mp4,0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 读 labels.csv
# ============================================================
def load_labels(label_csv: Optional[str]) -> Dict[str, int]:
    """返回 {basename_or_stem: label(0/1)}。读不到返回空 dict。"""
    if not label_csv:
        return {}
    out = {}
    with open(label_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("video") or row.get("filename") or row.get("name")
            lab = row.get("label") or row.get("y")
            if v is not None and lab is not None:
                out[v] = int(lab)
                # 同时允许 stem 匹配
                out[Path(v).stem] = int(lab)
    return out


def find_videos(video_dir: Path, exts=(".mp4", ".mov", ".avi", ".mkv")):
    out = []
    for p in sorted(video_dir.rglob("*")):
        if p.suffix.lower() in exts:
            out.append(p)
    return out


# ============================================================
# 调起 multitarget_realtime_demo(子进程)
# ============================================================
def run_one_video(video_path: Path, out_dir: Path, args, label: Optional[int]) -> dict:
    """对单个视频调一次 multitarget_realtime_demo。

    返回:{"video", "ok", "stderr_tail", "summary_path", "summary"}
    """
    stem = video_path.stem
    overlay = out_dir / "overlays" / f"{stem}_overlay.mp4"
    event_log = out_dir / "events" / f"{stem}_events.jsonl"
    prob_log = out_dir / "probs" / f"{stem}_prob.jsonl"
    summary_path = out_dir / "summaries" / f"{stem}_summary.json"
    snapshot_dir = out_dir / "snapshots" / stem

    cmd = [
        sys.executable, str(REPO_ROOT / "inference" / "multitarget_realtime_demo.py"),
        "--source", str(video_path),
        "--config", args.config,
        "--ckpt", args.ckpt,
        "--pose-weights", args.pose_weights,
        "--device", args.device,
        "--clip-len", str(args.clip_len),
        "--max-persons", str(args.max_persons),
        "--infer-every", str(args.infer_every),
        "--track-timeout", str(args.track_timeout),
        "--threshold", str(args.threshold),
        "--alert-k", str(args.alert_k),
        "--alert-hold", str(args.alert_hold),
        "--ema", str(args.ema),
        "--kpt-thr", str(args.kpt_thr),
        "--conf", str(args.conf),
        "--imgsz", str(args.imgsz),
        "--save-out", str(overlay),
        "--event-log", str(event_log),
        "--prob-log", str(prob_log),
        "--summary-json", str(summary_path),
        "--snapshot-dir", str(snapshot_dir),
        "--no-show",
    ]
    # 真实视频建议默认。target_fps 不覆盖源 fps;只用于在 demo 内推导训练等效窗口。
    if args.target_fps > 0:
        cmd += ["--target-fps", str(args.target_fps)]
    if args.time_window_sec > 0:
        cmd += ["--time-window-sec", str(args.time_window_sec)]
    if args.track_merge:
        cmd += ["--track-merge",
                "--track-merge-iou-thr", str(args.track_merge_iou_thr),
                "--track-merge-dist-thr", str(args.track_merge_dist_thr),
                "--track-merge-gap", str(args.track_merge_gap)]
    if args.high_thr < 1.0:
        cmd += ["--high-thr", str(args.high_thr)]
    if args.topk_mean_thr < 1.0:
        cmd += ["--topk-mean-thr", str(args.topk_mean_thr),
                "--topk-window", str(args.topk_window),
                "--topk-k", str(args.topk_k)]
    if args.pose_heuristic_alert:
        cmd += ["--pose-heuristic-alert",
                "--pose-heuristic-thr", str(args.pose_heuristic_thr),
                "--pose-heuristic-min-frames", str(args.pose_heuristic_min_frames)]
    if args.lost_track_alert:
        cmd += ["--lost-track-alert",
                "--lost-track-min-gap", str(args.lost_track_min_gap),
                "--lost-track-heuristic-thr", str(args.lost_track_heuristic_thr),
                "--lost-track-model-thr", str(args.lost_track_model_thr)]
    if label is not None:
        cmd += ["--ground-truth", str(int(label))]

    print(f"\n>>> [{video_path.name}] (gt={label})")
    env = os.environ.copy()
    local_mmaction = REPO_ROOT / "mmaction2_src"
    if local_mmaction.exists():
        old_pythonpath = env.get("PYTHONPATH", "")
        parts = [str(local_mmaction)]
        if old_pythonpath:
            parts.append(old_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    try:
        cp = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                            text=True, timeout=args.timeout_sec, env=env)
        ok = (cp.returncode == 0)
        if not ok:
            print(f"  [FAIL] returncode={cp.returncode}")
            print(f"  stderr tail:\n{cp.stderr[-2000:]}")
    except subprocess.TimeoutExpired:
        ok = False
        cp = None
        print(f"  [FAIL] timeout after {args.timeout_sec}s")

    # 读 summary(可能没产出)
    summary_dict = None
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_dict = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 读 summary 失败: {e}")

    return {
        "video": str(video_path),
        "video_name": video_path.name,
        "gt_label": label,
        "ok": ok,
        "stderr_tail": (cp.stderr[-500:] if cp and cp.stderr else ""),
        "overlay": str(overlay) if overlay.exists() else "",
        "event_log": str(event_log) if event_log.exists() else "",
        "prob_log": str(prob_log) if prob_log.exists() else "",
        "summary_path": str(summary_path) if summary_path.exists() else "",
        "summary": summary_dict,
    }


# ============================================================
# 汇总输出
# ============================================================
def write_summary_csv(per_video: List[dict], out_csv: Path):
    """把每个视频的核心字段汇总到 CSV。"""
    fields = [
        "video_name", "gt_label", "ok",
        "diagnosis", "num_alerts",
        "max_pfall", "mean_top5_pfall", "mean_pfall",
        "max_pose_heuristic", "mean_top5_pose_heuristic",
        "num_unique_tracks", "num_id_switches_handled", "suspected_id_switch",
        "total_inferences", "total_frames",
        "overlay", "prob_log",
    ]
    rows = []
    for r in per_video:
        row = {k: "" for k in fields}
        row["video_name"] = r["video_name"]
        row["gt_label"] = r["gt_label"] if r["gt_label"] is not None else ""
        row["ok"] = r["ok"]
        row["overlay"] = r["overlay"]
        row["prob_log"] = r["prob_log"]
        s = r["summary"]
        if s:
            for k in ["diagnosis", "num_alerts", "max_pfall", "mean_top5_pfall",
                      "mean_pfall", "max_pose_heuristic", "mean_top5_pose_heuristic",
                      "num_unique_tracks", "num_id_switches_handled",
                      "suspected_id_switch", "total_inferences", "total_frames"]:
                if k in s:
                    row[k] = s[k]
        rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[write_summary_csv] → {out_csv}  ({len(rows)} 行)")


def write_failure_csv(per_video: List[dict], out_csv: Path):
    """把诊断为问题的视频单独列出,带可执行的建议。"""
    bad_diag = {"model_unaware", "partial_signal",
                "just_below_threshold", "false_alarm", "no_inference"}
    rows = []
    for r in per_video:
        s = r["summary"] or {}
        diag = s.get("diagnosis", "")
        gt = r["gt_label"]
        if not r["ok"]:
            rows.append({"video_name": r["video_name"], "gt_label": gt,
                         "diagnosis": "error",
                         "max_pfall": "",
                         "recommendation": "检查 stderr_tail / 视频是否可读",
                         "stderr_tail": r["stderr_tail"]})
            continue
        if diag in bad_diag or (gt == 1 and s.get("num_alerts", 0) == 0):
            recom = {
                "model_unaware": "模型从头到尾给低分→必须微调(加入这种困难正样本)",
                "partial_signal": "模型给了 0.3-0.5,微调或放宽阈值/启 top-k mean",
                "just_below_threshold": "调阈值即可:降 threshold,或启 high-thr",
                "false_alarm": "FP:调高 threshold 或加困难负样本(类似动作)",
                "no_inference": "buffer 始终没达到 clip_len,可能视频太短或姿态全失败",
                "error": "执行失败,看 stderr",
            }.get(diag, "看 prob log 详细诊断")
            rows.append({
                "video_name": r["video_name"], "gt_label": gt,
                "diagnosis": diag,
                "max_pfall": s.get("max_pfall", ""),
                "recommendation": recom,
                "stderr_tail": "",
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_name", "gt_label", "diagnosis",
                                          "max_pfall", "recommendation", "stderr_tail"])
        w.writeheader()
        w.writerows(rows)
    print(f"[write_failure_csv] → {out_csv}  ({len(rows)} 个问题视频)")


def compute_metrics(per_video: List[dict]) -> dict:
    """如有 GT,算 TP/FP/TN/FN/P/R/F1。"""
    valid = [r for r in per_video if r["ok"] and r["gt_label"] is not None and r["summary"]]
    if not valid:
        return {}
    tp = fp = tn = fn = 0
    for r in valid:
        gt = int(r["gt_label"])
        pred = 1 if r["summary"].get("num_alerts", 0) > 0 else 0
        if gt == 1 and pred == 1: tp += 1
        elif gt == 0 and pred == 1: fp += 1
        elif gt == 0 and pred == 0: tn += 1
        else: fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {"num_with_gt": len(valid),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)}


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(description="一键真实视频批量评估")
    p.add_argument("--video-dir", required=True, help="视频文件夹")
    p.add_argument("--labels-csv", default=None,
                   help="可选 labels.csv (video,label)")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pose-weights", default="yolo26x-pose.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True,
                   help="输出根目录,如 outputs/real_eval/exp01")
    p.add_argument("--timeout-sec", type=int, default=600,
                   help="单个视频处理超时(秒)")

    # 透传给 multitarget_realtime_demo 的参数(都给合理默认)
    p.add_argument("--clip-len", type=int, default=48)
    p.add_argument("--max-persons", type=int, default=5)
    p.add_argument("--infer-every", type=int, default=6)
    p.add_argument("--track-timeout", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--alert-k", type=int, default=2)
    p.add_argument("--alert-hold", type=float, default=1.5)
    p.add_argument("--ema", type=float, default=0.5)
    p.add_argument("--kpt-thr", type=float, default=0.3)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)

    # 真实视频参数
    p.add_argument("--target-fps", type=float, default=0.0,
                   help="训练等效目标 fps。若未显式给 --time-window-sec,用 clip_len/target_fps 推导窗口")
    p.add_argument("--time-window-sec", type=float, default=0.0,
                   help=">0 启用 time-window buffer,推荐 1.6 或 2.0")
    p.add_argument("--track-merge", action="store_true")
    p.add_argument("--track-merge-iou-thr", type=float, default=0.3)
    p.add_argument("--track-merge-dist-thr", type=float, default=0.15)
    p.add_argument("--track-merge-gap", type=int, default=15)
    p.add_argument("--high-thr", type=float, default=1.01)
    p.add_argument("--topk-mean-thr", type=float, default=1.01)
    p.add_argument("--topk-window", type=int, default=5)
    p.add_argument("--topk-k", type=int, default=3)
    p.add_argument("--pose-heuristic-alert", action="store_true",
                   help="启用骨架几何兜底报警,用于快摔/翻倒但模型低分的真实视频")
    p.add_argument("--pose-heuristic-thr", type=float, default=0.62)
    p.add_argument("--pose-heuristic-min-frames", type=int, default=12)
    p.add_argument("--lost-track-alert", action="store_true",
                   help="启用低姿态/疑似跌倒后 track 消失的逻辑兜底报警")
    p.add_argument("--lost-track-min-gap", type=int, default=8)
    p.add_argument("--lost-track-heuristic-thr", type=float, default=0.45)
    p.add_argument("--lost-track-model-thr", type=float, default=0.35)

    args = p.parse_args()

    video_dir = Path(args.video_dir)
    assert video_dir.exists(), f"不存在: {video_dir}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(video_dir)
    print(f"[scan] {video_dir} 下发现 {len(videos)} 个视频")
    if not videos:
        print("没有视频可处理"); return

    labels_map = load_labels(args.labels_csv)
    if labels_map:
        print(f"[labels] 读到 {len(labels_map)//2} 条标签(允许 basename / stem 双匹配)")

    # 写入运行配置
    cfg_dump = {k: getattr(args, k) for k in vars(args)}
    cfg_dump["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2, ensure_ascii=False)

    # 跑所有视频
    per_video = []
    for v in videos:
        lab = labels_map.get(v.name, labels_map.get(v.stem))
        r = run_one_video(v, out_dir, args, lab)
        per_video.append(r)
        # 增量写一份,中途崩了不会一无所有
        with open(out_dir / "results_partial.json", "w", encoding="utf-8") as f:
            json.dump(per_video, f, indent=2, ensure_ascii=False, default=str)

    # 汇总
    write_summary_csv(per_video, out_dir / "summary.csv")
    write_failure_csv(per_video, out_dir / "failure_cases.csv")

    metrics = compute_metrics(per_video)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if metrics:
        print("\n=== 整体指标 ===")
        for k, v in metrics.items():
            print(f"  {k:>12s}: {v}")

    # 最终结果合并
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": cfg_dump,
            "metrics": metrics,
            "per_video": per_video,
        }, f, indent=2, ensure_ascii=False, default=str)
    (out_dir / "results_partial.json").unlink(missing_ok=True)

    print(f"\n[done] 全部输出在: {out_dir}")
    print("  • summary.csv         各视频核心指标")
    print("  • failure_cases.csv   问题视频 + 改进建议")
    print("  • metrics.json        整体 P/R/F1")
    print("  • overlays/           可视化 mp4")
    print("  • probs/              每次推理概率(诊断核心)")
    print("  • events/             报警事件")
    print("  • summaries/          每个视频独立 summary")
    print("  • run_config.json     本次跑的参数")


if __name__ == "__main__":
    main()

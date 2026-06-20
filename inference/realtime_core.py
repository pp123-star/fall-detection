"""
inference/realtime_core.py — 真实视频/实时推理的核心组件

本模块抽出 multitarget_realtime_demo 与 batch_predict 共享的"会因真实视频而变难"的逻辑,
便于两边复用且单元可测。包含 5 个核心组件:

  1. TimeAwareBuffer       — 时间感知的滚动缓冲区,解决"60fps + clip_len=48 只覆盖 0.8s"
  2. TrackMerger           — 处理 ByteTrack 快速动作下的 ID 切换(test7.mp4 那种)
  3. AlertPolicy           — 多策略报警(high 单次 / mid 连续 / top-k 平均)替代单一阈值
  4. ProbabilityLogger     — 每次推理都记 raw/smoothed 概率,未报警视频也能事后诊断
  5. VideoSummaryBuilder   — 视频结束时聚合 max/top-k/mean/疑似 ID switch 等诊断信息

设计原则:
  - 纯 Python + numpy,不依赖 cv2/torch/mmaction;便于单元测试
  - 与现有 multitarget_realtime_demo 的 TrackState 解耦,但兼容它的字段语义
  - 所有新能力默认 off:旧 demo 不传新参数时行为完全不变
"""
from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TimeAwareBuffer:
    """按目标真实时间窗口缓存原始帧,推理时均匀采样为模型 clip_len。"""

    clip_len: int = 48
    source_fps: float = 30.0
    time_window_sec: float = 0.0
    window_frames: int = field(init=False)
    kpts: deque = field(init=False, repr=False)
    scores: deque = field(init=False, repr=False)

    def __post_init__(self):
        if self.time_window_sec > 0:
            target = int(round(self.source_fps * self.time_window_sec))
        else:
            target = self.clip_len
        self.window_frames = max(self.clip_len, target)
        self.kpts = deque(maxlen=self.window_frames)
        self.scores = deque(maxlen=self.window_frames)

    def push(self, kpt: np.ndarray, score: np.ndarray):
        self.kpts.append(kpt.astype(np.float32))
        self.scores.append(score.astype(np.float32))

    @property
    def buffer_len(self) -> int:
        return len(self.kpts)

    @property
    def is_ready(self) -> bool:
        return len(self.kpts) >= self.clip_len

    @property
    def is_full(self) -> bool:
        return len(self.kpts) >= self.window_frames

    def sample_clip(self) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        n = len(self.kpts)
        if n < self.clip_len:
            return None
        if n == self.clip_len:
            return list(self.kpts), list(self.scores)
        idx = np.linspace(0, n - 1, self.clip_len)
        idx = np.round(idx).astype(int)
        idx = np.clip(idx, 0, n - 1)
        return [self.kpts[i] for i in idx], [self.scores[i] for i in idx]

    def inherit_from(self, other: "TimeAwareBuffer", max_keep: Optional[int] = None):
        if other is None:
            return
        keep = max_keep if max_keep is not None else self.window_frames
        for k, s in zip(list(other.kpts)[-keep:], list(other.scores)[-keep:]):
            self.push(k, s)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size != 4 or b.size != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-6 else 0.0


def bbox_center_dist_norm(a: np.ndarray, b: np.ndarray, img_diag: float) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ca = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    d = float(np.linalg.norm(ca - cb))
    return d / img_diag if img_diag > 1e-6 else d


@dataclass
class TombstoneTrack:
    track_id: int
    display_id: int
    last_frame: int
    last_bbox: np.ndarray
    buffer: TimeAwareBuffer
    last_smoothed_prob: float = 0.0
    last_raw_prob: float = 0.0
    over_thr_streak: int = 0


class TrackMerger:
    """把刚消失且空间接近的新旧 track 合并,继承历史 buffer 和概率状态。"""

    def __init__(
        self,
        iou_thr: float = 0.3,
        center_dist_norm_thr: float = 0.15,
        max_gap_frames: int = 15,
        enabled: bool = True,
    ):
        self.iou_thr = iou_thr
        self.center_dist_norm_thr = center_dist_norm_thr
        self.max_gap_frames = max_gap_frames
        self.enabled = enabled
        self.tombstones: Dict[int, TombstoneTrack] = {}
        self.merge_log: List[dict] = []

    def register_death(
        self,
        track_id: int,
        display_id: Optional[int],
        last_frame: int,
        last_bbox: np.ndarray,
        buffer: TimeAwareBuffer,
        last_smoothed_prob: float = 0.0,
        last_raw_prob: float = 0.0,
        over_thr_streak: int = 0,
    ):
        if not self.enabled:
            return
        self.tombstones[track_id] = TombstoneTrack(
            track_id=track_id,
            display_id=int(display_id if display_id is not None else track_id),
            last_frame=last_frame,
            last_bbox=np.asarray(last_bbox, dtype=np.float32).copy(),
            buffer=buffer,
            last_smoothed_prob=last_smoothed_prob,
            last_raw_prob=last_raw_prob,
            over_thr_streak=over_thr_streak,
        )

    def try_match(
        self,
        new_track_id: int,
        new_bbox: np.ndarray,
        current_frame: int,
        img_diag: float,
    ) -> Optional[TombstoneTrack]:
        if not self.enabled or not self.tombstones:
            return None

        best = None
        best_score = 0.0
        best_reason = None

        for old_tid, tomb in list(self.tombstones.items()):
            gap = current_frame - tomb.last_frame
            if gap > self.max_gap_frames:
                del self.tombstones[old_tid]
                continue

            iou = bbox_iou(tomb.last_bbox, new_bbox)
            dist = bbox_center_dist_norm(tomb.last_bbox, new_bbox, img_diag)
            score = 0.0
            reason = None
            if iou >= self.iou_thr:
                score = iou
                reason = f"iou={iou:.2f}"
            elif dist <= self.center_dist_norm_thr:
                score = max(0.0, 1.0 - dist / max(self.center_dist_norm_thr, 1e-6))
                reason = f"dist={dist:.3f}"

            if score > best_score:
                best_score = score
                best = tomb
                best_reason = reason

        if best is None:
            return None

        del self.tombstones[best.track_id]
        self.merge_log.append(
            {
                "frame": current_frame,
                "new_track_id": int(new_track_id),
                "inherited_from": int(best.track_id),
                "display_id": int(best.display_id),
                "reason": best_reason,
            }
        )
        return best

    def prune(self, current_frame: int):
        if not self.enabled:
            return
        stale = [
            tid
            for tid, t in self.tombstones.items()
            if current_frame - t.last_frame > self.max_gap_frames
        ]
        for tid in stale:
            del self.tombstones[tid]

    @property
    def merge_count(self) -> int:
        return len(self.merge_log)


@dataclass
class AlertDecision:
    alert: bool
    reason: str
    triggering_prob: float


class AlertPolicy:
    """组合单次高分、连续中分和最近窗口 top-k 均值三类报警策略。"""

    def __init__(
        self,
        high_thr: float = 1.01,
        mid_thr: float = 0.5,
        consecutive_k: int = 2,
        topk_window: int = 5,
        topk_k: int = 3,
        topk_mean_thr: float = 1.01,
    ):
        self.high_thr = high_thr
        self.mid_thr = mid_thr
        self.consecutive_k = max(1, consecutive_k)
        self.topk_window = max(1, topk_window)
        self.topk_k = max(1, topk_k)
        self.topk_mean_thr = topk_mean_thr

    def evaluate(
        self,
        raw_prob: float,
        smoothed_prob: float,
        over_mid_streak: int,
        recent_raw_probs: List[float],
    ) -> AlertDecision:
        if raw_prob >= self.high_thr:
            return AlertDecision(True, "high_single", raw_prob)

        if smoothed_prob >= self.mid_thr and over_mid_streak >= self.consecutive_k:
            return AlertDecision(True, "consec_mid", smoothed_prob)

        if len(recent_raw_probs) >= self.topk_k:
            window = recent_raw_probs[-self.topk_window:]
            topk = sorted(window, reverse=True)[: self.topk_k]
            mean = float(np.mean(topk)) if topk else 0.0
            if mean >= self.topk_mean_thr:
                return AlertDecision(True, "topk_mean", mean)

        return AlertDecision(False, "", smoothed_prob)


class ProbabilityLogger:
    """记录每一次动作分类概率,让未报警视频也能做事后诊断。"""

    def __init__(self, path: Optional[str], fmt: str = "jsonl", source: str = ""):
        self.path = Path(path) if path else None
        self.fmt = fmt
        self.source = source
        self._fh = None
        self._csv_writer = None
        self._fields = [
            "frame_idx",
            "timestamp",
            "source",
            "track_id",
            "raw_prob",
            "smoothed_prob",
            "buffer_len",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "alerted",
            "alert_reason",
        ]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", encoding="utf-8", newline="")
            if self.fmt == "csv":
                self._csv_writer = csv.DictWriter(self._fh, fieldnames=self._fields)
                self._csv_writer.writeheader()

    def log(
        self,
        frame_idx: int,
        track_id: int,
        raw_prob: float,
        smoothed_prob: float,
        buffer_len: int,
        bbox: np.ndarray,
        alerted: bool = False,
        alert_reason: str = "",
    ):
        if self._fh is None:
            return
        bbox = np.asarray(bbox, dtype=np.float32) if bbox is not None else np.zeros(4)
        rec = {
            "frame_idx": int(frame_idx),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "source": self.source,
            "track_id": int(track_id),
            "raw_prob": round(float(raw_prob), 6),
            "smoothed_prob": round(float(smoothed_prob), 6),
            "buffer_len": int(buffer_len),
            "bbox_x1": float(bbox[0]),
            "bbox_y1": float(bbox[1]),
            "bbox_x2": float(bbox[2]),
            "bbox_y2": float(bbox[3]),
            "alerted": bool(alerted),
            "alert_reason": str(alert_reason or ""),
        }
        if self.fmt == "csv":
            self._csv_writer.writerow(rec)
        else:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


class VideoSummaryBuilder:
    """聚合单个视频的推理概率、报警事件和 ID 合并信息。"""

    def __init__(self, source: str, ground_truth: Optional[int] = None):
        self.source = source
        self.ground_truth = ground_truth
        self.raw_probs: List[float] = []
        self.per_track_probs: Dict[int, List[float]] = {}
        self.alerts: List[dict] = []
        self.num_id_switches_handled = 0
        self.total_frames = 0
        self.total_inferences = 0
        self.error: Optional[str] = None
        self.start_time = time.time()

    def record_inference(self, track_id: int, raw_prob: float):
        self.raw_probs.append(float(raw_prob))
        self.per_track_probs.setdefault(int(track_id), []).append(float(raw_prob))
        self.total_inferences += 1

    def record_alert(self, frame_idx: int, track_id: int, prob: float, reason: str = ""):
        self.alerts.append(
            {
                "frame_idx": int(frame_idx),
                "track_id": int(track_id),
                "prob": float(prob),
                "reason": str(reason),
            }
        )

    def set_frames(self, n: int):
        self.total_frames = int(n)

    def set_merge_count(self, n: int):
        self.num_id_switches_handled = int(n)

    def set_error(self, msg: str):
        self.error = str(msg)

    def diagnose(self, mid_thr: float = 0.5, max_low_zone: float = 0.3) -> str:
        if self.error:
            return "error"
        if self.total_inferences == 0:
            return "no_inference"
        max_p = max(self.raw_probs) if self.raw_probs else 0.0
        has_alert = len(self.alerts) > 0
        if self.ground_truth == 0:
            return "false_alarm" if has_alert else "true_negative"
        if has_alert:
            return "detected"
        if max_p >= mid_thr:
            return "just_below_threshold"
        if max_p >= max_low_zone:
            return "partial_signal"
        return "model_unaware"

    def build(self, mid_thr: float = 0.5, max_low_zone: float = 0.3, topk: int = 5) -> dict:
        if self.error:
            return {
                "source": self.source,
                "ground_truth": self.ground_truth,
                "error": self.error,
                "diagnosis": "error",
                "elapsed_s": round(time.time() - self.start_time, 2),
            }

        probs = np.asarray(self.raw_probs) if self.raw_probs else np.zeros(0)
        topk_vals = sorted(self.raw_probs, reverse=True)[:topk]
        per_track_max = {
            int(tid): round(max(plist), 4)
            for tid, plist in self.per_track_probs.items()
        }

        return {
            "source": self.source,
            "ground_truth": self.ground_truth,
            "total_frames": self.total_frames,
            "total_inferences": self.total_inferences,
            "num_unique_tracks": len(self.per_track_probs),
            "num_id_switches_handled": self.num_id_switches_handled,
            "suspected_id_switch": self.num_id_switches_handled > 0,
            "max_pfall": round(float(probs.max()), 4) if probs.size else 0.0,
            "mean_pfall": round(float(probs.mean()), 4) if probs.size else 0.0,
            "median_pfall": round(float(np.median(probs)), 4) if probs.size else 0.0,
            f"top{topk}_pfall": [round(v, 4) for v in topk_vals],
            f"mean_top{topk}_pfall": (
                round(float(np.mean(topk_vals)), 4) if topk_vals else 0.0
            ),
            "per_track_max_pfall": per_track_max,
            "alerts": self.alerts,
            "num_alerts": len(self.alerts),
            "diagnosis": self.diagnose(mid_thr=mid_thr, max_low_zone=max_low_zone),
            "elapsed_s": round(time.time() - self.start_time, 2),
        }


def aggregate_summaries(summaries: List[dict], mid_thr: float = 0.5) -> dict:
    valid = [s for s in summaries if "error" not in s]
    errs = [s for s in summaries if "error" in s]
    has_gt = all(s.get("ground_truth") is not None for s in valid)
    out = {
        "num_videos": len(summaries),
        "num_valid": len(valid),
        "num_errors": len(errs),
        "diagnosis_count": {},
    }

    diag_count = {}
    for s in valid:
        d = s.get("diagnosis", "unknown")
        diag_count[d] = diag_count.get(d, 0) + 1
    out["diagnosis_count"] = diag_count

    if has_gt and valid:
        tp = fp = tn = fn = 0
        for s in valid:
            gt = int(s["ground_truth"])
            pred = 1 if s["num_alerts"] > 0 else 0
            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 0 and pred == 1:
                fp += 1
            elif gt == 0 and pred == 0:
                tn += 1
            else:
                fn += 1
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        out.update(
            {
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )

    return out

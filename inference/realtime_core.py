"""
inference/realtime_core.py — 真实视频/实时推理的核心组件

本模块抽出 multitarget_realtime_demo 与 batch_predict 共享的"会因真实视频而变难"的逻辑,
便于两边复用且单元可测。包含 5 个核心组件:

  1. TimeAwareBuffer       — 时间感知的滚动缓冲区,解决"60fps + clip_len=48 只覆盖 0.8s"
  2. TrackMerger           — 处理 ByteTrack 快速动作下的 ID 切换(test7.mp4 那种)
  3. AlertPolicy           — 多策略报警(high 单次 / mid 连续 / top-k 平均)替代单一阈值
  4. ProbabilityLogger     — 每次推理都记 raw/smoothed 概率,未报警视频也能事后诊断
  5. PoseHeuristicScorer   — 用骨架几何兜底识别快摔/翻倒等模型低分片段
  6. VideoSummaryBuilder   — 视频结束时聚合 max/top-k/mean/疑似 ID switch 等诊断信息

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


@dataclass
class PoseHeuristicResult:
    score: float
    reasons: List[str]
    features: dict


class PoseHeuristicScorer:
    """基于 COCO17 骨架的轻量兜底评分。

    目的不是替代 PoseConv3D,而是在真实视频里出现"骨架明显跌倒,但模型低分"
    的短促动作时给报警策略一个独立信号。默认由 CLI 关闭。
    """

    def __init__(self, kpt_thr: float = 0.3, min_frames: int = 12):
        self.kpt_thr = float(kpt_thr)
        self.min_frames = max(4, int(min_frames))

    def score(self, kpts_seq, scores_seq) -> PoseHeuristicResult:
        kpts = list(kpts_seq or [])
        scores = list(scores_seq or [])
        if len(kpts) < self.min_frames or len(kpts) != len(scores):
            return PoseHeuristicResult(0.0, ["insufficient_history"], {})

        posture = []
        hip_y = []
        skel_h = []
        leg_raise = []

        for k, s in zip(kpts, scores):
            k = np.asarray(k, dtype=np.float32)
            s = np.asarray(s, dtype=np.float32)
            if k.shape[0] < 17 or s.shape[0] < 17:
                continue

            valid = s >= self.kpt_thr
            if valid.sum() < 7:
                continue

            ys = k[valid, 1]
            xs = k[valid, 0]
            height = float(max(ys.max() - ys.min(), 1.0))
            width = float(max(xs.max() - xs.min(), 1.0))
            skel_h.append(height)

            shoulder = self._midpoint(k, s, 5, 6)
            hip = self._midpoint(k, s, 11, 12)
            if shoulder is not None and hip is not None:
                vec = shoulder - hip
                angle = float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)))
                aspect = width / height
                posture.append((angle, aspect))
                hip_y.append(float(hip[1]))

                raised = 0.0
                for j in (13, 14, 15, 16):
                    if s[j] >= self.kpt_thr and k[j, 1] < hip[1] - 0.18 * height:
                        raised = 1.0
                        break
                leg_raise.append(raised)

        if len(posture) < max(4, self.min_frames // 2):
            return PoseHeuristicResult(0.0, ["low_pose_confidence"], {})

        angles = np.asarray([p[0] for p in posture], dtype=np.float32)
        aspects = np.asarray([p[1] for p in posture], dtype=np.float32)
        hip_y_arr = np.asarray(hip_y, dtype=np.float32)
        heights = np.asarray(skel_h[-len(hip_y_arr):], dtype=np.float32)

        recent_n = max(3, min(8, len(angles) // 3))
        early_n = max(3, min(8, len(angles) // 3))
        recent_angle = float(np.mean(angles[-recent_n:]))
        recent_aspect = float(np.mean(aspects[-recent_n:]))
        early_angle = float(np.mean(angles[:early_n]))
        early_aspect = float(np.mean(aspects[:early_n]))
        angle_delta = recent_angle - early_angle
        aspect_delta = recent_aspect - early_aspect
        recent_leg_raise = float(np.mean(leg_raise[-recent_n:])) if leg_raise else 0.0
        ref_h = float(np.median(heights)) if heights.size else 1.0
        hip_drop = float(np.mean(hip_y_arr[-recent_n:]) - np.mean(hip_y_arr[:early_n]))
        hip_drop_norm = hip_drop / max(ref_h, 1.0)

        angle_score = self._ramp(recent_angle, 45.0, 78.0)
        angle_change_score = self._ramp(angle_delta, 18.0, 55.0)
        aspect_score = self._ramp(recent_aspect, 0.85, 1.45)
        aspect_change_score = self._ramp(aspect_delta, 0.18, 0.65)
        drop_score = self._ramp(hip_drop_norm, 0.14, 0.45)
        leg_score = float(np.clip(recent_leg_raise, 0.0, 1.0))

        signals = [
            angle_score >= 0.55,
            angle_change_score >= 0.55,
            aspect_score >= 0.55,
            aspect_change_score >= 0.55,
            drop_score >= 0.55,
            leg_score >= 0.50,
        ]
        signal_count = int(sum(signals))

        score = max(
            0.52 * angle_score + 0.30 * drop_score + 0.18 * aspect_score,
            0.45 * angle_score + 0.30 * leg_score + 0.25 * drop_score,
            0.42 * aspect_score + 0.33 * drop_score + 0.25 * angle_score,
            0.35 * angle_change_score + 0.35 * aspect_change_score + 0.30 * leg_score,
        )
        if signal_count < 2:
            score = min(score, 0.55)
        score = float(np.clip(score, 0.0, 1.0))

        reasons = []
        if angle_score >= 0.55:
            reasons.append(f"torso_tilt={recent_angle:.1f}")
        if angle_change_score >= 0.55:
            reasons.append(f"tilt_delta={angle_delta:.1f}")
        if aspect_score >= 0.55:
            reasons.append(f"wide_pose={recent_aspect:.2f}")
        if aspect_change_score >= 0.55:
            reasons.append(f"wide_delta={aspect_delta:.2f}")
        if drop_score >= 0.55:
            reasons.append(f"hip_drop={hip_drop_norm:.2f}")
        if leg_score >= 0.50:
            reasons.append(f"leg_raised={recent_leg_raise:.2f}")
        if not reasons:
            reasons.append("weak_pose_signal")

        return PoseHeuristicResult(
            score=score,
            reasons=reasons,
            features={
                "torso_angle_deg": round(recent_angle, 3),
                "torso_angle_delta_deg": round(angle_delta, 3),
                "pose_aspect": round(recent_aspect, 3),
                "pose_aspect_delta": round(aspect_delta, 3),
                "hip_drop_norm": round(hip_drop_norm, 3),
                "leg_raise": round(recent_leg_raise, 3),
                "signal_count": signal_count,
            },
        )

    def _midpoint(self, kpts: np.ndarray, scores: np.ndarray, a: int, b: int):
        pts = []
        if scores[a] >= self.kpt_thr:
            pts.append(kpts[a])
        if scores[b] >= self.kpt_thr:
            pts.append(kpts[b])
        if not pts:
            return None
        return np.mean(np.asarray(pts, dtype=np.float32), axis=0)

    @staticmethod
    def _ramp(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# ============================================================
# 6.5) FallTrendDetector — 趋势 + 几何 + 消失前留迹的复合检测
#
# 解决 elder_fall_7 那种"信号正在上升但卡在阈值前 0.001"的临界情况。
# 与 PoseHeuristicScorer 互补:那个看单帧/短窗口姿态,这个看时间序列变化。
# ============================================================
@dataclass
class FallTrendResult:
    """单次趋势检测结果。"""
    alert: bool
    strategy: str
    reason: str
    score: float


class FallTrendDetector:
    """趋势 + 几何 + 消失模式的复合摔倒检测器。"""

    def __init__(
        self,
        slope_window: int = 5,
        slope_prob_thr: float = 0.05,
        slope_heur_thr: float = 0.08,
        slope_min_current: float = 0.28,
        geom_window_frames: int = 15,
        bbox_h_drop_ratio: float = 0.35,
        aspect_rise: float = 0.10,
        geom_track_age_min: int = 3,
        disappear_lookback: int = 4,
        disappear_raw_min: float = 0.28,
        disappear_heur_min: float = 0.32,
        disappear_rising_tolerance: float = 0.03,
        autopsy_max_raw_thr: float = 0.30,
        autopsy_max_heur_thr: float = 0.35,
        autopsy_late_peak_ratio: float = 0.5,
        enable_slope: bool = True,
        enable_geometric: bool = True,
        enable_disappear: bool = True,
        enable_autopsy: bool = True,
    ):
        self.slope_window = max(2, int(slope_window))
        self.slope_prob_thr = float(slope_prob_thr)
        self.slope_heur_thr = float(slope_heur_thr)
        self.slope_min_current = float(slope_min_current)

        self.geom_window_frames = max(3, int(geom_window_frames))
        self.bbox_h_drop_ratio = float(bbox_h_drop_ratio)
        self.aspect_rise = float(aspect_rise)
        self.geom_track_age_min = max(0, int(geom_track_age_min))

        self.disappear_lookback = max(2, int(disappear_lookback))
        self.disappear_raw_min = float(disappear_raw_min)
        self.disappear_heur_min = float(disappear_heur_min)
        self.disappear_rising_tolerance = float(disappear_rising_tolerance)

        self.autopsy_max_raw_thr = float(autopsy_max_raw_thr)
        self.autopsy_max_heur_thr = float(autopsy_max_heur_thr)
        self.autopsy_late_peak_ratio = float(np.clip(autopsy_late_peak_ratio, 0.1, 0.95))

        self.enable_slope = bool(enable_slope)
        self.enable_geometric = bool(enable_geometric)
        self.enable_disappear = bool(enable_disappear)
        self.enable_autopsy = bool(enable_autopsy)

    @staticmethod
    def _slope(values) -> float:
        vs = [float(v) for v in values]
        if len(vs) < 2:
            return 0.0
        return (vs[-1] - vs[0]) / (len(vs) - 1)

    @staticmethod
    def _is_rising(values, tolerance: float = 0.03) -> bool:
        vs = [float(v) for v in values]
        if len(vs) < 2:
            return False
        prev = vs[0]
        ups = 0
        for v in vs[1:]:
            if v >= prev - tolerance:
                ups += 1
            prev = v
        return ups >= len(vs) - 2

    def check_slope(self, raw_probs, heuristics) -> FallTrendResult:
        if not self.enable_slope:
            return FallTrendResult(False, "slope", "", 0.0)

        raw = list(raw_probs)[-self.slope_window:]
        heur = list(heuristics)[-self.slope_window:]

        if len(raw) >= 3:
            slope = self._slope(raw)
            cur = float(raw[-1])
            if slope >= self.slope_prob_thr and cur >= self.slope_min_current:
                trig = float(np.clip(cur + slope, 0.0, 1.0))
                return FallTrendResult(True, "slope_prob", f"slope={slope:.3f},cur={cur:.2f}", trig)

        if len(heur) >= 3:
            slope = self._slope(heur)
            cur = float(heur[-1])
            if slope >= self.slope_heur_thr and cur >= self.slope_min_current:
                trig = float(np.clip(cur + slope, 0.0, 1.0))
                return FallTrendResult(True, "slope_heur", f"slope={slope:.3f},cur={cur:.2f}", trig)

        return FallTrendResult(False, "slope", "", 0.0)

    def check_geometric(self, bboxes, track_age: int = 999) -> FallTrendResult:
        if not self.enable_geometric or track_age < self.geom_track_age_min:
            return FallTrendResult(False, "geometric", "", 0.0)

        bb = list(bboxes)[-self.geom_window_frames:]
        if len(bb) < 3:
            return FallTrendResult(False, "geometric", "", 0.0)

        arr = np.asarray(bb, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 4:
            return FallTrendResult(False, "geometric", "", 0.0)

        heights = arr[:, 3] - arr[:, 1]
        widths = arr[:, 2] - arr[:, 0]
        aspects = widths / np.maximum(heights, 1.0)

        n = len(heights)
        q = max(1, n // 4)
        h_first = float(np.median(heights[:q]))
        h_last = float(np.median(heights[-q:]))
        h_drop = (h_first - h_last) / max(h_first, 1.0)

        a_first = float(np.median(aspects[:q]))
        a_last = float(np.median(aspects[-q:]))
        a_rise = a_last - a_first

        h_strong = h_drop >= self.bbox_h_drop_ratio
        a_strong = a_rise >= self.aspect_rise
        h_half = h_drop >= self.bbox_h_drop_ratio * 0.5
        a_half = a_rise >= self.aspect_rise * 0.5

        if (h_strong and a_half) or (a_strong and h_half):
            score = float(np.clip(
                0.5 * (h_drop / max(self.bbox_h_drop_ratio, 1e-6))
                + 0.5 * (a_rise / max(self.aspect_rise, 1e-6)),
                0.0, 1.0,
            ))
            return FallTrendResult(True, "geometric", f"h_drop={h_drop:.2f},a_rise={a_rise:.2f}", score)
        return FallTrendResult(False, "geometric", "", 0.0)

    def check_disappearance(
        self, raw_probs, heuristics,
        track_age: int, min_lost_gap: int,
    ) -> FallTrendResult:
        if not self.enable_disappear or track_age < min_lost_gap:
            return FallTrendResult(False, "disappear", "", 0.0)

        k = self.disappear_lookback
        tail_raw = list(raw_probs)[-k:]
        tail_heur = list(heuristics)[-k:]

        if len(tail_heur) >= 2:
            max_h = float(max(tail_heur))
            if max_h >= self.disappear_heur_min and self._is_rising(
                tail_heur, self.disappear_rising_tolerance
            ):
                return FallTrendResult(
                    True, "disappear_heur",
                    f"max_heur={max_h:.2f},age={track_age},rising_tail={tail_heur}",
                    max_h,
                )

        if len(tail_raw) >= 2:
            max_r = float(max(tail_raw))
            if max_r >= self.disappear_raw_min and self._is_rising(
                tail_raw, self.disappear_rising_tolerance
            ):
                return FallTrendResult(
                    True, "disappear_raw",
                    f"max_raw={max_r:.2f},age={track_age},rising_tail={tail_raw}",
                    max_r,
                )

        return FallTrendResult(False, "disappear", "", 0.0)

    def check_autopsy(self, raw_probs, heuristics) -> FallTrendResult:
        if not self.enable_autopsy:
            return FallTrendResult(False, "autopsy", "", 0.0)

        raw = list(raw_probs)
        heur = list(heuristics)
        if not raw and not heur:
            return FallTrendResult(False, "autopsy", "", 0.0)

        max_r = float(max(raw)) if raw else 0.0
        max_h = float(max(heur)) if heur else 0.0
        n = max(len(raw), len(heur))
        if n == 0:
            return FallTrendResult(False, "autopsy", "", 0.0)

        peak_r_pos = (int(np.argmax(raw)) / max(len(raw) - 1, 1)) if raw else 0.0
        peak_h_pos = (int(np.argmax(heur)) / max(len(heur) - 1, 1)) if heur else 0.0
        late_r = peak_r_pos >= self.autopsy_late_peak_ratio
        late_h = peak_h_pos >= self.autopsy_late_peak_ratio

        if (max_r >= self.autopsy_max_raw_thr and late_r) or \
           (max_h >= self.autopsy_max_heur_thr and late_h):
            return FallTrendResult(
                True, "autopsy",
                f"max_raw={max_r:.2f},max_heur={max_h:.2f},late_peak={peak_r_pos:.2f}/{peak_h_pos:.2f}",
                max(max_r, max_h),
            )
        return FallTrendResult(False, "autopsy", "", 0.0)


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
            "heuristic_score",
            "heuristic_reason",
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
        heuristic_score: float = 0.0,
        heuristic_reason: str = "",
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
            "heuristic_score": round(float(heuristic_score), 6),
            "heuristic_reason": str(heuristic_reason or ""),
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
        self.heuristic_scores: List[float] = []
        self.per_track_probs: Dict[int, List[float]] = {}
        self.per_track_heuristics: Dict[int, List[float]] = {}
        self.alerts: List[dict] = []
        self.num_id_switches_handled = 0
        self.total_frames = 0
        self.total_inferences = 0
        self.error: Optional[str] = None
        self.start_time = time.time()

    def record_inference(self, track_id: int, raw_prob: float, heuristic_score: float = 0.0):
        self.raw_probs.append(float(raw_prob))
        self.heuristic_scores.append(float(heuristic_score))
        self.per_track_probs.setdefault(int(track_id), []).append(float(raw_prob))
        self.per_track_heuristics.setdefault(int(track_id), []).append(float(heuristic_score))
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
        heur = np.asarray(self.heuristic_scores) if self.heuristic_scores else np.zeros(0)
        topk_vals = sorted(self.raw_probs, reverse=True)[:topk]
        topk_heur = sorted(self.heuristic_scores, reverse=True)[:topk]
        per_track_max = {
            int(tid): round(max(plist), 4)
            for tid, plist in self.per_track_probs.items()
        }
        per_track_heur_max = {
            int(tid): round(max(plist), 4)
            for tid, plist in self.per_track_heuristics.items()
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
            "max_pose_heuristic": round(float(heur.max()), 4) if heur.size else 0.0,
            f"top{topk}_pose_heuristic": [round(v, 4) for v in topk_heur],
            f"mean_top{topk}_pose_heuristic": (
                round(float(np.mean(topk_heur)), 4) if topk_heur else 0.0
            ),
            f"top{topk}_pfall": [round(v, 4) for v in topk_vals],
            f"mean_top{topk}_pfall": (
                round(float(np.mean(topk_vals)), 4) if topk_vals else 0.0
            ),
            "per_track_max_pfall": per_track_max,
            "per_track_max_pose_heuristic": per_track_heur_max,
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

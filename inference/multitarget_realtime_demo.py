"""
inference/multitarget_realtime_demo.py — 多目标实时摔倒检测

升级版(v2):在原有多目标实时检测能力之上,新增针对真实手机视频的三类改进:

  1. 时间感知缓冲(--target-fps / --time-window-sec)
       解决"60fps 视频 + clip_len=48 只覆盖 0.8 秒"导致摔倒不完整的问题。
       推理时按目标时间窗口从 raw buffer 中均匀采样 clip_len 帧喂模型。

  2. Track 合并(--track-merge / --track-merge-iou-thr / --track-merge-gap)
       ByteTrack 在快速摔倒时容易 ID 切换(如 test7.mp4 出现 3 个不同 id)。
       此选项把刚消失的 track 暂存,新出现且 IoU/距离接近时继承其 buffer + 状态。

  3. 多策略报警(--high-thr / --topk-mean-thr)
       原来只有"连续 K 次超阈值"一种策略,对短促摔倒不敏感。新增:
         a) 单次 raw_prob ≥ --high-thr 直接报警(快速摔倒救命)
         b) top-k 平均 ≥ --topk-mean-thr 报警(短促摔倒)
         c) 原"连续 K 次 mid"逻辑保留为兜底

  4. 全量概率日志(--prob-log)
       每次推理都记 raw/smoothed 概率。即使整段视频没报警,事后也能从日志
       判断"模型给了 0.49 没过 0.5"还是"模型从头到尾只给 0.05"。

  5. 视频级 summary(--summary-json)
       结束时输出 max/top-k/mean 概率、报警事件、是否疑似 ID switch、
       自动诊断标签(detected / just_below_threshold / partial_signal /
       model_unaware / false_alarm / true_negative)。

向后兼容:所有新参数都默认 off,不传时行为与原版一致。

=============================================================================
CLI 示例
=============================================================================
# 1) 原单视频用法(行为不变)
python inference/multitarget_realtime_demo.py \
    --source 0 --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --max-persons 5

# 2) 真实 60fps 手机视频(新)—— 时间窗口 + 多策略 + 概率日志 + summary
python inference/multitarget_realtime_demo.py \
    --source data/real_test/test7.mp4 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --target-fps 30 --time-window-sec 1.6 \
    --track-merge --track-merge-iou-thr 0.3 --track-merge-gap 15 \
    --high-thr 0.8 --threshold 0.45 --topk-mean-thr 0.55 \
    --prob-log outputs/test7_prob.jsonl \
    --summary-json outputs/test7_summary.json \
    --save-out outputs/test7_demo.mp4 --no-show

# 3) RTSP / 摄像头(--source 0 / rtsp://...) 行为同 (1)/(2),只是 source 不同
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

# 让 import 找到本包
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.extract_pose_yolo26 import load_pose_model, _extract_one_frame
from inference.pose_to_pyskl_format import build_sample
from inference.batch_predict import load_action_model, predict_clip as _predict_clip_fallback
from inference.realtime_core import (
    TimeAwareBuffer, TrackMerger, AlertPolicy, bbox_iou, bbox_center_dist_norm,
    PoseHeuristicScorer, ProbabilityLogger, VideoSummaryBuilder,
    FallTrendDetector, SimpleKalmanBoxTracker, PoseInterpolator, AlertSource,
)


# ============================================================
# COCO 17 点骨骼连线
# ============================================================
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
COLOR_NORMAL = (60, 200, 60)
COLOR_FALL = (60, 60, 240)
COLOR_TREND_FALL = (0, 165, 255)
COLOR_LOGIC_FALL = (220, 60, 220)
COLOR_INTERP = (100, 160, 220)
COLOR_NOID = (160, 160, 160)


def bbox_min_overlap(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size != 4 or b.size != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = min(area_a, area_b)
    return float(inter / denom) if denom > 1e-6 else 0.0


def bbox_aspect(bbox: np.ndarray) -> float:
    bbox = np.asarray(bbox, dtype=np.float32)
    if bbox.size != 4:
        return 0.0
    w = max(float(bbox[2] - bbox[0]), 1.0)
    h = max(float(bbox[3] - bbox[1]), 1.0)
    return w / h


# ============================================================
# 带缓存的 clip predictor —— 已修复 v1 的两个 bug:
#   • Compose 改用 mmengine.dataset.Compose
#   • test_step 输入用 pseudo_collate 包装
# ============================================================
class CachedClipPredictor:
    """把 MMAction2 的 val/test pipeline 构建一次缓存起来,多次推理复用。"""

    def __init__(self, model, device="cuda:0"):
        self.model = model
        self.device = device
        self.pipeline = None
        self._pseudo_collate = None
        try:
            from mmengine.dataset import Compose, pseudo_collate
            cfg = model.cfg
            if hasattr(cfg, "val_pipeline"):
                pcfg = cfg.val_pipeline
            elif hasattr(cfg, "test_pipeline"):
                pcfg = cfg.test_pipeline
            else:
                pcfg = cfg.val_dataloader.dataset.pipeline
            self.pipeline = Compose(pcfg)
            self._pseudo_collate = pseudo_collate
        except Exception as e:  # noqa: BLE001
            print(f"[CachedClipPredictor] 构建缓存 pipeline 失败,回退到 predict_clip:{e}")
            self.pipeline = None

    @torch.no_grad()
    def __call__(self, clip_sample) -> float:
        if self.pipeline is None:
            return _predict_clip_fallback(self.model, clip_sample, device=self.device)

        data = self.pipeline(clip_sample.copy())
        # 用 pseudo_collate 包装单个样本以匹配 MMAction2 v1.x test_step 期望的 batch
        batch = self._pseudo_collate([data])
        result = self.model.test_step(batch)[0]
        score = result.pred_score if hasattr(result, "pred_score") else result.get("pred_score")
        if torch.is_tensor(score):
            score = score.cpu().numpy()
        return float(score[1])


# ============================================================
# 单个 track 的状态(用 TimeAwareBuffer 作为底层缓冲)
# ============================================================
@dataclass
class TrackState:
    """一个 track_id 的全部运行时状态。

    底层缓冲是 TimeAwareBuffer。不传 target_fps/time_window 时,
    window_frames 退化为 clip_len,行为与原 deque(maxlen=clip_len) 等价。

    历史序列(recent_*)用于 FallTrendDetector 的趋势分析:
      - recent_raw_probs:   最近 N 次推理的 raw 概率
      - recent_heuristics:  最近 N 次推理的 pose heuristic 分数
      - recent_bboxes:      最近 N 帧的 bbox(每帧 push 时更新,密度高于推理)
    """
    track_id: int
    clip_len: int
    display_id: Optional[int] = None
    source_fps: float = 30.0
    time_window_sec: float = 0.0
    recent_window: int = 30                # AlertPolicy + FallTrendDetector 共用,扩到 30
    recent_bbox_window: int = 60           # bbox 每帧记一次,需要更大窗口(2 秒@30fps)

    buffer: TimeAwareBuffer = field(default=None, repr=False)
    recent_raw_probs: deque = field(default=None, repr=False)
    recent_heuristics: deque = field(default=None, repr=False)
    recent_bboxes: deque = field(default=None, repr=False)

    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_kpts: np.ndarray = field(default_factory=lambda: np.zeros((17, 2), dtype=np.float32))
    last_scores: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=np.float32))

    last_seen_frame: int = 0
    frames_since_infer: int = 10 ** 9
    infer_count: int = 0

    last_prob: float = 0.0           # raw
    smoothed_prob: float = 0.0       # EMA
    heuristic_score: float = 0.0
    heuristic_reason: str = ""
    over_thr_streak: int = 0
    alerted: bool = False
    alert_frames_left: int = 0
    ever_alerted: bool = False
    last_alert_reason: str = ""
    lost_track_alerted: bool = False
    kalman: Optional[SimpleKalmanBoxTracker] = field(default=None, repr=False)
    n_interpolated_frames: int = 0
    total_interpolated_frames: int = 0
    last_real_seen_frame: int = -1
    alert_source_tag: str = ""

    def __post_init__(self):
        if self.display_id is None:
            self.display_id = self.track_id
        if self.buffer is None:
            self.buffer = TimeAwareBuffer(
                clip_len=self.clip_len,
                source_fps=self.source_fps,
                time_window_sec=self.time_window_sec,
            )
        if self.recent_raw_probs is None:
            self.recent_raw_probs = deque(maxlen=max(self.recent_window, 5))
        if self.recent_heuristics is None:
            self.recent_heuristics = deque(maxlen=max(self.recent_window, 5))
        if self.recent_bboxes is None:
            self.recent_bboxes = deque(maxlen=max(self.recent_bbox_window, 10))

    @property
    def is_ready(self) -> bool:
        return self.buffer.is_ready

    def push(self, kpt: np.ndarray, score: np.ndarray, bbox: np.ndarray, frame_idx: int):
        self.buffer.push(kpt, score)
        self.bbox = bbox.astype(np.float32)
        # 每帧 push 都记录 bbox 历史 (供 FallTrendDetector 策略 C 用)
        self.recent_bboxes.append(self.bbox.copy())
        self.last_kpts = kpt.astype(np.float32)
        self.last_scores = score.astype(np.float32)
        self.last_seen_frame = frame_idx
        self.last_real_seen_frame = frame_idx
        self.frames_since_infer += 1
        self.n_interpolated_frames = 0
        if self.kalman is None:
            self.kalman = SimpleKalmanBoxTracker(self.bbox, frame_idx)
        else:
            self.kalman.update(self.bbox, frame_idx)

    def push_interpolated(self, interp_frame, frame_idx: int):
        self.buffer.push(interp_frame.kpts, interp_frame.scores)
        self.bbox = interp_frame.bbox.astype(np.float32)
        self.recent_bboxes.append(self.bbox.copy())
        self.last_kpts = interp_frame.kpts.astype(np.float32)
        self.last_scores = interp_frame.scores.astype(np.float32)
        self.last_seen_frame = frame_idx
        self.frames_since_infer += 1
        self.n_interpolated_frames += 1
        self.total_interpolated_frames += 1

    def adopt(self, tomb):
        """从 tombstone 继承历史 buffer + 概率状态。"""
        if tomb is None:
            return
        self.display_id = int(getattr(tomb, "display_id", tomb.track_id))
        # buffer 迁移(从 tomb.buffer 继承最近若干帧)
        self.buffer.inherit_from(tomb.buffer)
        # 概率状态延续
        self.last_prob = float(tomb.last_raw_prob)
        self.smoothed_prob = float(tomb.last_smoothed_prob)
        self.over_thr_streak = int(tomb.over_thr_streak)
        kalman = getattr(tomb, "kalman", None)
        if kalman is not None:
            self.kalman = kalman
        # 继承历史序列(如果 tomb 有的话)
        for hist_name in ("recent_raw_probs", "recent_heuristics", "recent_bboxes"):
            tomb_hist = getattr(tomb, hist_name, None)
            self_hist = getattr(self, hist_name, None)
            if tomb_hist is not None and self_hist is not None:
                for v in list(tomb_hist):
                    self_hist.append(v)

    def adopt_state(self, other: "TrackState"):
        """从刚断开的 active track 继承状态,用于处理同帧/短间隔 ID 切换。"""
        if other is None:
            return
        self.display_id = int(other.display_id)
        self.buffer.inherit_from(other.buffer)
        self.last_prob = float(other.last_prob)
        self.smoothed_prob = float(other.smoothed_prob)
        self.heuristic_score = float(other.heuristic_score)
        self.heuristic_reason = str(other.heuristic_reason or "")
        self.over_thr_streak = int(other.over_thr_streak)
        self.recent_raw_probs.extend(list(other.recent_raw_probs))
        self.recent_heuristics.extend(list(other.recent_heuristics))
        self.recent_bboxes.extend(list(other.recent_bboxes))
        if other.kalman is not None:
            self.kalman = other.kalman
        self.last_real_seen_frame = int(other.last_real_seen_frame)
        self.alerted = bool(other.alerted)
        self.alert_frames_left = int(other.alert_frames_left)
        self.ever_alerted = bool(other.ever_alerted)
        self.last_alert_reason = str(other.last_alert_reason or "")
        self.lost_track_alerted = bool(other.lost_track_alerted)
        self.alert_source_tag = str(other.alert_source_tag or "")


# ============================================================
# 多目标检测器(集成 TrackMerger / AlertPolicy / ProbLogger / Summary)
# ============================================================
class MultiTrackFallDetector:
    """维护 {track_id: TrackState},负责喂数据、调度分类、报警判定、清理过期 track。

    新增组件(均可为 None,None 时走简化行为):
      - track_merger: TrackMerger 处理 ID 切换
      - alert_policy: AlertPolicy 替代单一阈值
      - prob_logger:  ProbabilityLogger 每次推理都记
      - summary:      VideoSummaryBuilder 视频级聚合
    """

    def __init__(
        self,
        predictor: CachedClipPredictor,
        clip_len: int = 48,
        source_fps: float = 30.0,
        time_window_sec: float = 0.0,
        infer_every: int = 6,
        threshold: float = 0.5,
        alert_k: int = 2,
        alert_hold_frames: int = 45,
        ema: float = 0.5,
        track_timeout: int = 30,
        kpt_thr: float = 0.3,
        source_name: str = "",
        # 新组件(默认 None)
        track_merger: Optional[TrackMerger] = None,
        alert_policy: Optional[AlertPolicy] = None,
        prob_logger: Optional[ProbabilityLogger] = None,
        summary: Optional[VideoSummaryBuilder] = None,
        pose_heuristic: Optional[PoseHeuristicScorer] = None,
        pose_heuristic_thr: float = 1.01,
        lost_track_alert: bool = False,
        lost_track_min_gap: int = 8,
        lost_track_heuristic_thr: float = 0.45,
        lost_track_model_thr: float = 0.35,
        track_merge_same_frame: bool = False,
        # FallTrendDetector — 趋势 + 几何 + 消失模式
        fall_trend: Optional[FallTrendDetector] = None,
        pose_interpolator: Optional[PoseInterpolator] = None,
        # 兼容字段
        event_logger=None,
    ):
        self.predictor = predictor
        self.clip_len = clip_len
        self.source_fps = source_fps
        self.time_window_sec = time_window_sec
        self.infer_every = max(1, infer_every)
        self.threshold = threshold
        self.alert_k = max(1, alert_k)
        self.alert_hold_frames = alert_hold_frames
        self.ema = float(np.clip(ema, 0.05, 1.0))
        self.track_timeout = track_timeout
        self.kpt_thr = kpt_thr
        self.source_name = source_name

        self.track_merger = track_merger
        self.alert_policy = alert_policy
        self.prob_logger = prob_logger
        self.summary = summary
        self.pose_heuristic = pose_heuristic
        self.pose_heuristic_thr = float(pose_heuristic_thr)
        self.lost_track_alert = bool(lost_track_alert)
        self.lost_track_min_gap = max(1, int(lost_track_min_gap))
        self.lost_track_heuristic_thr = float(lost_track_heuristic_thr)
        self.lost_track_model_thr = float(lost_track_model_thr)
        self.track_merge_same_frame = bool(track_merge_same_frame)
        self.fall_trend = fall_trend
        self.pose_interpolator = pose_interpolator
        self.event_logger = event_logger

        self.tracks: Dict[int, TrackState] = {}
        self.alerted_ids = set()
        self.last_infer_ms = 0.0

    # --------------------------------------------------------
    def _new_track(self, tid: int) -> TrackState:
        return TrackState(
            track_id=tid,
            clip_len=self.clip_len,
            source_fps=self.source_fps,
            time_window_sec=self.time_window_sec,
        )

    # --------------------------------------------------------
    def _try_adopt_recent_inactive_track(
        self,
        st: TrackState,
        new_tid: int,
        new_bbox: np.ndarray,
        frame_idx: int,
        img_diag: float,
        current_ids: set,
    ):
        """让新 ID 直接继承刚刚断开的 active track。

        旧逻辑只有 track 超过 timeout 被清理时才进入 tombstone,但快速摔倒时新 ID
        往往在旧 ID 刚消失的 1-2 帧内出现,此时旧 track 仍在 self.tracks 中。
        这里直接匹配这些最近未出现的 active tracks,解决同帧/短间隔 ID switch。
        """
        if self.track_merger is None:
            return None

        best_tid = None
        best_st = None
        best_score = 0.0
        best_reason = None

        for old_tid, old_st in self.tracks.items():
            if old_tid == new_tid or old_tid in current_ids:
                continue
            gap = frame_idx - old_st.last_seen_frame
            if gap < 0 or gap > self.track_merger.max_gap_frames:
                continue

            iou = bbox_iou(old_st.bbox, new_bbox)
            dist = bbox_center_dist_norm(old_st.bbox, new_bbox, img_diag)
            score = 0.0
            reason = None
            if iou >= self.track_merger.iou_thr:
                score = iou
                reason = f"active_iou={iou:.2f}"
            elif dist <= self.track_merger.center_dist_norm_thr:
                score = max(0.0, 1.0 - dist / max(self.track_merger.center_dist_norm_thr, 1e-6))
                reason = f"active_dist={dist:.3f}"

            if score > best_score:
                best_tid = old_tid
                best_st = old_st
                best_score = score
                best_reason = reason

        if best_st is None:
            return None

        st.adopt_state(best_st)
        self.track_merger.merge_log.append({
            "frame": frame_idx,
            "new_track_id": int(new_tid),
            "inherited_from": int(best_tid),
            "display_id": int(best_st.display_id),
            "reason": best_reason,
        })
        del self.tracks[best_tid]
        return best_tid

    # --------------------------------------------------------
    def _try_adopt_same_frame_duplicate(
        self,
        st: TrackState,
        new_tid: int,
        new_bbox: np.ndarray,
        frame_idx: int,
        img_diag: float,
        seen_now: set,
    ):
        """Merge a same-frame split ID for the same falling person."""
        if self.track_merger is None or not self.track_merge_same_frame:
            return None

        best_tid = None
        best_st = None
        best_score = 0.0
        best_reason = None
        new_aspect = bbox_aspect(new_bbox)

        for old_tid in list(seen_now):
            if old_tid == new_tid or old_tid not in self.tracks:
                continue
            old_st = self.tracks[old_tid]
            if old_st.last_seen_frame != frame_idx:
                continue

            old_bbox = old_st.bbox
            iou = bbox_iou(old_bbox, new_bbox)
            min_ov = bbox_min_overlap(old_bbox, new_bbox)
            dist = bbox_center_dist_norm(old_bbox, new_bbox, img_diag)
            old_aspect = bbox_aspect(old_bbox)

            fall_like = (
                old_st.heuristic_score >= 0.35
                or old_st.smoothed_prob >= 0.20
                or old_aspect >= 1.20
                or new_aspect >= 1.20
            )
            spatial_overlap = iou >= max(0.10, self.track_merger.iou_thr * 0.4) or min_ov >= 0.45
            spatial_near = (
                dist <= min(0.10, self.track_merger.center_dist_norm_thr)
                and (old_aspect >= 1.05 or new_aspect >= 1.05)
            )
            if not (fall_like and (spatial_overlap or spatial_near)):
                continue

            score = iou + 0.5 * min_ov + max(0.0, 0.10 - dist)
            reason_parts = [
                f"same_frame_iou={iou:.2f}",
                f"overlap={min_ov:.2f}",
                f"dist={dist:.3f}",
            ]
            if old_st.heuristic_score >= 0.35:
                reason_parts.append(f"old_heur={old_st.heuristic_score:.2f}")
            if old_st.smoothed_prob >= 0.20:
                reason_parts.append(f"old_p={old_st.smoothed_prob:.2f}")
            reason = ",".join(reason_parts)

            if score > best_score:
                best_tid = old_tid
                best_st = old_st
                best_score = score
                best_reason = reason

        if best_st is None:
            return None

        st.adopt_state(best_st)
        self.track_merger.merge_log.append({
            "frame": frame_idx,
            "new_track_id": int(new_tid),
            "inherited_from": int(best_tid),
            "display_id": int(best_st.display_id),
            "reason": best_reason,
        })
        del self.tracks[best_tid]
        seen_now.discard(best_tid)
        return best_tid

    # --------------------------------------------------------
    def update(self, frame_idx, kpts, scores, bboxes, track_ids, img_shape, frame=None):
        H, W = img_shape
        img_diag = float(np.hypot(H, W))
        current_ids = {
            int(tid)
            for i, tid in enumerate(track_ids)
            if int(tid) >= 0 and np.any(kpts[i])
        }
        seen_now = set()
        suppressed_ids = set()

        # 1. 喂数据 + 尝试 track 合并
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid < 0 or tid in suppressed_ids:
                continue
            kpt = kpts[i]
            scr = scores[i]
            if not np.any(kpt):
                continue

            # 新出现的 track:尝试从 tombstones 继承
            if tid not in self.tracks:
                st = self._new_track(tid)
                if self.track_merger is not None:
                    adopted_tid = self._try_adopt_same_frame_duplicate(
                        st=st,
                        new_tid=tid,
                        new_bbox=bboxes[i],
                        frame_idx=frame_idx,
                        img_diag=img_diag,
                        seen_now=seen_now,
                    )
                    if adopted_tid is None:
                        adopted_tid = self._try_adopt_recent_inactive_track(
                            st=st,
                            new_tid=tid,
                            new_bbox=bboxes[i],
                            frame_idx=frame_idx,
                            img_diag=img_diag,
                            current_ids=current_ids,
                        )
                    if adopted_tid is not None:
                        suppressed_ids.add(adopted_tid)
                    else:
                        tomb = self.track_merger.try_match(
                            new_track_id=tid, new_bbox=bboxes[i],
                            current_frame=frame_idx, img_diag=img_diag,
                        )
                        if tomb is not None:
                            st.adopt(tomb)
                self.tracks[tid] = st

            self.tracks[tid].push(kpt, scr, bboxes[i], frame_idx)
            seen_now.add(tid)

        # 1.5) Tracking continuity: keep short lost tracks feeding the pose buffer.
        if self.pose_interpolator is not None:
            for tid, st in list(self.tracks.items()):
                if tid in seen_now:
                    continue
                if st.kalman is None or st.last_real_seen_frame < 0:
                    continue
                gap = frame_idx - st.last_real_seen_frame
                if gap < 1:
                    continue

                pred_bbox = st.kalman.predict_bbox(frame_idx)
                pred_bbox = np.array([
                    np.clip(pred_bbox[0], 0, W - 1),
                    np.clip(pred_bbox[1], 0, H - 1),
                    np.clip(pred_bbox[2], 0, W - 1),
                    np.clip(pred_bbox[3], 0, H - 1),
                ], dtype=np.float32)
                hist_kpts = list(st.buffer.kpts)[-6:]
                hist_scores = list(st.buffer.scores)[-6:]
                hist_bboxes = list(st.recent_bboxes)[-6:]
                interp_frame = self.pose_interpolator.extrapolate_one(
                    hist_kpts,
                    hist_scores,
                    hist_bboxes,
                    gap=st.n_interpolated_frames + 1,
                    kalman_predicted_bbox=pred_bbox,
                )
                if interp_frame is None:
                    continue
                st.push_interpolated(interp_frame, frame_idx)

        # 2. 调度推理(交错相位)
        infer_ms_accum = 0.0
        for tid, st in self.tracks.items():
            if not st.is_ready:
                continue
            was_pushed = (tid in seen_now) or (st.last_seen_frame == frame_idx)
            if not was_pushed:
                continue
            first_time = st.infer_count == 0
            phase_due = (frame_idx % self.infer_every) == (tid % self.infer_every)
            due = st.frames_since_infer >= self.infer_every and phase_due
            if not (first_time or due):
                continue

            t0 = time.time()
            raw_prob = self._infer_one(st, img_shape)
            infer_ms_accum += (time.time() - t0) * 1000
            st.frames_since_infer = 0
            st.infer_count += 1
            st.last_prob = raw_prob
            st.recent_raw_probs.append(raw_prob)
            if self.pose_heuristic is not None:
                heur = self.pose_heuristic.score(st.buffer.kpts, st.buffer.scores)
                st.heuristic_score = heur.score
                st.heuristic_reason = ",".join(heur.reasons)
            else:
                st.heuristic_score = 0.0
                st.heuristic_reason = ""
            # 每次推理后同步更新 heuristic 历史 (供 FallTrendDetector 用)
            st.recent_heuristics.append(st.heuristic_score)

            # EMA 平滑
            if st.infer_count == 1:
                st.smoothed_prob = raw_prob
            else:
                st.smoothed_prob = self.ema * raw_prob + (1 - self.ema) * st.smoothed_prob

            # 报警判定 + 写日志
            decision = self._update_alert(st, frame_idx, frame)

            # FallTrendDetector 策略 B + C: 推理刚结束时立刻检查趋势 / 几何
            # 这是对 fall_7 那种"信号上升但卡阈值前"的最早救援机会
            if (
                self.fall_trend is not None
                and not st.ever_alerted
                and not decision["alert_onset"]
            ):
                ft_decision = self._check_fall_trend_at_infer(st, frame_idx, frame)
                if ft_decision is not None:
                    # 触发后用 fall_trend 的结果覆盖 decision,确保后续 prob_logger / summary 正确记录
                    decision = ft_decision

            # 记录:任何一次推理都写 prob log + summary
            if self.prob_logger is not None:
                self.prob_logger.log(
                    frame_idx=frame_idx, track_id=st.display_id,
                    raw_prob=raw_prob, smoothed_prob=st.smoothed_prob,
                    buffer_len=st.buffer.buffer_len, bbox=st.bbox,
                    alerted=decision["alert_onset"],
                    alert_reason=decision["reason"],
                    heuristic_score=st.heuristic_score,
                    heuristic_reason=st.heuristic_reason,
                )
            if self.summary is not None:
                self.summary.record_inference(st.display_id, raw_prob, st.heuristic_score)
                if decision["alert_onset"]:
                    self.summary.record_alert(
                        frame_idx=frame_idx, track_id=st.display_id,
                        prob=decision["triggering_prob"], reason=decision["reason"],
                    )

        if infer_ms_accum > 0:
            self.last_infer_ms = infer_ms_accum

        # 2.5 跌倒姿态后跟踪丢失:躺倒/遮挡导致 pose 不再输出时的工程兜底
        if self.lost_track_alert:
            for st in self.tracks.values():
                age = frame_idx - st.last_seen_frame
                if age < self.lost_track_min_gap or st.lost_track_alerted or st.ever_alerted:
                    continue
                # === bug fix ===
                # 旧实现用 st.smoothed_prob 比 lost_track_model_thr,但 EMA 有滞后:
                # fall_7 帧 133 时 raw=0.361 / smoothed 才 0.255,导致 model_signal=False。
                # 改为看最近 3 次 raw 推理的最大值,能抓住"摔倒瞬间的瞬时高分"。
                recent_raw_top3 = list(st.recent_raw_probs)[-3:] if st.recent_raw_probs else []
                recent_max_raw = max(recent_raw_top3) if recent_raw_top3 else 0.0
                model_signal = recent_max_raw >= self.lost_track_model_thr
                logic_signal = st.heuristic_score >= self.lost_track_heuristic_thr

                # === FallTrendDetector 策略 A: disappearance ===
                # 即使绝对阈值差一点点(fall_7 的 heur=0.449 vs 阈值 0.45),
                # 只要 raw/heur 在 track 消失前呈"上升趋势 + 高位",就视为跌倒
                disappear_res = None
                if self.fall_trend is not None:
                    disappear_res = self.fall_trend.check_disappearance(
                        list(st.recent_raw_probs),
                        list(st.recent_heuristics),
                        track_age=age,
                        min_lost_gap=self.lost_track_min_gap,
                    )
                disappear_signal = disappear_res is not None and disappear_res.alert

                if not (model_signal or logic_signal or disappear_signal):
                    continue

                reasons = [f"lost_gap={age}"]
                if model_signal:
                    reasons.append(f"raw_max3={recent_max_raw:.2f}")
                if logic_signal:
                    reasons.append(f"heur={st.heuristic_score:.2f}")
                    if st.heuristic_reason:
                        reasons.append(st.heuristic_reason)
                if disappear_signal:
                    reasons.append(disappear_res.strategy)
                    reasons.append(disappear_res.reason)
                reason = "track_lost_after_fall_pose:" + ",".join(reasons)
                trigger = max(
                    recent_max_raw,
                    st.heuristic_score,
                    disappear_res.score if disappear_signal else 0.0,
                )
                st.alerted = True
                st.ever_alerted = True
                st.lost_track_alerted = True
                st.alert_source_tag = "trend" if disappear_signal and not (model_signal or logic_signal) else "logic"
                st.alert_frames_left = max(st.alert_frames_left, self.alert_hold_frames)
                st.last_alert_reason = reason
                self.alerted_ids.add(st.display_id)
                if self.event_logger is not None:
                    self.event_logger.log(
                        frame_idx=frame_idx, track_id=st.display_id,
                        fall_prob=trigger, bbox=st.bbox,
                        source=self.source_name, event="onset",
                        reason=reason, frame=frame,
                    )
                if self.summary is not None:
                    self.summary.record_alert(
                        frame_idx=frame_idx, track_id=st.display_id,
                        prob=trigger, reason=reason,
                    )

        # 3. 报警横幅倒计时
        for st in self.tracks.values():
            if st.alert_frames_left > 0:
                st.alert_frames_left -= 1
                if st.alert_frames_left == 0:
                    st.alerted = False
                    st.over_thr_streak = 0

        # 4. 清理过期 track(消亡时若有合并器,放入 tombstones)
        stale = [tid for tid, st in self.tracks.items()
                 if frame_idx - st.last_seen_frame > self.track_timeout
                 and not (st.alerted and st.alert_frames_left > 0)]
        for tid in stale:
            st = self.tracks[tid]
            # === FallTrendDetector 策略 D: autopsy ===
            # track 即将被永久清理,最后一次审判:看完整生命中是否有强迹象
            if (
                self.fall_trend is not None
                and not st.ever_alerted
            ):
                au_res = self.fall_trend.check_autopsy(
                    list(st.recent_raw_probs),
                    list(st.recent_heuristics),
                )
                if au_res.alert:
                    reason = f"autopsy:{au_res.reason}"
                    if self.event_logger is not None:
                        self.event_logger.log(
                            frame_idx=frame_idx, track_id=st.display_id,
                            fall_prob=au_res.score, bbox=st.bbox,
                            source=self.source_name, event="onset",
                            reason=reason, frame=None,
                        )
                    if self.summary is not None:
                        self.summary.record_alert(
                            frame_idx=frame_idx, track_id=st.display_id,
                            prob=au_res.score, reason=reason,
                        )
                    self.alerted_ids.add(st.display_id)
                    st.ever_alerted = True
                    st.last_alert_reason = reason
                    st.alert_source_tag = "trend"

            if self.track_merger is not None:
                self.track_merger.register_death(
                    track_id=tid, last_frame=st.last_seen_frame,
                    display_id=st.display_id,
                    last_bbox=st.bbox, buffer=st.buffer,
                    last_smoothed_prob=st.smoothed_prob,
                    last_raw_prob=st.last_prob,
                    over_thr_streak=st.over_thr_streak,
                    kalman=st.kalman,
                    recent_raw_probs=st.recent_raw_probs,
                    recent_heuristics=st.recent_heuristics,
                    recent_bboxes=st.recent_bboxes,
                )
            del self.tracks[tid]
        if self.track_merger is not None:
            self.track_merger.prune(frame_idx)

    # --------------------------------------------------------
    def _check_fall_trend_at_infer(self, st: TrackState, frame_idx: int, frame):
        """每次推理后检查 FallTrendDetector 的策略 B (slope) + C (geometric)。

        若触发,立即报警并返回符合 _update_alert 格式的 dict,
        让上层逻辑能继续记录 prob_log / summary。
        """
        if self.fall_trend is None or st.ever_alerted:
            return None

        # 策略 B: 变化率 (raw_prob / heur 短窗口斜率)
        slope_res = self.fall_trend.check_slope(
            list(st.recent_raw_probs),
            list(st.recent_heuristics),
        )
        if slope_res.alert:
            return self._fire_fall_trend_alert(st, frame_idx, frame, slope_res)

        # 策略 C: 几何形变 (bbox 高度急剧下降 + aspect 上升)
        track_age = max(1, len(st.recent_bboxes))
        geom_res = self.fall_trend.check_geometric(
            list(st.recent_bboxes), track_age=track_age,
        )
        if geom_res.alert:
            return self._fire_fall_trend_alert(st, frame_idx, frame, geom_res)
        return None

    # --------------------------------------------------------
    def _fire_fall_trend_alert(self, st: TrackState, frame_idx, frame, res):
        """统一发射 FallTrendDetector 触发的报警事件,返回标准 decision dict。"""
        reason = f"fall_trend:{res.strategy}:{res.reason}"
        st.alerted = True
        st.ever_alerted = True
        st.alert_frames_left = max(st.alert_frames_left, self.alert_hold_frames)
        st.last_alert_reason = reason
        st.alert_source_tag = "trend"
        self.alerted_ids.add(st.display_id)
        if self.event_logger is not None:
            self.event_logger.log(
                frame_idx=frame_idx, track_id=st.display_id,
                fall_prob=res.score, bbox=st.bbox,
                source=self.source_name, event="onset",
                reason=reason, frame=frame,
            )
        return {
            "alert_onset": True,
            "reason": reason,
            "triggering_prob": float(res.score),
        }

    # --------------------------------------------------------
    def _infer_one(self, st: TrackState, img_shape) -> float:
        """对单个 track 跑一次动作分类。

        关键改动:从 TimeAwareBuffer 均匀采样 clip_len 帧,而不是直接喂全部 buffer。
        """
        try:
            sampled = st.buffer.sample_clip()
            if sampled is None:
                return st.last_prob
            kpts_list, scrs_list = sampled
            # (17,2) -> (1,17,2) / (17,) -> (1,17),给 build_sample
            kpts_seq = [k[None, ...] for k in kpts_list]
            scrs_seq = [s[None, ...] for s in scrs_list]
            sample = build_sample(
                keypoints_seq=kpts_seq,
                scores_seq=scrs_seq,
                img_shape=img_shape,
                frame_dir=f"track{st.track_id}",
            )
            return float(self.predictor(sample))
        except Exception as e:  # noqa: BLE001
            print(f"[infer] track {st.track_id} 推理异常,沿用上次概率:{e}")
            return st.last_prob

    # --------------------------------------------------------
    def _update_alert(self, st: TrackState, frame_idx, frame) -> dict:
        """报警判定。返回这次推理是否首次触发报警 + 触发原因。"""
        # 维护 mid streak(供 AlertPolicy 的 consec_mid 用)
        if st.smoothed_prob >= self.threshold:
            st.over_thr_streak += 1
        else:
            st.over_thr_streak = 0

        # 判定
        if self.alert_policy is not None:
            d = self.alert_policy.evaluate(
                raw_prob=st.last_prob,
                smoothed_prob=st.smoothed_prob,
                over_mid_streak=st.over_thr_streak,
                recent_raw_probs=list(st.recent_raw_probs),
            )
            should_alert = d.alert
            reason = d.reason
            trig_prob = d.triggering_prob
            source_tag = "model" if should_alert else ""
        else:
            # 旧逻辑:连续 K 次 smoothed > threshold
            should_alert = st.over_thr_streak >= self.alert_k and st.smoothed_prob >= self.threshold
            reason = "consec_mid" if should_alert else ""
            trig_prob = st.smoothed_prob
            source_tag = "model" if should_alert else ""

        if (
            not should_alert
            and self.pose_heuristic is not None
            and st.heuristic_score >= self.pose_heuristic_thr
        ):
            should_alert = True
            reason = "pose_heuristic"
            if st.heuristic_reason:
                reason = f"{reason}:{st.heuristic_reason}"
            trig_prob = st.heuristic_score
            source_tag = "logic"

        # 状态机:首次触发 → onset;持续超阈值 → 维持 alert_frames_left
        alert_onset = False
        if should_alert:
            st.alert_frames_left = self.alert_hold_frames
            st.last_alert_reason = reason
            if source_tag:
                st.alert_source_tag = source_tag
            if not st.alerted:
                st.alerted = True
                st.ever_alerted = True
                self.alerted_ids.add(st.display_id)
                alert_onset = True
                if self.event_logger is not None:
                    self.event_logger.log(
                        frame_idx=frame_idx, track_id=st.display_id,
                        fall_prob=trig_prob, bbox=st.bbox,
                        source=self.source_name, event="onset",
                        reason=reason,
                        frame=frame,
                    )

        return {"alert_onset": alert_onset, "reason": reason,
                "triggering_prob": trig_prob}

    # --------------------------------------------------------
    def snapshot(self) -> List[TrackState]:
        return list(self.tracks.values())

    def visible_snapshot(self, frame_idx: int, max_age: int = 8,
                         alert_max_age: int = 15) -> List[TrackState]:
        """Return tracks that should still be drawn on the current frame."""
        visible = []
        max_age = max(0, int(max_age))
        alert_max_age = max(max_age, int(alert_max_age))
        for st in self.tracks.values():
            age = frame_idx - st.last_seen_frame
            if age <= max_age:
                visible.append(st)
                continue
            if st.alerted and st.alert_frames_left > 0 and age <= alert_max_age:
                visible.append(st)
        return visible

    @property
    def active_count(self) -> int:
        return len(self.tracks)


# ============================================================
# 事件日志(JSONL)+ 可选 snapshot
# ============================================================
class EventLogger:
    def __init__(self, jsonl_path: Optional[str], snapshot_dir: Optional[str] = None,
                 repeat_sec: float = 0.0, fps: float = 30.0):
        self.path = Path(jsonl_path) if jsonl_path else None
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.repeat_frames = int(repeat_sec * fps) if repeat_sec > 0 else 0
        self._last_log_frame: Dict[int, int] = {}
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def log(self, frame_idx, track_id, fall_prob, bbox, source, event="onset",
            reason="", frame=None):
        if event == "ongoing" and self.repeat_frames > 0:
            last = self._last_log_frame.get(track_id, -10 ** 9)
            if frame_idx - last < self.repeat_frames:
                return
        self._last_log_frame[track_id] = frame_idx

        rec = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame_idx": int(frame_idx),
            "track_id": int(track_id),
            "fall_prob": round(float(fall_prob), 4),
            "bbox": [int(x) for x in np.asarray(bbox).tolist()],
            "source": str(source),
            "event": event,
            "reason": str(reason or ""),
        }
        if self.snapshot_dir is not None and frame is not None and event == "onset":
            fn = self.snapshot_dir / f"fall_t{track_id}_f{frame_idx}.jpg"
            try:
                cv2.imwrite(str(fn), frame)
                rec["snapshot"] = str(fn)
            except Exception as e:  # noqa: BLE001
                print(f"[EventLogger] snapshot 写失败:{e}")

        if self._fh:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
        suffix = f" reason={reason}" if reason else ""
        print(f"[ALERT] frame={frame_idx} track={track_id} P(fall)={rec['fall_prob']} ({event}){suffix}")

    def close(self):
        if self._fh:
            self._fh.close()


# ============================================================
# 可视化叠加
# ============================================================
def draw_multitrack_overlay(frame, tracks: List[TrackState], threshold, kpt_thr,
                            fps, infer_ms, active_count, total_alerts,
                            noid_dets=None):
    H, W = frame.shape[:2]

    for st in tracks:
        trend_alert = _is_trend_alert(st)
        logic_alert = _is_logic_alert(st)
        model_score_fall = st.smoothed_prob >= threshold
        model_alert = (st.alerted and not logic_alert and not trend_alert) or model_score_fall
        model_and_logic = model_alert and logic_alert
        model_and_trend = model_alert and trend_alert
        is_fall = st.alerted or model_score_fall
        if model_alert:
            color = COLOR_FALL
        elif trend_alert:
            color = COLOR_TREND_FALL
        elif logic_alert:
            color = COLOR_LOGIC_FALL
        elif getattr(st, "n_interpolated_frames", 0) > 0:
            color = COLOR_INTERP
        else:
            color = COLOR_NORMAL
        _draw_skeleton(frame, st.last_kpts, st.last_scores, color, kpt_thr)
        if st.bbox is not None and np.any(st.bbox):
            x1, y1, x2, y2 = st.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if model_and_logic and model_and_trend:
                status = "MODEL+TREND+LOGIC FALL"
            elif model_and_trend:
                status = "MODEL+TREND FALL"
            elif model_and_logic:
                status = "MODEL+LOGIC FALL"
            elif trend_alert:
                status = "TREND FALL"
            elif logic_alert:
                status = "LOGIC FALL"
            elif getattr(st, "n_interpolated_frames", 0) > 0:
                status = f"INTERP({st.n_interpolated_frames})"
            elif is_fall:
                status = "MODEL FALL"
            else:
                status = "NORMAL"
            label = f"id:{st.display_id} {status} P:{st.smoothed_prob:.2f}"
            if getattr(st, "heuristic_score", 0.0) >= 0.3 or logic_alert:
                label += f" H:{st.heuristic_score:.2f}"
            _draw_label(frame, label, (x1, y1), color)
            if st.alerted:
                if model_and_logic or model_and_trend:
                    parts = ["MODEL"]
                    if trend_alert:
                        parts.append("TREND")
                    if logic_alert:
                        parts.append("LOGIC")
                    tag = f"{'+'.join(parts)} FALL P:{st.smoothed_prob:.2f} H:{st.heuristic_score:.2f}"
                    _draw_label(frame, tag, (x1, y2 + 24), COLOR_FALL, scale=0.72)
                    reason = _short_alert_reason(st.last_alert_reason)
                    if reason:
                        _draw_label(frame, reason, (x1, y2 + 48), COLOR_FALL, scale=0.52)
                elif trend_alert:
                    tag = f"TREND FALL P:{st.smoothed_prob:.2f} H:{st.heuristic_score:.2f}"
                    reason = _short_alert_reason(st.last_alert_reason)
                    _draw_label(frame, tag, (x1, y2 + 24), COLOR_TREND_FALL, scale=0.78)
                    if reason:
                        _draw_label(frame, reason, (x1, y2 + 48), COLOR_TREND_FALL, scale=0.52)
                elif logic_alert:
                    tag = f"LOGIC FALL H:{st.heuristic_score:.2f}"
                    reason = _short_alert_reason(st.last_alert_reason)
                    _draw_label(frame, tag, (x1, y2 + 24), COLOR_LOGIC_FALL, scale=0.78)
                    if reason:
                        _draw_label(frame, reason, (x1, y2 + 48), COLOR_LOGIC_FALL, scale=0.52)
                else:
                    tag = f"MODEL FALL P:{st.smoothed_prob:.2f}"
                    if st.last_alert_reason:
                        reason = _short_alert_reason(st.last_alert_reason)
                        if reason:
                            tag = f"{tag} {reason}"
                    _draw_label(frame, tag, (x1, y2 + 22), COLOR_FALL, scale=0.68)

    if noid_dets:
        for bbox in noid_dets:
            if np.any(bbox):
                x1, y1, x2, y2 = np.asarray(bbox).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NOID, 1)
                _draw_label(frame, "id:?", (x1, y1), COLOR_NOID, scale=0.45)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 36), (35, 35, 35), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    hud = (f"FPS:{fps:5.1f}   active:{active_count:2d}   "
           f"infer:{infer_ms:5.1f}ms   alerts:{total_alerts}")
    cv2.putText(frame, hud, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    if any(st.alerted for st in tracks):
        has_model = any(
            (st.alerted and not _is_logic_alert(st) and not _is_trend_alert(st))
            or st.smoothed_prob >= threshold
            for st in tracks
        )
        has_trend = any(_is_trend_alert(st) for st in tracks)
        light = COLOR_FALL if has_model else (COLOR_TREND_FALL if has_trend else COLOR_LOGIC_FALL)
        cv2.circle(frame, (W - 20, 18), 8, light, -1)


def _is_trend_alert(st: TrackState) -> bool:
    reason = str(st.last_alert_reason or "")
    return bool(
        st.alerted
        and (
            reason.startswith("fall_trend")
            or reason.startswith("autopsy")
            or (
                reason.startswith("track_lost_after_fall_pose")
                and ("disappear_" in reason)
            )
        )
    )


def _is_logic_alert(st: TrackState) -> bool:
    reason = str(st.last_alert_reason or "")
    if _is_trend_alert(st):
        return False
    return bool(
        st.alerted
        and (
            reason.startswith("pose_heuristic")
            or reason.startswith("track_lost_after_fall_pose")
        )
    )


def _short_alert_reason(reason: str) -> str:
    if not reason:
        return ""
    reason = str(reason)
    for prefix in (
        "pose_heuristic:",
        "track_lost_after_fall_pose:",
        "fall_trend:",
        "autopsy:",
    ):
        reason = reason.replace(prefix, "")
    keys = []
    for part in reason.split(","):
        name = part.split("=", 1)[0].strip()
        if name:
            keys.append(name)
    return "/".join(keys[:4])


def _draw_skeleton(img, kpts, scores, color, kpt_thr):
    for j, (x, y) in enumerate(kpts):
        if scores[j] < kpt_thr:
            continue
        cv2.circle(img, (int(x), int(y)), 3, color, -1)
    for a, b in COCO_SKELETON:
        if scores[a] < kpt_thr or scores[b] < kpt_thr:
            continue
        cv2.line(img, (int(kpts[a, 0]), int(kpts[a, 1])),
                 (int(kpts[b, 0]), int(kpts[b, 1])), color, 2)


def _draw_label(img, text, org, color, scale=0.55):
    x, y = int(org[0]), int(max(org[1], 14))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================
# 帧 + 检测结果 的生成器
# ============================================================
def frame_result_generator(source, pose_model, args):
    track_kwargs = dict(
        persist=True, conf=args.conf, imgsz=args.imgsz,
        verbose=False, tracker=args.tracker,
    )
    if args.device:
        track_kwargs["device"] = args.device

    if args.frame_mode:
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise IOError(f"无法打开视频源: {source}")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose_model.track(frame, stream=False, **track_kwargs)[0]
            yield frame, res
        cap.release()
    else:
        results = pose_model.track(
            source=int(source) if str(source).isdigit() else source,
            stream=True, **track_kwargs,
        )
        for res in results:
            frame = res.orig_img
            if frame is None:
                continue
            yield frame, res


def probe_source(source):
    try:
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            return None, None, 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps <= 0 or fps > 240:
            fps = 30.0
        return W, H, fps
    except Exception:  # noqa: BLE001
        return None, None, 30.0


# ============================================================
# 主循环
# ============================================================
def run_multitarget_realtime(args):
    # 1. 模型
    pose_model = load_pose_model(args.pose_weights, args.device)
    action_model = load_action_model(args.config, args.ckpt, args.device)
    predictor = CachedClipPredictor(action_model, device=args.device)

    # 2. 探测源(必须早探测,后面要按真实 fps 决定 buffer 长度)
    src = args.source
    W0, H0, src_fps = probe_source(src)
    # source_fps:若用户给了 --source-fps 用之,否则用探测值。
    # target_fps 不覆盖真实源 fps;它只用于在未显式给 time-window 时推导训练等效窗口。
    source_fps = args.source_fps if args.source_fps > 0 else src_fps
    time_window_sec = args.time_window_sec
    if time_window_sec <= 0 and args.target_fps > 0:
        time_window_sec = args.clip_len / args.target_fps

    print(f"[源] {src}  探测尺寸={W0}x{H0}  fps≈{src_fps:.1f}  "
          f"用于 buffer 的 source_fps={source_fps:.1f}  "
          f"time_window={time_window_sec:.2f}s  "
          f"→ buffer 窗口 ≈ {max(args.clip_len, int(round(source_fps * time_window_sec)))} 帧 (clip_len={args.clip_len})  "
          f"模式={'逐帧 fallback' if args.frame_mode else 'stream track'}")

    # 3. 事件日志
    event_logger = None
    if args.event_log or args.snapshot_dir:
        event_logger = EventLogger(
            jsonl_path=args.event_log,
            snapshot_dir=args.snapshot_dir,
            repeat_sec=args.event_repeat_sec,
            fps=src_fps,
        )

    # 4. 新增组件
    track_merger = None
    if args.track_merge:
        track_merger = TrackMerger(
            iou_thr=args.track_merge_iou_thr,
            center_dist_norm_thr=args.track_merge_dist_thr,
            max_gap_frames=args.track_merge_gap,
            enabled=True,
        )

    # AlertPolicy:任一新阈值参数被显式设置 → 启用 policy
    use_policy = (args.high_thr < 1.0) or (args.topk_mean_thr < 1.0)
    alert_policy = None
    if use_policy:
        alert_policy = AlertPolicy(
            high_thr=args.high_thr,
            mid_thr=args.threshold,
            consecutive_k=args.alert_k,
            topk_window=args.topk_window,
            topk_k=args.topk_k,
            topk_mean_thr=args.topk_mean_thr,
        )

    prob_logger = None
    if args.prob_log:
        prob_logger = ProbabilityLogger(
            path=args.prob_log,
            fmt="csv" if args.prob_log.endswith(".csv") else "jsonl",
            source=str(src),
        )

    summary = None
    if args.summary_json or args.print_summary:
        summary = VideoSummaryBuilder(
            source=str(src),
            ground_truth=args.ground_truth if args.ground_truth in (0, 1) else None,
        )

    pose_heuristic = None
    if args.pose_heuristic_alert:
        pose_heuristic = PoseHeuristicScorer(
            kpt_thr=args.kpt_thr,
            min_frames=args.pose_heuristic_min_frames,
        )

    fall_trend = None
    if args.fall_trend:
        fall_trend = FallTrendDetector(
            slope_window=args.fall_trend_slope_window,
            slope_prob_thr=args.fall_trend_slope_prob_thr,
            slope_heur_thr=args.fall_trend_slope_heur_thr,
            slope_min_current=args.fall_trend_slope_min_current,
            geom_window_frames=args.fall_trend_geom_window,
            bbox_h_drop_ratio=args.fall_trend_bbox_h_drop,
            aspect_rise=args.fall_trend_aspect_rise,
            geom_track_age_min=args.fall_trend_geom_age_min,
            disappear_lookback=args.fall_trend_disappear_lookback,
            disappear_raw_min=args.fall_trend_disappear_raw_min,
            disappear_heur_min=args.fall_trend_disappear_heur_min,
            disappear_rising_tolerance=args.fall_trend_disappear_tolerance,
            autopsy_max_raw_thr=args.fall_trend_autopsy_raw_thr,
            autopsy_max_heur_thr=args.fall_trend_autopsy_heur_thr,
            autopsy_late_peak_ratio=args.fall_trend_autopsy_late_peak,
            enable_slope=not args.fall_trend_disable_slope,
            enable_geometric=not args.fall_trend_disable_geom,
            enable_disappear=not args.fall_trend_disable_disappear,
            enable_autopsy=not args.fall_trend_disable_autopsy,
        )

    pose_interpolator = None
    if args.pose_interp:
        pose_interpolator = PoseInterpolator(
            max_extrapolation_frames=args.pose_interp_max,
            score_decay=args.pose_interp_score_decay,
            velocity_window=args.pose_interp_velocity_window,
            min_history_required=args.pose_interp_min_history,
            anchor_to_kalman=not args.pose_interp_no_anchor,
        )

    # 5. 检测器
    detector = MultiTrackFallDetector(
        predictor=predictor,
        clip_len=args.clip_len,
        source_fps=source_fps,
        time_window_sec=time_window_sec,
        infer_every=args.infer_every,
        threshold=args.threshold,
        alert_k=args.alert_k,
        alert_hold_frames=int(args.alert_hold * src_fps),
        ema=args.ema,
        track_timeout=args.track_timeout,
        kpt_thr=args.kpt_thr,
        source_name=str(src),
        track_merger=track_merger,
        alert_policy=alert_policy,
        prob_logger=prob_logger,
        summary=summary,
        pose_heuristic=pose_heuristic,
        pose_heuristic_thr=args.pose_heuristic_thr,
        lost_track_alert=args.lost_track_alert,
        lost_track_min_gap=args.lost_track_min_gap,
        lost_track_heuristic_thr=args.lost_track_heuristic_thr,
        lost_track_model_thr=args.lost_track_model_thr,
        track_merge_same_frame=args.track_merge_same_frame,
        fall_trend=fall_trend,
        pose_interpolator=pose_interpolator,
        event_logger=event_logger,
    )

    # 6. 主循环
    writer = None
    fps_hist = deque(maxlen=30)
    frame_idx = 0
    err_msg = None

    print("[开始] 按 q 退出(仅窗口模式)")
    try:
        for frame, res in frame_result_generator(src, pose_model, args):
            t_loop = time.time()
            H, W = frame.shape[:2]

            kpts, scores, bboxes, track_ids = _extract_one_frame(res, max_persons=args.max_persons)
            noid_dets = [bboxes[i] for i, t in enumerate(track_ids)
                         if int(t) < 0 and np.any(bboxes[i])]

            detector.update(frame_idx, kpts, scores, bboxes, track_ids,
                            img_shape=(H, W), frame=frame)

            loop_ms = (time.time() - t_loop) * 1000
            fps_hist.append(1000.0 / max(loop_ms, 1e-6))
            cur_fps = float(np.mean(fps_hist))
            visible_tracks = detector.visible_snapshot(
                frame_idx,
                max_age=args.draw_track_max_age,
                alert_max_age=args.draw_alert_max_age,
            )
            draw_multitrack_overlay(
                frame, visible_tracks, args.threshold, args.kpt_thr,
                cur_fps, detector.last_infer_ms, detector.active_count,
                len(detector.alerted_ids), noid_dets=noid_dets,
            )

            if args.save_out:
                if writer is None:
                    Path(args.save_out).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save_out, fourcc, src_fps, (W, H))
                    print(f"[写] 输出到 {args.save_out}  ({W}x{H}@{src_fps:.1f})")
                writer.write(frame)
            if not args.no_show:
                cv2.imshow("Multi-target Fall Detection (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl-C")
    except Exception as e:  # noqa: BLE001
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[ERROR] 主循环异常: {err_msg}")
        if summary is not None:
            summary.set_error(err_msg)
    finally:
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        if event_logger is not None:
            event_logger.close()
        if prob_logger is not None:
            prob_logger.close()

    # 7. summary 构建 & 落盘
    if summary is not None:
        summary.set_frames(frame_idx)
        summary.set_merge_count(track_merger.merge_count if track_merger else 0)
        s_dict = summary.build(
            mid_thr=args.threshold,
            max_low_zone=args.max_low_zone,
            topk=args.summary_topk,
        )
        if args.summary_json:
            Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.summary_json, "w", encoding="utf-8") as f:
                json.dump(s_dict, f, indent=2, ensure_ascii=False)
            print(f"[summary] 已保存 → {args.summary_json}")
        if args.print_summary:
            print(json.dumps(s_dict, indent=2, ensure_ascii=False))

    avg_fps = float(np.mean(fps_hist)) if fps_hist else 0.0
    print("\n" + "=" * 60)
    print("  运行结束 summary")
    print("=" * 60)
    print(f"  总帧数:        {frame_idx}")
    print(f"  平均 FPS:      {avg_fps:.1f}")
    print(f"  曾报警 track:  {sorted(detector.alerted_ids) if detector.alerted_ids else '无'}")
    print(f"  报警 track 数: {len(detector.alerted_ids)}")
    if track_merger:
        print(f"  ID 合并次数:   {track_merger.merge_count}")
    if args.save_out:
        print(f"  可视化视频:    {args.save_out}")
    if args.event_log:
        print(f"  事件日志:      {args.event_log}")
    if args.prob_log:
        print(f"  概率日志:      {args.prob_log}")
    if args.summary_json:
        print(f"  视频摘要:      {args.summary_json}")
    if args.snapshot_dir:
        print(f"  报警快照目录:  {args.snapshot_dir}")
    print("=" * 60)
    if err_msg is not None:
        raise SystemExit(1)


# ============================================================
# CLI
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(description="多目标实时摔倒检测(v2:真实视频友好)")

    # 输入/模型
    p.add_argument("--source", default="0",
                   help="视频路径 / RTSP / HTTP / 摄像头编号(默认 0)")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pose-weights", default="yolo26x-pose.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tracker", default="bytetrack.yaml")

    # 时序/调度
    p.add_argument("--clip-len", type=int, default=48,
                   help="模型期望的输入帧数,等于训练 config 的 clip_len(不要乱改)")
    p.add_argument("--source-fps", type=float, default=0.0,
                   help=">0 时覆盖探测到的源 fps,仅在探测 fps 不可信时使用")
    p.add_argument("--target-fps", type=float, default=0.0,
                   help="训练等效目标 fps。若未显式给 --time-window-sec,用 clip_len/target_fps 推导窗口")
    p.add_argument("--time-window-sec", type=float, default=0.0,
                   help=">0 启用时间窗口缓冲。推荐真实 60fps 手机视频用 1.6 或 2.0")
    p.add_argument("--infer-every", type=int, default=6)
    p.add_argument("--max-persons", type=int, default=5)
    p.add_argument("--track-timeout", type=int, default=30)
    p.add_argument("--draw-track-max-age", type=int, default=8,
                   help="仅用于 overlay: 普通 track 未重新观测超过 N 帧后不再绘制")
    p.add_argument("--draw-alert-max-age", type=int, default=15,
                   help="仅用于 overlay: 报警 track 未重新观测超过 N 帧后不再绘制")

    # Track 合并(ID switch 处理)
    p.add_argument("--track-merge", action="store_true",
                   help="启用 track 合并(刚消失 + IoU/距离接近 → 继承 buffer)")
    p.add_argument("--track-merge-iou-thr", type=float, default=0.3)
    p.add_argument("--track-merge-dist-thr", type=float, default=0.15,
                   help="中心点距离归一化阈值(占图像对角线比例)")
    p.add_argument("--track-merge-gap", type=int, default=15,
                   help="允许的最大消失帧数")
    p.add_argument("--track-merge-same-frame", action="store_true",
                   help="合并同一帧中疑似同一摔倒者被拆出的重复 track")

    # 报警策略
    p.add_argument("--threshold", type=float, default=0.5,
                   help="mid 阈值(用于 consec_mid 与显示)")
    p.add_argument("--alert-k", type=int, default=2,
                   help="连续超 mid 阈值次数(consec_mid)")
    p.add_argument("--high-thr", type=float, default=1.01,
                   help="高阈值,raw_prob ≥ 此值单次即报警(默认 1.01 = 关闭)")
    p.add_argument("--topk-window", type=int, default=5)
    p.add_argument("--topk-k", type=int, default=3)
    p.add_argument("--topk-mean-thr", type=float, default=1.01,
                   help="最近 N 次推理中 top-k 平均 ≥ 此值报警(默认 1.01 = 关闭)")
    p.add_argument("--alert-hold", type=float, default=1.5)
    p.add_argument("--ema", type=float, default=0.5)
    p.add_argument("--pose-heuristic-alert", action="store_true",
                   help="启用骨架几何兜底报警,用于模型低分但姿态明显跌倒的快摔片段")
    p.add_argument("--pose-heuristic-thr", type=float, default=0.62,
                   help="骨架启发式分数达到该值时报警,仅在 --pose-heuristic-alert 时生效")
    p.add_argument("--pose-heuristic-min-frames", type=int, default=12,
                   help="启发式评分至少需要的历史骨架帧数")
    p.add_argument("--lost-track-alert", action="store_true",
                   help="启用低姿态/疑似跌倒后 track 消失的逻辑兜底报警")
    p.add_argument("--lost-track-min-gap", type=int, default=8,
                   help="track 连续消失至少 N 帧后才考虑 lost-track 兜底")
    p.add_argument("--lost-track-heuristic-thr", type=float, default=0.45,
                   help="track 消失前启发式分数达到该值时触发 lost-track 兜底")
    p.add_argument("--lost-track-model-thr", type=float, default=0.35,
                   help="track 消失前模型平滑分数达到该值时触发 lost-track 兜底")

    # FallTrendDetector — 趋势 + 几何 + 消失模式 (4 个互补策略)
    p.add_argument("--fall-trend", action="store_true",
                   help="启用 FallTrendDetector (4 个互补策略: slope/geom/disappear/autopsy)")
    p.add_argument("--fall-trend-slope-window", type=int, default=5,
                   help="策略 B: slope 回看推理次数")
    p.add_argument("--fall-trend-slope-prob-thr", type=float, default=0.05,
                   help="策略 B: raw_prob 斜率阈值 (per-推理次)")
    p.add_argument("--fall-trend-slope-heur-thr", type=float, default=0.08,
                   help="策略 B: heuristic 斜率阈值")
    p.add_argument("--fall-trend-slope-min-current", type=float, default=0.28,
                   help="策略 B: 当前值至少达此值才触发")
    p.add_argument("--fall-trend-geom-window", type=int, default=15,
                   help="策略 C: 几何形变看最近 N 帧 bbox")
    p.add_argument("--fall-trend-bbox-h-drop", type=float, default=0.35,
                   help="策略 C: bbox 高度下降比例")
    p.add_argument("--fall-trend-aspect-rise", type=float, default=0.10,
                   help="策略 C: aspect 上升绝对值")
    p.add_argument("--fall-trend-geom-age-min", type=int, default=3)
    p.add_argument("--fall-trend-disappear-lookback", type=int, default=4,
                   help="策略 A: 消失前回看推理次数")
    p.add_argument("--fall-trend-disappear-raw-min", type=float, default=0.28,
                   help="策略 A: raw 最低高位阈值")
    p.add_argument("--fall-trend-disappear-heur-min", type=float, default=0.32,
                   help="策略 A: heuristic 最低高位阈值")
    p.add_argument("--fall-trend-disappear-tolerance", type=float, default=0.03,
                   help="策略 A: 上升判定的回撤容忍")
    p.add_argument("--fall-trend-autopsy-raw-thr", type=float, default=0.30,
                   help="策略 D: track 死亡审判 raw 阈值")
    p.add_argument("--fall-trend-autopsy-heur-thr", type=float, default=0.35,
                   help="策略 D: track 死亡审判 heuristic 阈值")
    p.add_argument("--fall-trend-autopsy-late-peak", type=float, default=0.5,
                   help="策略 D: 峰值需出现在生命的后多少比例")
    p.add_argument("--fall-trend-disable-slope", action="store_true",
                   help="关闭策略 B (slope)")
    p.add_argument("--fall-trend-disable-geom", action="store_true",
                   help="关闭策略 C (geometric)")
    p.add_argument("--fall-trend-disable-disappear", action="store_true",
                   help="关闭策略 A (disappearance)")
    p.add_argument("--fall-trend-disable-autopsy", action="store_true",
                   help="关闭策略 D (autopsy)")

    # Tracking continuity: short-gap pose extrapolation.
    p.add_argument("--pose-interp", action="store_true",
                   help="短时跟丢时外推骨架并继续喂 buffer,默认关闭")
    p.add_argument("--pose-interp-max", type=int, default=8,
                   help="最大连续外推帧数")
    p.add_argument("--pose-interp-score-decay", type=float, default=0.85,
                   help="每外推一帧 keypoint score 的衰减系数")
    p.add_argument("--pose-interp-velocity-window", type=int, default=4,
                   help="用最近 N 帧估算骨架速度")
    p.add_argument("--pose-interp-min-history", type=int, default=3,
                   help="至少有 N 帧历史才允许外推")
    p.add_argument("--pose-interp-no-anchor", action="store_true",
                   help="关闭 Kalman bbox 锚定")

    # YOLO
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--kpt-thr", type=float, default=0.3)

    # 输出
    p.add_argument("--save-out", default=None)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--event-log", default=None)
    p.add_argument("--event-repeat-sec", type=float, default=0.0)
    p.add_argument("--snapshot-dir", default=None)

    # 新增:概率日志 + summary
    p.add_argument("--prob-log", default=None,
                   help="每次推理的概率日志(.jsonl 或 .csv)")
    p.add_argument("--summary-json", default=None,
                   help="视频结束时输出聚合摘要 JSON")
    p.add_argument("--print-summary", action="store_true")
    p.add_argument("--summary-topk", type=int, default=5)
    p.add_argument("--max-low-zone", type=float, default=0.3,
                   help="summary 诊断:max_pfall 低于此值判为 model_unaware")
    p.add_argument("--ground-truth", type=int, default=-1,
                   help="本视频的真实标签(0/1)。仅用于 summary 诊断,默认 -1=未知")

    # 取流
    p.add_argument("--frame-mode", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    run_multitarget_realtime(args)


if __name__ == "__main__":
    main()

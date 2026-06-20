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
COLOR_LOGIC_FALL = (220, 60, 220)
COLOR_NOID = (160, 160, 160)


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
    """
    track_id: int
    clip_len: int
    display_id: Optional[int] = None
    source_fps: float = 30.0
    time_window_sec: float = 0.0
    recent_window: int = 10                # AlertPolicy 的 top-k 滑窗用

    buffer: TimeAwareBuffer = field(default=None, repr=False)
    recent_raw_probs: deque = field(default=None, repr=False)

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

    @property
    def is_ready(self) -> bool:
        return self.buffer.is_ready

    def push(self, kpt: np.ndarray, score: np.ndarray, bbox: np.ndarray, frame_idx: int):
        self.buffer.push(kpt, score)
        self.bbox = bbox.astype(np.float32)
        self.last_kpts = kpt.astype(np.float32)
        self.last_scores = score.astype(np.float32)
        self.last_seen_frame = frame_idx
        self.frames_since_infer += 1

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
        self.alerted = bool(other.alerted)
        self.alert_frames_left = int(other.alert_frames_left)
        self.ever_alerted = bool(other.ever_alerted)
        self.last_alert_reason = str(other.last_alert_reason or "")


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
    def update(self, frame_idx, kpts, scores, bboxes, track_ids, img_shape, frame=None):
        H, W = img_shape
        img_diag = float(np.hypot(H, W))
        current_ids = {
            int(tid)
            for i, tid in enumerate(track_ids)
            if int(tid) >= 0 and np.any(kpts[i])
        }
        seen_now = set()

        # 1. 喂数据 + 尝试 track 合并
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid < 0:
                continue
            kpt = kpts[i]
            scr = scores[i]
            if not np.any(kpt):
                continue

            # 新出现的 track:尝试从 tombstones 继承
            if tid not in self.tracks:
                st = self._new_track(tid)
                if self.track_merger is not None:
                    adopted_tid = self._try_adopt_recent_inactive_track(
                        st=st,
                        new_tid=tid,
                        new_bbox=bboxes[i],
                        frame_idx=frame_idx,
                        img_diag=img_diag,
                        current_ids=current_ids,
                    )
                    if adopted_tid is None:
                        tomb = self.track_merger.try_match(
                            new_track_id=tid, new_bbox=bboxes[i],
                            current_frame=frame_idx, img_diag=img_diag,
                        )
                        if tomb is not None:
                            st.adopt(tomb)
                self.tracks[tid] = st

            self.tracks[tid].push(kpt, scr, bboxes[i], frame_idx)
            seen_now.add(tid)

        # 2. 调度推理(交错相位)
        infer_ms_accum = 0.0
        for tid, st in self.tracks.items():
            if tid not in seen_now or not st.is_ready:
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

            # EMA 平滑
            if st.infer_count == 1:
                st.smoothed_prob = raw_prob
            else:
                st.smoothed_prob = self.ema * raw_prob + (1 - self.ema) * st.smoothed_prob

            # 报警判定 + 写日志
            decision = self._update_alert(st, frame_idx, frame)

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

        # 3. 报警横幅倒计时
        for st in self.tracks.values():
            if st.alert_frames_left > 0:
                st.alert_frames_left -= 1
                if st.alert_frames_left == 0:
                    st.alerted = False
                    st.over_thr_streak = 0

        # 4. 清理过期 track(消亡时若有合并器,放入 tombstones)
        stale = [tid for tid, st in self.tracks.items()
                 if frame_idx - st.last_seen_frame > self.track_timeout]
        for tid in stale:
            st = self.tracks[tid]
            if self.track_merger is not None:
                self.track_merger.register_death(
                    track_id=tid, last_frame=st.last_seen_frame,
                    display_id=st.display_id,
                    last_bbox=st.bbox, buffer=st.buffer,
                    last_smoothed_prob=st.smoothed_prob,
                    last_raw_prob=st.last_prob,
                    over_thr_streak=st.over_thr_streak,
                )
            del self.tracks[tid]
        if self.track_merger is not None:
            self.track_merger.prune(frame_idx)

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
        else:
            # 旧逻辑:连续 K 次 smoothed > threshold
            should_alert = st.over_thr_streak >= self.alert_k and st.smoothed_prob >= self.threshold
            reason = "consec_mid" if should_alert else ""
            trig_prob = st.smoothed_prob

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

        # 状态机:首次触发 → onset;持续超阈值 → 维持 alert_frames_left
        alert_onset = False
        if should_alert:
            st.alert_frames_left = self.alert_hold_frames
            st.last_alert_reason = reason
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

    def visible_snapshot(self, frame_idx: int, max_age_frames: int = 0) -> List[TrackState]:
        """Return tracks recently observed enough to draw on the current frame."""
        max_age_frames = max(0, int(max_age_frames))
        return [
            st for st in self.tracks.values()
            if frame_idx - st.last_seen_frame <= max_age_frames
        ]

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
        logic_alert = _is_logic_alert(st)
        model_alert = st.alerted and not logic_alert
        model_score_fall = st.smoothed_prob >= threshold
        is_fall = st.alerted or model_score_fall
        if logic_alert:
            color = COLOR_LOGIC_FALL
        elif model_alert or model_score_fall:
            color = COLOR_FALL
        else:
            color = COLOR_NORMAL
        _draw_skeleton(frame, st.last_kpts, st.last_scores, color, kpt_thr)
        if st.bbox is not None and np.any(st.bbox):
            x1, y1, x2, y2 = st.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if logic_alert:
                status = "LOGIC FALL"
            elif is_fall:
                status = "MODEL FALL"
            else:
                status = "NORMAL"
            label = f"id:{st.display_id} {status} P:{st.smoothed_prob:.2f}"
            if getattr(st, "heuristic_score", 0.0) >= 0.3 or logic_alert:
                label += f" H:{st.heuristic_score:.2f}"
            _draw_label(frame, label, (x1, y1), color)
            if st.alerted:
                if logic_alert:
                    tag = f"LOGIC FALL H:{st.heuristic_score:.2f}"
                    reason = _short_logic_reason(st.last_alert_reason)
                    _draw_label(frame, tag, (x1, y2 + 24), COLOR_LOGIC_FALL, scale=0.78)
                    if reason:
                        _draw_label(frame, reason, (x1, y2 + 48), COLOR_LOGIC_FALL, scale=0.52)
                else:
                    tag = f"MODEL FALL P:{st.smoothed_prob:.2f}"
                    if st.last_alert_reason:
                        tag = f"{tag} {st.last_alert_reason}"
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
        light = COLOR_LOGIC_FALL if any(_is_logic_alert(st) for st in tracks) else COLOR_FALL
        cv2.circle(frame, (W - 20, 18), 8, light, -1)


def _is_logic_alert(st: TrackState) -> bool:
    return bool(st.alerted and str(st.last_alert_reason or "").startswith("pose_heuristic"))


def _short_logic_reason(reason: str) -> str:
    if not reason:
        return ""
    reason = str(reason).replace("pose_heuristic:", "")
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
                frame_idx, max_age_frames=args.draw_track_max_age
            )
            draw_multitrack_overlay(
                frame, visible_tracks, args.threshold, args.kpt_thr,
                cur_fps, detector.last_infer_ms, len(visible_tracks),
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
    p.add_argument("--draw-track-max-age", type=int, default=3,
                   help="只绘制最近 N 帧内被重新观测到的 track；默认 3 帧，减少短时漏检闪断并避免拼接视频长时间残留骨架")

    # Track 合并(ID switch 处理)
    p.add_argument("--track-merge", action="store_true",
                   help="启用 track 合并(刚消失 + IoU/距离接近 → 继承 buffer)")
    p.add_argument("--track-merge-iou-thr", type=float, default=0.3)
    p.add_argument("--track-merge-dist-thr", type=float, default=0.15,
                   help="中心点距离归一化阈值(占图像对角线比例)")
    p.add_argument("--track-merge-gap", type=int, default=15,
                   help="允许的最大消失帧数")

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

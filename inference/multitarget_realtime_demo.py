"""
inference/multitarget_realtime_demo.py — 多目标实时摔倒检测

在不改动训练逻辑的前提下,为现有项目新增"多人实时摔倒检测"能力:
每个 track_id 独立维护滚动缓冲区、独立分类、独立报警,支持摄像头 / 视频文件 / RTSP 流。

与单人版 (inference/realtime_demo.py) 的区别:
  - 单人版:全局一个 deque,只盯"最大框"那个人
  - 本版:  每个 track_id 一个 TrackState(独立 deque + 独立概率 + 独立报警状态机)

复用现有代码(不改旧文件):
  - extract_pose_yolo26.load_pose_model     加载 YOLO Pose
  - extract_pose_yolo26._extract_one_frame  从单帧 Result 抽关键点/框/track_id
  - batch_predict.load_action_model         加载训练好的动作识别模型
  - batch_predict.predict_clip              单 clip 推理(作为兜底)
  - pose_to_pyskl_format.build_sample       拼成 MMAction2 PoseDataset 格式

在复用之上做的两点低成本优化(详见 docs/06_multitarget_realtime_detection.md):
  1. 带缓存的 clip predictor:Compose pipeline 只构建一次,多人多次推理不重复建管线
  2. 交错推理调度:不同 track 的分类落在不同帧上触发,避免所有人在同一帧一起推理导致 FPS 抖动

=============================================================================
CLI 示例
=============================================================================
# 1) 摄像头(本机有显示器)
python inference/multitarget_realtime_demo.py \
    --source 0 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_X.pth \
    --max-persons 5

# 2) 视频文件,保存可视化结果(服务器无窗口环境)
python inference/multitarget_realtime_demo.py \
    --source test.mp4 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_X.pth \
    --save-out outputs/demo.mp4 --no-show

# 3) RTSP 流 + 事件日志
python inference/multitarget_realtime_demo.py \
    --source "rtsp://admin:pass@192.168.1.108:554/h264/ch1/main/av_stream" \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_X.pth \
    --max-persons 10 --event-log outputs/events.jsonl

# 多人实时分类比单人慢,FPS 不够时调大 --infer-every(如 8/12),或换更小的 --pose-weights
=============================================================================
"""
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

# 让 import 找到本包(与 batch_predict / realtime_demo 同样的做法)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.extract_pose_yolo26 import load_pose_model, _extract_one_frame
from inference.pose_to_pyskl_format import build_sample
from inference.batch_predict import load_action_model, predict_clip as _predict_clip_fallback


# ============================================================
# COCO 17 点骨骼连线(本地常量,避免与单人版耦合;顺序与训练严格一致)
# ============================================================
COCO_SKELETON = [
    (5, 7), (7, 9),            # 左臂
    (6, 8), (8, 10),           # 右臂
    (5, 6),                    # 肩
    (5, 11), (6, 12),          # 躯干
    (11, 12),                  # 髋
    (11, 13), (13, 15),        # 左腿
    (12, 14), (14, 16),        # 右腿
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头
    (3, 5), (4, 6),            # 耳到肩
]

# 颜色(BGR)
COLOR_NORMAL = (60, 200, 60)   # 绿:正常
COLOR_FALL = (60, 60, 240)     # 红:摔倒
COLOR_NOID = (160, 160, 160)   # 灰:无 track_id 的检测,不参与分类


# ============================================================
# 带缓存的 clip predictor —— 复用 predict_clip 的逻辑,但 pipeline 只建一次
# ============================================================
class CachedClipPredictor:
    """把 MMAction2 的 val/test pipeline 构建一次缓存起来,多次推理复用。

    原 batch_predict.predict_clip 每次调用都重建 Compose,单人离线无所谓,
    但多人实时每个 track 每次都重建会明显拖慢 FPS。这里只在初始化时构建一次。
    如果初始化失败(例如 cfg 结构异常),自动退回到 batch_predict.predict_clip。
    """

    def __init__(self, model, device="cuda:0"):
        self.model = model
        self.device = device
        self.pipeline = None
        try:
            from mmengine.dataset import Compose
            cfg = model.cfg
            if hasattr(cfg, "val_pipeline"):
                pcfg = cfg.val_pipeline
            elif hasattr(cfg, "test_pipeline"):
                pcfg = cfg.test_pipeline
            else:
                pcfg = cfg.val_dataloader.dataset.pipeline
            self.pipeline = Compose(pcfg)
        except Exception as e:  # noqa: BLE001
            print(f"[CachedClipPredictor] 构建缓存 pipeline 失败,回退到 predict_clip:{e}")
            self.pipeline = None

    @torch.no_grad()
    def __call__(self, clip_sample) -> float:
        if self.pipeline is None:
            # 兜底:用原始 predict_clip(每次自建 pipeline,但保证可用)
            return _predict_clip_fallback(self.model, clip_sample, device=self.device)

        from mmengine.dataset import pseudo_collate

        data = pseudo_collate([self.pipeline(clip_sample.copy())])
        result = self.model.test_step(data)[0]

        score = result.pred_score if hasattr(result, "pred_score") else result.get("pred_score")
        if torch.is_tensor(score):
            score = score.cpu().numpy()
        return float(score[1])  # 类别 1 = 摔倒


# ============================================================
# 单个 track 的状态(dataclass)
# ============================================================
@dataclass
class TrackState:
    """一个 track_id 的全部运行时状态。"""
    track_id: int
    clip_len: int

    # 滚动缓冲区(在 __post_init__ 里按 clip_len 建 maxlen)
    kpts: deque = field(default=None, repr=False)     # 每帧 (17, 2)
    scores: deque = field(default=None, repr=False)   # 每帧 (17,)

    # 最近一次的可视化信息
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_kpts: np.ndarray = field(default_factory=lambda: np.zeros((17, 2), dtype=np.float32))
    last_scores: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=np.float32))

    # 时序/调度
    last_seen_frame: int = 0
    frames_since_infer: int = 10 ** 9   # 初值很大 → 缓冲一满立刻先推一次
    infer_count: int = 0

    # 概率与报警
    last_prob: float = 0.0          # 最近一次原始 P(fall)
    smoothed_prob: float = 0.0      # EMA 平滑后的 P(fall),用于显示与判定
    over_thr_streak: int = 0        # 连续超阈值的推理次数(去抖)
    alerted: bool = False           # 当前是否处于报警态
    alert_frames_left: int = 0      # 报警横幅剩余帧
    ever_alerted: bool = False      # 整段视频里是否报过警(用于 summary)

    def __post_init__(self):
        if self.kpts is None:
            self.kpts = deque(maxlen=self.clip_len)
        if self.scores is None:
            self.scores = deque(maxlen=self.clip_len)

    @property
    def is_ready(self) -> bool:
        """缓冲区是否已满 clip_len 帧。"""
        return len(self.kpts) >= self.clip_len

    def push(self, kpt: np.ndarray, score: np.ndarray, bbox: np.ndarray, frame_idx: int):
        self.kpts.append(kpt.astype(np.float32))
        self.scores.append(score.astype(np.float32))
        self.bbox = bbox.astype(np.float32)
        self.last_kpts = kpt.astype(np.float32)
        self.last_scores = score.astype(np.float32)
        self.last_seen_frame = frame_idx
        self.frames_since_infer += 1


# ============================================================
# 多目标检测器:管理所有 track 的状态、调度推理、报警状态机
# ============================================================
class MultiTrackFallDetector:
    """维护 {track_id: TrackState},负责喂数据、调度分类、更新报警、清理过期 track。"""

    def __init__(
        self,
        predictor: CachedClipPredictor,
        clip_len: int = 48,
        infer_every: int = 6,
        threshold: float = 0.5,
        alert_k: int = 2,
        alert_hold_frames: int = 45,
        ema: float = 0.5,
        track_timeout: int = 30,
        kpt_thr: float = 0.3,
        source_name: str = "",
        event_logger: "Optional[EventLogger]" = None,
    ):
        self.predictor = predictor
        self.clip_len = clip_len
        self.infer_every = max(1, infer_every)
        self.threshold = threshold
        self.alert_k = max(1, alert_k)
        self.alert_hold_frames = alert_hold_frames
        self.ema = float(np.clip(ema, 0.05, 1.0))   # 1.0 = 不平滑
        self.track_timeout = track_timeout
        self.kpt_thr = kpt_thr
        self.source_name = source_name
        self.event_logger = event_logger

        self.tracks: Dict[int, TrackState] = {}
        self.alerted_ids = set()    # 整段视频里曾报警的 id(summary 用)
        self.last_infer_ms = 0.0

    # --------------------------------------------------------
    def update(self, frame_idx, kpts, scores, bboxes, track_ids, img_shape, frame=None):
        """喂入一帧的多人检测结果,更新所有 track,触发到期的推理。

        Args:
            kpts:     (M, 17, 2)
            scores:   (M, 17)
            bboxes:   (M, 4) xyxy
            track_ids:list[int],长度 M;-1 表示该检测没有 track_id
            img_shape:(H, W) 当前帧真实尺寸(用于热图)
            frame:    当前帧 BGR(仅用于报警 snapshot,可为 None)
        """
        H, W = img_shape
        seen_now = set()

        # 1. 喂数据
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid < 0:
                continue  # 无 track_id 的目标不参与分类(后面可灰框画出但不分类)
            kpt = kpts[i]
            scr = scores[i]
            # 容错:全零关键点(没检测到人却被 pad 出来的)跳过
            if not np.any(kpt):
                continue
            if tid not in self.tracks:
                self.tracks[tid] = TrackState(track_id=tid, clip_len=self.clip_len)
            self.tracks[tid].push(kpt, scr, bboxes[i], frame_idx)
            seen_now.add(tid)

        # 2. 调度推理(交错:不同 track 落在不同帧,避免一起推理)
        infer_ms_accum = 0.0
        for tid, st in self.tracks.items():
            if tid not in seen_now or not st.is_ready:
                continue
            first_time = st.infer_count == 0
            # 交错相位:每个 track 用 (tid % infer_every) 作为固定偏移
            phase_due = (frame_idx % self.infer_every) == (tid % self.infer_every)
            due = st.frames_since_infer >= self.infer_every and phase_due
            if not (first_time or due):
                continue

            t0 = time.time()
            prob = self._infer_one(st, img_shape)
            infer_ms_accum += (time.time() - t0) * 1000
            st.frames_since_infer = 0
            st.infer_count += 1

            # EMA 平滑
            if st.infer_count == 1:
                st.smoothed_prob = prob
            else:
                st.smoothed_prob = self.ema * prob + (1 - self.ema) * st.smoothed_prob
            st.last_prob = prob

            # 报警状态机(基于平滑概率)
            self._update_alert(st, frame_idx, frame)

        if infer_ms_accum > 0:
            self.last_infer_ms = infer_ms_accum

        # 3. 报警横幅倒计时(对所有 track,每帧都减)
        for st in self.tracks.values():
            if st.alert_frames_left > 0:
                st.alert_frames_left -= 1
                if st.alert_frames_left == 0:
                    st.alerted = False
                    st.over_thr_streak = 0  # 复位,允许下一次摔倒重新触发事件

        # 4. 清理长期未出现的 track
        stale = [tid for tid, st in self.tracks.items()
                 if frame_idx - st.last_seen_frame > self.track_timeout]
        for tid in stale:
            del self.tracks[tid]

    # --------------------------------------------------------
    def _infer_one(self, st: TrackState, img_shape) -> float:
        """对单个 track 的缓冲区跑一次动作分类,返回 P(fall)。出错不崩,返回上次值。"""
        try:
            # deque 里每帧是 (17,2)/(17,);堆成 (1, T, 17, 2)/(1, T, 17)
            kpts_seq = [k[None, ...] for k in st.kpts]      # 每帧 (1,17,2)
            scrs_seq = [s[None, ...] for s in st.scores]    # 每帧 (1,17)
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
    def _update_alert(self, st: TrackState, frame_idx, frame):
        """去抖报警状态机:连续 alert_k 次超阈值才正式报警。"""
        if st.smoothed_prob > self.threshold:
            st.over_thr_streak += 1
            st.alert_frames_left = self.alert_hold_frames  # 维持横幅
            if st.over_thr_streak >= self.alert_k and not st.alerted:
                # 报警首次触发(onset)
                st.alerted = True
                st.ever_alerted = True
                self.alerted_ids.add(st.track_id)
                if self.event_logger is not None:
                    self.event_logger.log(
                        frame_idx=frame_idx,
                        track_id=st.track_id,
                        fall_prob=st.smoothed_prob,
                        bbox=st.bbox,
                        source=self.source_name,
                        event="onset",
                        frame=frame,
                    )
        else:
            st.over_thr_streak = 0

    # --------------------------------------------------------
    def snapshot(self) -> List[TrackState]:
        """返回当前所有 track 状态的浅快照(供绘制)。"""
        return list(self.tracks.values())

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

    def log(self, frame_idx, track_id, fall_prob, bbox, source, event="onset", frame=None):
        # 持续报警限频
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
        }

        # 可选 snapshot
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
        # 同时在控制台提示一声
        print(f"[ALERT] frame={frame_idx} track={track_id} P(fall)={rec['fall_prob']} ({event})")

    def close(self):
        if self._fh:
            self._fh.close()


# ============================================================
# 可视化叠加
# ============================================================
def draw_multitrack_overlay(frame, tracks: List[TrackState], threshold, kpt_thr,
                            fps, infer_ms, active_count, total_alerts,
                            noid_dets=None):
    """在 frame 上画每个 track 的骨骼/框/概率,以及全局 HUD。"""
    H, W = frame.shape[:2]

    # --- 每个 track ---
    for st in tracks:
        is_fall = st.alerted or st.smoothed_prob > threshold
        color = COLOR_FALL if is_fall else COLOR_NORMAL

        # 骨骼
        _draw_skeleton(frame, st.last_kpts, st.last_scores, color, kpt_thr)

        # bbox + 标签
        if st.bbox is not None and np.any(st.bbox):
            x1, y1, x2, y2 = st.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            status = "FALL" if is_fall else "NORMAL"
            label = f"id:{st.track_id} {status} P(fall):{st.smoothed_prob:.2f}"
            _draw_label(frame, label, (x1, y1), color)
            if st.alerted:
                _draw_label(frame, "FALL", (x1, y2 + 18), COLOR_FALL, scale=0.6)

    # --- 无 track_id 的检测(灰框,不分类) ---
    if noid_dets:
        for bbox in noid_dets:
            if np.any(bbox):
                x1, y1, x2, y2 = np.asarray(bbox).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NOID, 1)
                _draw_label(frame, "id:?", (x1, y1), COLOR_NOID, scale=0.45)

    # --- 顶部全局 HUD ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 36), (35, 35, 35), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    hud = (f"FPS:{fps:5.1f}   active:{active_count:2d}   "
           f"infer:{infer_ms:5.1f}ms   alerts:{total_alerts}")
    cv2.putText(frame, hud, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    # 有任何人报警时,右上角加一个红点提示
    if any(st.alerted for st in tracks):
        cv2.circle(frame, (W - 20, 18), 8, COLOR_FALL, -1)


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
# 帧 + 检测结果 的统一生成器(stream 模式 / 逐帧 fallback)
# ============================================================
def frame_result_generator(source, pose_model, args):
    """产出 (frame_bgr, ultralytics_result)。

    默认走 model.track(stream=True, persist=True),保持连续 track_id;
    --frame-mode 时改为 cv2 逐帧读取 + 每帧 track(persist=True)。

    为什么提供逐帧 fallback:
      某些 RTSP/编码组合下,ultralytics 内部的 stream 取流循环会因网络抖动卡住或缓冲堆积;
      cv2.VideoCapture 逐帧模式便于自己控制 CAP_PROP_BUFFERSIZE、超时与重连,
      在不稳定网络源上更可控(代价是略慢)。
    """
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
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 降低延迟(部分后端支持)
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
    """探测视频源的 (W, H, fps)。失败则返回 (None, None, 30.0),不阻塞主流程。"""
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
    # 1. 模型(不在 import 阶段加载;这里才加载)
    pose_model = load_pose_model(args.pose_weights, args.device)
    action_model = load_action_model(args.config, args.ckpt, args.device)
    predictor = CachedClipPredictor(action_model, device=args.device)

    # 2. 探测源信息(拿 fps 给 writer / 事件限频)
    src = args.source
    W0, H0, fps = probe_source(src)
    print(f"[源] {src}  探测尺寸={W0}x{H0}  fps≈{fps:.1f}  "
          f"模式={'逐帧 fallback' if args.frame_mode else 'stream track'}")

    # 3. 事件日志
    event_logger = None
    if args.event_log or args.snapshot_dir:
        event_logger = EventLogger(
            jsonl_path=args.event_log,
            snapshot_dir=args.snapshot_dir,
            repeat_sec=args.event_repeat_sec,
            fps=fps,
        )

    # 4. 检测器
    detector = MultiTrackFallDetector(
        predictor=predictor,
        clip_len=args.clip_len,
        infer_every=args.infer_every,
        threshold=args.threshold,
        alert_k=args.alert_k,
        alert_hold_frames=int(args.alert_hold * fps),
        ema=args.ema,
        track_timeout=args.track_timeout,
        kpt_thr=args.kpt_thr,
        source_name=str(src),
        event_logger=event_logger,
    )

    # 5. writer 延迟到第一帧再建(用真实帧尺寸)
    writer = None
    fps_hist = deque(maxlen=30)
    frame_idx = 0

    print("[开始] 按 q 退出(仅窗口模式)")
    try:
        for frame, res in frame_result_generator(src, pose_model, args):
            t_loop = time.time()
            H, W = frame.shape[:2]

            # 5.1 抽该帧多人结果(复用现有函数;max_persons 控制最多人数)
            kpts, scores, bboxes, track_ids = _extract_one_frame(res, max_persons=args.max_persons)

            # 收集"无 id"的检测框(灰框画出,不分类)
            noid_dets = [bboxes[i] for i, t in enumerate(track_ids)
                         if int(t) < 0 and np.any(bboxes[i])]

            # 5.2 更新所有 track + 调度推理
            detector.update(frame_idx, kpts, scores, bboxes, track_ids,
                            img_shape=(H, W), frame=frame)

            # 5.3 画
            loop_ms = (time.time() - t_loop) * 1000
            fps_hist.append(1000.0 / max(loop_ms, 1e-6))
            cur_fps = float(np.mean(fps_hist))
            draw_multitrack_overlay(
                frame, detector.snapshot(), args.threshold, args.kpt_thr,
                cur_fps, detector.last_infer_ms, detector.active_count,
                len(detector.alerted_ids), noid_dets=noid_dets,
            )

            # 5.4 输出
            if args.save_out:
                if writer is None:
                    Path(args.save_out).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save_out, fourcc, fps, (W, H))
                    print(f"[写] 输出到 {args.save_out}  ({W}x{H}@{fps:.1f})")
                writer.write(frame)
            if not args.no_show:
                cv2.imshow("Multi-target Fall Detection (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl-C")
    finally:
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        if event_logger is not None:
            event_logger.close()

    # 6. summary
    avg_fps = float(np.mean(fps_hist)) if fps_hist else 0.0
    print("\n" + "=" * 50)
    print("  运行结束 summary")
    print("=" * 50)
    print(f"  总帧数:        {frame_idx}")
    print(f"  平均 FPS:      {avg_fps:.1f}")
    print(f"  曾报警 track:  {sorted(detector.alerted_ids) if detector.alerted_ids else '无'}")
    print(f"  报警 track 数: {len(detector.alerted_ids)}")
    if args.save_out:
        print(f"  可视化视频:    {args.save_out}")
    if args.event_log:
        print(f"  事件日志:      {args.event_log}")
    if args.snapshot_dir:
        print(f"  报警快照目录:  {args.snapshot_dir}")
    print("=" * 50)


# ============================================================
# CLI
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(description="多目标实时摔倒检测")
    # 输入/模型
    p.add_argument("--source", default="0",
                   help="视频路径 / RTSP / HTTP / 摄像头编号(默认 0)")
    p.add_argument("--config", required=True, help="MMAction2 config")
    p.add_argument("--ckpt", required=True, help="训练好的 .pth")
    p.add_argument("--pose-weights", default="yolo26x-pose.pt",
                   help="YOLO Pose 权重(FPS 不够可换 yolo26m/s-pose.pt)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="ultralytics 跟踪器配置")

    # 时序/调度
    p.add_argument("--clip-len", type=int, default=48,
                   help="每个 track 的滚动缓冲长度,必须等于训练 config 的 clip_len")
    p.add_argument("--infer-every", type=int, default=6,
                   help="每个 track 每 N 帧分类一次(多人调大可提 FPS)")
    p.add_argument("--max-persons", type=int, default=5, help="每帧最多处理人数")
    p.add_argument("--track-timeout", type=int, default=30,
                   help="track 超过这么多帧未出现就清理")

    # 概率/报警
    p.add_argument("--threshold", type=float, default=0.5, help="摔倒报警阈值")
    p.add_argument("--alert-k", type=int, default=2,
                   help="连续超阈值多少次推理才正式报警(去抖)")
    p.add_argument("--alert-hold", type=float, default=1.5, help="报警横幅保持秒数")
    p.add_argument("--ema", type=float, default=0.5,
                   help="概率 EMA 平滑系数(1.0=不平滑,越小越平滑)")

    # YOLO
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 人体框置信度")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--kpt-thr", type=float, default=0.3, help="画骨骼的关键点置信度阈值")

    # 输出
    p.add_argument("--save-out", default=None, help="保存可视化结果 mp4")
    p.add_argument("--no-show", action="store_true", help="不开窗口(服务器必加)")
    p.add_argument("--event-log", default=None, help="事件日志 JSONL 路径")
    p.add_argument("--event-repeat-sec", type=float, default=0.0,
                   help=">0 时持续报警每隔这么多秒补记一条 ongoing 事件(默认只记 onset)")
    p.add_argument("--snapshot-dir", default=None,
                   help="报警瞬间保存帧图到该目录(配合事件日志做演示证据)")

    # 取流
    p.add_argument("--frame-mode", action="store_true",
                   help="改用 cv2 逐帧读取 + 每帧 track(RTSP 不稳时更可控,略慢)")
    return p


def main():
    args = build_argparser().parse_args()
    run_multitarget_realtime(args)


if __name__ == "__main__":
    main()

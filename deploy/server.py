"""
deploy/server.py — WebSocket 推理服务端

启动:
    python -m deploy.server \
        --host 0.0.0.0 --port 8765 \
        --config configs/posec3d_fall_binary.py \
        --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth

设计:
  - 每个 WebSocket 连接 = 1 个 FallDetectionSession
  - 全局共享 1 个 PoseConv3D 模型 (避免重复加载浪费显存)
  - session 内独立维护 MultiTrackFallDetector (track 状态隔离)
  - 接收骨架 JSON → 喂 detector.update() → 回 result JSON
  - 报警事件通过 pending queue 异步推回客户端

复用现有组件:
  - inference.realtime_core.* — TrackMerger / AlertPolicy / PoseHeuristicScorer / FallTrendDetector
  - inference.multitarget_realtime_demo.MultiTrackFallDetector / CachedClipPredictor
  - inference.batch_predict.load_action_model
  - inference.pose_to_pyskl_format.build_sample
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

# 让 deploy/ 能直接被脚本调用 (python -m deploy.server)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("[FATAL] 需要安装 websockets: pip install websockets", file=sys.stderr)
    raise

from deploy.protocol import (
    MSG_HELLO, MSG_HELLO_ACK, MSG_FRAME, MSG_RESULT,
    MSG_ALERT, MSG_PING, MSG_PONG, MSG_BYE, MSG_ERROR,
    SERVER_VERSION, MAX_MSG_SIZE, DEFAULT_PORT,
    build_error_msg,
)

# 复用现有组件
from inference.realtime_core import (
    TrackMerger, AlertPolicy, PoseHeuristicScorer,
    FallTrendDetector,
)
from inference.multitarget_realtime_demo import (
    MultiTrackFallDetector, CachedClipPredictor,
)
from inference.batch_predict import load_action_model

logger = logging.getLogger("server")


# ============================================================
# 全局共享:模型加载一次,所有 session 复用
# ============================================================
_GLOBAL_PREDICTOR: Optional[CachedClipPredictor] = None


def init_global_predictor(config_path: str, ckpt_path: str, device: str = "cuda:0"):
    """启动时加载一次模型,所有 session 共享。"""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        logger.info(f"[init] loading action model: {config_path} + {ckpt_path}")
        t0 = time.time()
        action_model = load_action_model(config_path, ckpt_path, device)
        _GLOBAL_PREDICTOR = CachedClipPredictor(action_model, device=device)
        logger.info(f"[init] model loaded in {time.time()-t0:.1f}s")
    return _GLOBAL_PREDICTOR


# ============================================================
# 适配:把 detector 的同步 event_logger 接口转为 WebSocket 推送
# ============================================================
class SessionEventLogger:
    """将 detector 触发的事件入队,主循环消化后通过 WebSocket 推送。"""

    def __init__(self, session: "FallDetectionSession"):
        self.session = session

    def log(self, frame_idx, track_id, fall_prob, bbox, source,
            event, reason, frame=None, **kwargs):
        # detector 是同步代码,这里只能用同步 append (list.append 线程安全)
        bbox_list = [float(v) for v in (bbox if bbox is not None else [0, 0, 0, 0])]
        self.session.pending_alerts.append({
            "type": MSG_ALERT,
            "frame_idx": int(frame_idx),
            "track_id": int(track_id),
            "display_id": int(track_id),
            "fall_prob": float(fall_prob),
            "bbox": bbox_list,
            "reason": str(reason or ""),
            "event": str(event or "onset"),
        })


# ============================================================
# Session
# ============================================================
class FallDetectionSession:
    """单个客户端的会话状态。

    生命周期:
      __init__ → 收到 frame → process_frame → ... → close
    """

    def __init__(self, session_id: str, predictor: CachedClipPredictor, args):
        self.session_id = session_id
        self.predictor = predictor
        self.args = args
        self.last_seen = time.time()
        self.client_fps = 30.0  # 客户端 hello 时上报,这里只是默认

        # 报警事件队列 (detector 同步 → 主循环异步推送)
        self.pending_alerts: list = []

        # 构造 detector
        self.detector = self._build_detector()

        # 统计
        self.n_frames_received = 0
        self.n_inferences = 0
        self.t_start = time.time()

    def configure_from_hello(self, hello_msg: dict):
        """从客户端 hello 中拿到 fps 等参数,可能需要重建 detector。"""
        new_fps = float(hello_msg.get("source_fps", 30.0))
        if abs(new_fps - self.client_fps) > 1.0:
            self.client_fps = new_fps
            self.detector = self._build_detector()
            logger.info(f"[session {self.session_id}] rebuilt detector for fps={new_fps}")

    def _build_detector(self) -> MultiTrackFallDetector:
        args = self.args

        track_merger = None
        if args.track_merge:
            track_merger = TrackMerger(
                iou_thr=args.track_merge_iou_thr,
                center_dist_norm_thr=args.track_merge_dist_thr,
                max_gap_frames=args.track_merge_gap,
            )

        alert_policy = None
        if args.high_thr < 1.0 or args.topk_mean_thr < 1.0:
            alert_policy = AlertPolicy(
                high_thr=args.high_thr,
                mid_thr=args.threshold,
                consecutive_k=args.alert_k,
                topk_window=args.topk_window,
                topk_k=args.topk_k,
                topk_mean_thr=args.topk_mean_thr,
            )

        pose_heuristic = None
        if args.pose_heuristic_alert:
            pose_heuristic = PoseHeuristicScorer(
                kpt_thr=args.kpt_thr,
                min_frames=args.pose_heuristic_min_frames,
            )

        fall_trend = FallTrendDetector() if args.fall_trend else None

        event_logger = SessionEventLogger(self)

        return MultiTrackFallDetector(
            predictor=self.predictor,
            clip_len=args.clip_len,
            source_fps=self.client_fps,
            time_window_sec=args.time_window_sec,
            infer_every=args.infer_every,
            threshold=args.threshold,
            alert_k=args.alert_k,
            alert_hold_frames=int(args.alert_hold * self.client_fps),
            ema=args.ema,
            track_timeout=args.track_timeout,
            kpt_thr=args.kpt_thr,
            source_name=f"ws_session_{self.session_id}",
            track_merger=track_merger,
            alert_policy=alert_policy,
            pose_heuristic=pose_heuristic,
            pose_heuristic_thr=args.pose_heuristic_thr,
            lost_track_alert=args.lost_track_alert,
            lost_track_min_gap=args.lost_track_min_gap,
            lost_track_heuristic_thr=args.lost_track_heuristic_thr,
            lost_track_model_thr=args.lost_track_model_thr,
            track_merge_same_frame=args.track_merge_same_frame,
            fall_trend=fall_trend,
            event_logger=event_logger,
        )

    def process_frame(self, msg: dict) -> dict:
        """同步处理一帧消息,返回 result dict。"""
        self.last_seen = time.time()
        self.n_frames_received += 1

        frame_idx = int(msg["frame_idx"])
        frame_h = int(msg["frame_h"])
        frame_w = int(msg["frame_w"])
        persons = msg.get("persons", [])

        # 转 detector.update() 期望的格式
        if persons:
            kpts_list = []
            scores_list = []
            bboxes_list = []
            track_ids = []
            for p in persons:
                kpts_arr = np.asarray(p["keypoints"], dtype=np.float32)  # (17, 3)
                if kpts_arr.shape != (17, 3):
                    continue
                kpts_list.append(kpts_arr[:, :2])
                scores_list.append(kpts_arr[:, 2])
                bboxes_list.append(np.asarray(p["bbox"], dtype=np.float32))
                track_ids.append(int(p["track_id"]))

            kpts = np.stack(kpts_list) if kpts_list else np.zeros((0, 17, 2), dtype=np.float32)
            scores = np.stack(scores_list) if scores_list else np.zeros((0, 17), dtype=np.float32)
            bboxes = np.stack(bboxes_list) if bboxes_list else np.zeros((0, 4), dtype=np.float32)
        else:
            kpts = np.zeros((0, 17, 2), dtype=np.float32)
            scores = np.zeros((0, 17), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
            track_ids = []

        # 喂入 detector
        self.detector.update(
            frame_idx, kpts, scores, bboxes, track_ids,
            img_shape=(frame_h, frame_w),
            frame=None,  # 不需要原始帧,客户端自己有
        )

        # 提取当前所有 track 的状态
        tracks_state = []
        for tid, st in self.detector.tracks.items():
            tracks_state.append({
                "track_id": int(tid),
                "display_id": int(st.display_id),
                "fall_prob": float(st.last_prob),
                "smoothed_prob": float(st.smoothed_prob),
                "heuristic_score": float(st.heuristic_score),
                "alerted": bool(st.alerted),
                "ever_alerted": bool(st.ever_alerted),
                "alert_reason": str(st.last_alert_reason or ""),
                "bbox": [float(v) for v in st.bbox],
                "is_ready": bool(st.is_ready),
            })

        return {
            "type": MSG_RESULT,
            "frame_idx": frame_idx,
            "tracks": tracks_state,
        }

    def drain_pending_alerts(self) -> list:
        """取出所有积压的报警事件 (主循环每次发送 result 后调用)。"""
        out = list(self.pending_alerts)
        self.pending_alerts.clear()
        return out

    def stats(self) -> dict:
        elapsed = max(time.time() - self.t_start, 1e-3)
        return {
            "session_id": self.session_id,
            "n_frames": self.n_frames_received,
            "effective_fps": self.n_frames_received / elapsed,
            "n_tracks": len(self.detector.tracks),
            "n_alerted": len(self.detector.alerted_ids),
            "uptime_s": elapsed,
        }


# ============================================================
# WebSocket handler
# ============================================================
async def handle_client(websocket: WebSocketServerProtocol, args):
    session_id = uuid.uuid4().hex[:8]
    peer = websocket.remote_address
    logger.info(f"[session {session_id}] new connection from {peer}")

    session = FallDetectionSession(session_id, _GLOBAL_PREDICTOR, args)

    try:
        # 第 1 步:等待 hello
        first_raw = await asyncio.wait_for(websocket.recv(), timeout=15.0)
        try:
            first_msg = json.loads(first_raw)
        except json.JSONDecodeError:
            await websocket.send(json.dumps(build_error_msg("BAD_JSON", "first message must be JSON")))
            return
        if first_msg.get("type") != MSG_HELLO:
            await websocket.send(json.dumps(build_error_msg("BAD_HELLO", "expect 'hello' first")))
            return

        session.configure_from_hello(first_msg)
        await websocket.send(json.dumps({
            "type": MSG_HELLO_ACK,
            "session_id": session_id,
            "server_version": SERVER_VERSION,
            "config_summary": {
                "clip_len": args.clip_len,
                "time_window_sec": args.time_window_sec,
                "threshold": args.threshold,
                "fall_trend_enabled": args.fall_trend,
                "pose_heuristic_enabled": args.pose_heuristic_alert,
                "lost_track_alert_enabled": args.lost_track_alert,
            },
        }))
        logger.info(f"[session {session_id}] hello ok, client_fps={session.client_fps:.1f}, "
                    f"frame={first_msg.get('frame_w')}x{first_msg.get('frame_h')}")

        # 第 2 步:主循环
        last_stats_log = time.time()
        async for raw_msg in websocket:
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                logger.warning(f"[session {session_id}] invalid JSON, skip")
                continue

            msg_type = msg.get("type", "")

            if msg_type == MSG_FRAME:
                # 在线程池跑 detector (避免阻塞事件循环 — predictor 调用是阻塞的)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, session.process_frame, msg)
                await websocket.send(json.dumps(result))

                # 推送 detector 在 update() 中产生的 alert (异步事件)
                alerts = session.drain_pending_alerts()
                for alert in alerts:
                    await websocket.send(json.dumps(alert))

                # 定期打 stats
                if time.time() - last_stats_log > 30:
                    s = session.stats()
                    logger.info(f"[session {session_id}] stats: frames={s['n_frames']} "
                                f"eff_fps={s['effective_fps']:.1f} tracks={s['n_tracks']} "
                                f"alerted={s['n_alerted']}")
                    last_stats_log = time.time()

            elif msg_type == MSG_PING:
                await websocket.send(json.dumps({"type": MSG_PONG}))

            elif msg_type == MSG_BYE:
                logger.info(f"[session {session_id}] client said bye")
                break

            else:
                logger.warning(f"[session {session_id}] unknown msg type: {msg_type}")

    except asyncio.TimeoutError:
        logger.info(f"[session {session_id}] hello timeout")
    except websockets.ConnectionClosed as e:
        logger.info(f"[session {session_id}] connection closed: {e.code} {e.reason}")
    except Exception as e:
        logger.exception(f"[session {session_id}] error: {e}")
        try:
            await websocket.send(json.dumps(build_error_msg("INTERNAL", str(e))))
        except Exception:
            pass
    finally:
        s = session.stats()
        logger.info(f"[session {session_id}] closed. final: {s}")


# ============================================================
# CLI
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(
        description="Fall detection WebSocket server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 网络
    p.add_argument("--host", default="0.0.0.0", help="监听地址")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口")

    # 模型
    p.add_argument("--config", required=True, help="MMAction2 配置文件路径")
    p.add_argument("--ckpt", required=True, help="checkpoint 路径")
    p.add_argument("--device", default="cuda:0", help="模型设备")

    # detector 核心参数 (与 multitarget_realtime_demo.py 对齐)
    p.add_argument("--clip-len", type=int, default=48)
    p.add_argument("--time-window-sec", type=float, default=1.6)
    p.add_argument("--infer-every", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--alert-k", type=int, default=2)
    p.add_argument("--alert-hold", type=float, default=1.5)
    p.add_argument("--ema", type=float, default=0.5)
    p.add_argument("--track-timeout", type=int, default=120)
    p.add_argument("--kpt-thr", type=float, default=0.3)

    # 多策略报警
    p.add_argument("--high-thr", type=float, default=0.7)
    p.add_argument("--topk-window", type=int, default=5)
    p.add_argument("--topk-k", type=int, default=3)
    p.add_argument("--topk-mean-thr", type=float, default=0.5)

    # track 合并
    p.add_argument("--track-merge", action="store_true", default=True)
    p.add_argument("--no-track-merge", dest="track_merge", action="store_false")
    p.add_argument("--track-merge-iou-thr", type=float, default=0.3)
    p.add_argument("--track-merge-dist-thr", type=float, default=0.15)
    p.add_argument("--track-merge-gap", type=int, default=45)
    p.add_argument("--track-merge-same-frame", action="store_true", default=True)

    # 姿态启发式
    p.add_argument("--pose-heuristic-alert", action="store_true", default=True)
    p.add_argument("--no-pose-heuristic-alert", dest="pose_heuristic_alert", action="store_false")
    p.add_argument("--pose-heuristic-thr", type=float, default=0.62)
    p.add_argument("--pose-heuristic-min-frames", type=int, default=12)

    # lost_track
    p.add_argument("--lost-track-alert", action="store_true", default=True)
    p.add_argument("--no-lost-track-alert", dest="lost_track_alert", action="store_false")
    p.add_argument("--lost-track-min-gap", type=int, default=8)
    p.add_argument("--lost-track-heuristic-thr", type=float, default=0.45)
    p.add_argument("--lost-track-model-thr", type=float, default=0.35)

    # FallTrendDetector
    p.add_argument("--fall-trend", action="store_true", default=True)
    p.add_argument("--no-fall-trend", dest="fall_trend", action="store_false")

    # 日志
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


async def main_async():
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 加载模型 (阻塞,只做一次)
    init_global_predictor(args.config, args.ckpt, args.device)

    logger.info(f"Listening on ws://{args.host}:{args.port}")
    logger.info(f"Server version: {SERVER_VERSION}")

    # 启动 WebSocket 服务器
    async def handler(websocket):
        # websockets v11+ 的 handler 签名只接收 websocket
        await handle_client(websocket, args)

    async with websockets.serve(
        handler, args.host, args.port,
        max_size=MAX_MSG_SIZE,
        ping_interval=30, ping_timeout=20,
    ):
        await asyncio.Future()  # run forever


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("shutdown")


if __name__ == "__main__":
    main()

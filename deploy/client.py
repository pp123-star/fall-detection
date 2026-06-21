"""
deploy/client.py — 本地摄像头客户端

本地采集摄像头 → YOLO Pose (CPU/GPU 均可) → 上传骨架到服务器 → 接收结果并显示。

启动:
    python -m deploy.client \
        --server ws://your-server-host:8765 \
        --pose-weights yolo11s-pose.pt \
        --camera 0

带宽:
  上行 30fps × 5 人骨架 ≈ 0.3 Mbps
  下行 JSON 结果 ≈ 0.01 Mbps
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# 让脚本可独立运行 (python -m deploy.client)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2
except ImportError:
    print("[FATAL] 需要 OpenCV: pip install opencv-python", file=sys.stderr)
    raise

try:
    import websockets
except ImportError:
    print("[FATAL] 需要 websockets: pip install websockets", file=sys.stderr)
    raise

from deploy.protocol import (
    MSG_HELLO_ACK, MSG_RESULT, MSG_ALERT, MSG_PONG, MSG_BYE, MSG_PING,
    MAX_MSG_SIZE,
    build_hello_msg, build_frame_msg,
)

logger = logging.getLogger("client")


# ============================================================
# COCO 17 骨架连线 (与服务端 demo 保持一致)
# ============================================================
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]


# ============================================================
# YOLO Pose 封装
# ============================================================
class LocalPoseExtractor:
    """本地 YOLO Pose,提取每帧的多人骨架 + track id。"""

    def __init__(self, weights: str, device: str = "cpu",
                 conf: float = 0.15, imgsz: int = 960):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "需要 ultralytics: pip install ultralytics"
            ) from e
        logger.info(f"Loading YOLO Pose: {weights} on {device}")
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.max_persons = 5

    def __call__(self, frame: np.ndarray) -> list:
        """返回 List[person dict]。"""
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[0],            # 仅 person
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()        # (N, 4)
        confs = r.boxes.conf.cpu().numpy()        # (N,)
        if r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            # 没 track id 时给负数表示
            track_ids = -1 * (np.arange(len(boxes)) + 1)

        kpts_xy = r.keypoints.xy.cpu().numpy()    # (N, 17, 2)
        if r.keypoints.conf is not None:
            kpts_conf = r.keypoints.conf.cpu().numpy()  # (N, 17)
        else:
            kpts_conf = np.ones((len(boxes), 17), dtype=np.float32)

        order = np.argsort(-confs)[:self.max_persons]
        persons = []
        for i in order:
            kpts_list = []
            for j in range(17):
                kpts_list.append([
                    float(kpts_xy[i, j, 0]),
                    float(kpts_xy[i, j, 1]),
                    float(kpts_conf[i, j]),
                ])
            persons.append({
                "track_id": int(track_ids[i]),
                "bbox": [float(v) for v in boxes[i]],
                "keypoints": kpts_list,
            })
        return persons


# ============================================================
# 渲染:在帧上叠加骨架、bbox、结果
# ============================================================
def draw_overlay(frame: np.ndarray, persons: list,
                 server_state: dict, recent_alerts: deque,
                 fps_eff: float = 0.0, latency_ms: float = 0.0,
                 kpt_thr: float = 0.3) -> np.ndarray:
    """在 frame 上画 bbox + skeleton + label + 报警 overlay。

    server_state: dict 形如 {track_id: latest result from server}
    """
    H, W = frame.shape[:2]
    has_alert = any(s.get("alerted") for s in server_state.values())

    # 画每个人
    for p in persons:
        tid = int(p["track_id"])
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        kpts = p["keypoints"]

        # 从服务端结果查状态
        st = server_state.get(tid, {})
        alerted = bool(st.get("alerted"))
        ever_alerted = bool(st.get("ever_alerted"))
        fall_prob = float(st.get("fall_prob", 0.0))
        smoothed = float(st.get("smoothed_prob", 0.0))
        heur = float(st.get("heuristic_score", 0.0))
        display_id = int(st.get("display_id", tid))
        is_ready = bool(st.get("is_ready", False))

        if alerted:
            color = (0, 0, 255); thickness = 4
        elif ever_alerted:
            color = (0, 120, 220); thickness = 3
        elif is_ready:
            color = (0, 200, 0); thickness = 2
        else:
            color = (180, 180, 180); thickness = 1  # buffer 未满

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        status = "FALL!" if alerted else ("WARNING" if ever_alerted else "NORMAL")
        label1 = f"id:{display_id} {status}"
        label2 = f"P:{fall_prob:.2f} S:{smoothed:.2f} H:{heur:.2f}"

        y_text = max(y1 - 28, 0)
        cv2.rectangle(frame, (x1, y_text), (x1 + 230, y_text + 26),
                      (0, 0, 0), -1)
        cv2.putText(frame, label1, (x1 + 4, y_text + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(frame, label2, (x1 + 4, y_text + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # 骨架
        for a, b in COCO_SKELETON:
            if a < len(kpts) and b < len(kpts):
                if kpts[a][2] >= kpt_thr and kpts[b][2] >= kpt_thr:
                    cv2.line(frame,
                             (int(kpts[a][0]), int(kpts[a][1])),
                             (int(kpts[b][0]), int(kpts[b][1])),
                             color, 2)
        for kx, ky, kc in kpts:
            if kc >= kpt_thr:
                cv2.circle(frame, (int(kx), int(ky)), 3, color, -1)

    # 顶部状态条
    top_bar_color = (0, 0, 200) if has_alert else (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (W, 50), top_bar_color, -1)
    if has_alert:
        cv2.putText(frame, "FALL DETECTED!", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Fall Detection (Distributed Mode)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)

    # 右上角:延迟 + FPS
    stats_text = f"FPS: {fps_eff:.1f}   RTT: {latency_ms:.0f}ms"
    text_size, _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, stats_text, (W - text_size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # 底部:最近 3 条报警
    if recent_alerts:
        y0 = H - 75
        cv2.rectangle(frame, (0, y0), (W, H), (20, 20, 20), -1)
        cv2.putText(frame, "RECENT ALERTS:", (10, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
        for i, alert in enumerate(list(recent_alerts)[-3:]):
            reason = str(alert.get("reason", ""))[:80]
            text = f"  [frame {alert.get('frame_idx', '?')}] id={alert.get('track_id', '?')}: {reason}"
            cv2.putText(frame, text, (10, y0 + 32 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


# ============================================================
# 主循环
# ============================================================
async def client_loop(args):
    pose = LocalPoseExtractor(
        weights=args.pose_weights,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera ready: {actual_w}x{actual_h} @ {src_fps:.1f} fps")

    uri = args.server
    logger.info(f"Connecting to {uri}")

    writer = None
    if args.save_out:
        Path(args.save_out).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_out, fourcc, src_fps, (actual_w, actual_h))
        logger.info(f"Saving annotated stream to {args.save_out}")

    try:
        async with websockets.connect(
            uri,
            max_size=MAX_MSG_SIZE,
            ping_interval=30, ping_timeout=20,
            open_timeout=15,
        ) as ws:
            # hello
            await ws.send(json.dumps(build_hello_msg(
                frame_h=actual_h, frame_w=actual_w,
                source_fps=src_fps, client_id="webcam",
            )))
            ack_raw = await asyncio.wait_for(ws.recv(), timeout=15)
            ack = json.loads(ack_raw)
            if ack.get("type") != MSG_HELLO_ACK:
                logger.error(f"Bad hello_ack: {ack}")
                return
            session_id = ack.get("session_id")
            logger.info(f"Connected. session_id={session_id} "
                        f"server={ack.get('server_version')}")
            logger.info(f"Server config: {ack.get('config_summary')}")

            # 共享状态 (主循环写 / recv 任务读)
            server_state = {}                  # track_id -> latest result
            recent_alerts = deque(maxlen=20)
            send_timestamps = {}                # frame_idx -> send time (用于估算 RTT)
            latest_latency_ms = 0.0
            connection_alive = asyncio.Event()
            connection_alive.set()

            # 接收任务
            async def recv_loop():
                nonlocal latest_latency_ms
                try:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        mt = msg.get("type")
                        if mt == MSG_RESULT:
                            for t in msg.get("tracks", []):
                                server_state[int(t["track_id"])] = t
                            # 估算 RTT
                            fid = msg.get("frame_idx", -1)
                            t_sent = send_timestamps.pop(fid, None)
                            if t_sent is not None:
                                latest_latency_ms = (time.time() - t_sent) * 1000
                        elif mt == MSG_ALERT:
                            recent_alerts.append(msg)
                            logger.warning(
                                f"⚠️ ALERT frame={msg.get('frame_idx')} "
                                f"tid={msg.get('track_id')} "
                                f"reason={msg.get('reason', '')[:80]}"
                            )
                        elif mt == MSG_PONG:
                            pass
                        else:
                            logger.debug(f"recv: {mt}")
                except websockets.ConnectionClosed:
                    logger.warning("server closed connection")
                finally:
                    connection_alive.clear()

            recv_task = asyncio.create_task(recv_loop())

            frame_idx = 0
            fps_window = deque(maxlen=30)
            t_last = time.time()

            try:
                while connection_alive.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("camera read failed, retry")
                        await asyncio.sleep(0.05)
                        continue

                    # 本地 YOLO Pose (这里是阻塞操作, 在事件循环里 OK 但若 CPU 慢会拖慢)
                    loop = asyncio.get_event_loop()
                    persons = await loop.run_in_executor(None, pose, frame)

                    # 发送
                    msg = build_frame_msg(
                        frame_idx=frame_idx,
                        timestamp=time.time(),
                        frame_h=actual_h, frame_w=actual_w,
                        persons=persons,
                    )
                    send_timestamps[frame_idx] = time.time()
                    if len(send_timestamps) > 60:
                        # 清理过老的(可能丢了)
                        old_keys = sorted(send_timestamps.keys())[:30]
                        for k in old_keys:
                            send_timestamps.pop(k, None)
                    try:
                        await ws.send(json.dumps(msg))
                    except websockets.ConnectionClosed:
                        logger.warning("send failed, connection closed")
                        break

                    # FPS 统计
                    now = time.time()
                    fps_window.append(now)
                    if len(fps_window) >= 2:
                        fps_eff = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0] + 1e-6)
                    else:
                        fps_eff = 0.0

                    # 渲染
                    overlay = draw_overlay(
                        frame.copy(), persons, server_state, recent_alerts,
                        fps_eff=fps_eff, latency_ms=latest_latency_ms,
                        kpt_thr=args.kpt_thr,
                    )

                    if writer is not None:
                        writer.write(overlay)

                    if not args.no_show:
                        cv2.imshow("Fall Detection (Distributed)", overlay)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            break

                    frame_idx += 1

                    if now - t_last > 10:
                        logger.info(
                            f"[stats] frame={frame_idx} fps={fps_eff:.1f} "
                            f"rtt={latest_latency_ms:.0f}ms tracks={len(server_state)} "
                            f"alerts={len(recent_alerts)}"
                        )
                        t_last = now

            finally:
                try:
                    await ws.send(json.dumps({"type": MSG_BYE}))
                except Exception:
                    pass
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await recv_task

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(
        description="Fall detection distributed client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--server", default="ws://localhost:8765",
                   help="WebSocket 服务器地址 (例如 ws://1.2.3.4:8765)")

    # 摄像头
    p.add_argument("--camera", type=int, default=0,
                   help="摄像头索引 (通常 0 = 默认摄像头)")
    p.add_argument("--cam-width", type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=720)

    # YOLO Pose
    p.add_argument("--pose-weights", default="yolo11s-pose.pt",
                   help="YOLO Pose 权重。CPU 推荐 yolo11n-pose.pt 或 yolo11s-pose.pt")
    p.add_argument("--device", default="cpu",
                   help="YOLO 设备 (cpu 或 cuda:0)")
    p.add_argument("--conf", type=float, default=0.15,
                   help="YOLO 检测置信度阈值")
    p.add_argument("--imgsz", type=int, default=640,
                   help="YOLO 推理尺寸,CPU 上建议 640")
    p.add_argument("--kpt-thr", type=float, default=0.3)

    # UI / 输出
    p.add_argument("--no-show", action="store_true", help="不开窗口,纯后台")
    p.add_argument("--save-out", default=None,
                   help="保存带 overlay 的视频到指定路径 (.mp4)")

    # 日志
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main():
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        asyncio.run(client_loop(args))
    except KeyboardInterrupt:
        logger.info("interrupted")


if __name__ == "__main__":
    main()

"""
deploy/protocol.py — 客户端 ↔ 服务端 WebSocket 通信协议

设计原则:
  - 消息全部用 JSON,可读性 > 极致带宽
  - 实测上行 ~0.3 Mbps @ 30fps 5 人,完全够用
  - 类型用字符串常量,避免魔术值散落各处

============================================================
帧请求 (Client → Server, 每帧一次)
============================================================
{
    "type": "frame",
    "frame_idx": int,             # 客户端递增帧号
    "timestamp": float,           # Unix 时间戳 (秒)
    "frame_h": int,               # 帧高 (像素)
    "frame_w": int,               # 帧宽
    "persons": [
        {
            "track_id": int,      # YOLO 的 byteTrack id (负数表示无 track)
            "bbox": [x1, y1, x2, y2],
            "keypoints": [[x, y, conf], ...]   # 17 个 COCO 点
        }
    ]
}

============================================================
推理结果 (Server → Client, 收到帧后回一次)
============================================================
{
    "type": "result",
    "frame_idx": int,
    "tracks": [
        {
            "track_id": int,
            "display_id": int,            # 经 track_merger 后的稳定 id
            "fall_prob": float,           # raw 模型概率
            "smoothed_prob": float,       # EMA 平滑
            "heuristic_score": float,
            "alerted": bool,              # 当前是否处于报警状态
            "ever_alerted": bool,         # 历史是否报警过
            "alert_reason": str,
            "bbox": [x1, y1, x2, y2],     # 服务端记录的最新 bbox (用于追踪一致)
            "is_ready": bool              # buffer 是否填满可推理
        }
    ]
}

============================================================
报警事件 (Server → Client, 异步推送, 触发即送)
============================================================
{
    "type": "alert",
    "frame_idx": int,
    "track_id": int,
    "display_id": int,
    "fall_prob": float,
    "bbox": [x1, y1, x2, y2],
    "reason": str,
    "event": "onset"               # 仅 onset 类型
}

============================================================
会话控制
============================================================
{ "type": "hello", "frame_h": int, "frame_w": int, "source_fps": float, "client_id": str }
{ "type": "hello_ack", "session_id": str, "server_version": str, "config_summary": {...} }
{ "type": "ping" } / { "type": "pong" }
{ "type": "bye" }
{ "type": "error", "code": str, "message": str }
"""

# 消息类型常量
MSG_HELLO = "hello"
MSG_HELLO_ACK = "hello_ack"
MSG_FRAME = "frame"
MSG_RESULT = "result"
MSG_ALERT = "alert"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_BYE = "bye"
MSG_ERROR = "error"

# 服务端版本
SERVER_VERSION = "fall-detection-distributed/0.1.0"

# 最大消息大小 (单帧 5 人时约 1.2 KB,留 4 MB 缓冲)
MAX_MSG_SIZE = 4 * 1024 * 1024

# 默认端口
DEFAULT_PORT = 8765


def build_frame_msg(frame_idx, timestamp, frame_h, frame_w, persons):
    """构造帧消息 (客户端用)。

    persons: List[dict],每个 dict 含 track_id / bbox / keypoints
    """
    return {
        "type": MSG_FRAME,
        "frame_idx": int(frame_idx),
        "timestamp": float(timestamp),
        "frame_h": int(frame_h),
        "frame_w": int(frame_w),
        "persons": persons,
    }


def build_hello_msg(frame_h, frame_w, source_fps, client_id=""):
    return {
        "type": MSG_HELLO,
        "frame_h": int(frame_h),
        "frame_w": int(frame_w),
        "source_fps": float(source_fps),
        "client_id": str(client_id),
    }


def build_error_msg(code, message):
    return {"type": MSG_ERROR, "code": str(code), "message": str(message)}

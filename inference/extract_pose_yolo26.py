"""
inference/extract_pose_yolo26.py — 用 YOLO26-Pose 从视频提取骨骼

设计目标:
  - 输入:任意视频文件 / RTSP 流 / 摄像头编号
  - 输出:逐帧的 (keypoints, scores, bboxes, track_ids)
  - 集成 ByteTrack(Ultralytics .track() 原生支持)做多人跟踪

为什么用 YOLO26-Pose:
  - 单 pip 包(ultralytics),无 Caffe/CMake 编译噩梦
  - 2026 年 1 月发布,内置 Residual Log-Likelihood Estimation (RLE),关键点定位精度优于 YOLOv11/v8
  - 直接输出 COCO 17 点格式,与训练时 OpenMMLab HRNet 提取的 ntu60_2d.pkl 一致
  - .track() 一行集成 ByteTrack,免去自己接 ByteTrack 仓库

CLI 用法:
    # 提取整段视频骨骼并保存
    python inference/extract_pose_yolo26.py \
        --video your_clip.mp4 \
        --out  poses/your_clip.pkl

    # 也可作为库函数被 batch_predict / realtime_demo 调用,见底部接口
"""
import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 模型加载(惰性,避免 import 时就拉模型)
# ============================================================
_MODEL_CACHE = {}


def load_pose_model(weights="yolo26x-pose.pt", device=None):
    """加载 YOLO26-Pose 模型(带缓存)。

    Args:
        weights: 权重文件名,'yolo26x-pose.pt' 或 'yolo26m-pose.pt' 等
                 首次会自动下载到 ~/.config/Ultralytics
        device:  'cuda:0' / 'cpu' / None(自动)
    """
    cache_key = (weights, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # 延迟 import,避免无 GPU 环境下也强依赖
    from ultralytics import YOLO

    model = YOLO(weights)
    if device is not None:
        model.to(device)
    _MODEL_CACHE[cache_key] = model
    print(f"[load_pose_model] 已加载 {weights}  device={device or 'auto'}")
    return model


# ============================================================
# 视频提取(主接口)
# ============================================================
def extract_video(
    video_path,
    weights="yolo26x-pose.pt",
    device=None,
    track=True,
    conf=0.25,
    iou=0.5,
    imgsz=640,
    max_persons=1,
    verbose=False,
):
    """从一段视频中提取每帧的关键点和跟踪 ID。

    Args:
        video_path:  视频路径 / RTSP / 摄像头索引(int)
        track:       True=用 ByteTrack 跟踪同一人,False=每帧独立检测
        conf:        人体框置信度阈值
        iou:         NMS IoU(YOLO26 端到端无 NMS 时此参数被忽略)
        imgsz:       推理输入尺寸,默认 640
        max_persons: 每帧最多保留几人(摔倒检测单人场景设 1;多人场景设更大)
        verbose:     是否打印 ultralytics 内部日志

    Returns:
        dict:
          {
            'keypoints':  list[T], 每个 np.ndarray (M, 17, 2)
            'scores':     list[T], 每个 np.ndarray (M, 17)
            'bboxes':     list[T], 每个 np.ndarray (M, 4) xyxy
            'track_ids':  list[T], 每个 list[int],长度=M;无 track 时为 None
            'img_shape':  (H, W)
            'num_frames': int
            'fps':        float
          }
    """
    model = load_pose_model(weights, device)

    # 打开视频拿元信息
    cap = cv2.VideoCapture(str(video_path) if not isinstance(video_path, int) else video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    # 跑推理(stream=True 是关键,否则会一次性把所有帧载内存)
    if track:
        results_gen = model.track(
            source=str(video_path) if not isinstance(video_path, int) else video_path,
            stream=True, persist=True,
            conf=conf, iou=iou, imgsz=imgsz, verbose=verbose,
            tracker="bytetrack.yaml",
        )
    else:
        results_gen = model.predict(
            source=str(video_path) if not isinstance(video_path, int) else video_path,
            stream=True,
            conf=conf, iou=iou, imgsz=imgsz, verbose=verbose,
        )

    all_kpts, all_scrs, all_boxes, all_ids = [], [], [], []
    t0 = time.time()
    for frame_idx, r in enumerate(results_gen):
        kpts_frame, scrs_frame, boxes_frame, ids_frame = _extract_one_frame(
            r, max_persons=max_persons
        )
        all_kpts.append(kpts_frame)
        all_scrs.append(scrs_frame)
        all_boxes.append(boxes_frame)
        all_ids.append(ids_frame if track else None)

    elapsed = time.time() - t0
    n = len(all_kpts)
    print(f"[extract_video] {video_path}: {n} 帧,用时 {elapsed:.2f}s ({n/max(elapsed,1e-6):.1f} FPS)")

    return {
        "keypoints": all_kpts,
        "scores": all_scrs,
        "bboxes": all_boxes,
        "track_ids": all_ids,
        "img_shape": (H, W),
        "num_frames": n,
        "fps": fps,
    }


def _extract_one_frame(result, max_persons=1):
    """从单帧 ultralytics Result 抽出关键点、置信度、框、track id。

    没检测到人时返回零数组(避免下游崩溃)。
    """
    # 没人
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return (
            np.zeros((max_persons, 17, 2), dtype=np.float32),
            np.zeros((max_persons, 17), dtype=np.float32),
            np.zeros((max_persons, 4), dtype=np.float32),
            [-1] * max_persons,
        )

    # ultralytics 返回的是 torch.Tensor,转 numpy
    kpts_xy = result.keypoints.xy.cpu().numpy()         # (N, 17, 2)
    kpts_conf = result.keypoints.conf.cpu().numpy() \
        if result.keypoints.conf is not None else np.ones(kpts_xy.shape[:2])  # (N, 17)
    bboxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else \
        np.zeros((kpts_xy.shape[0], 4))  # (N, 4)
    track_ids = []
    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int).tolist()
    else:
        track_ids = [-1] * kpts_xy.shape[0]

    N = kpts_xy.shape[0]

    # 选前 max_persons 个(按框面积大小,摔倒检测里大框=主目标)
    if N > max_persons:
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        order = np.argsort(-areas)[:max_persons]
        kpts_xy = kpts_xy[order]
        kpts_conf = kpts_conf[order]
        bboxes = bboxes[order]
        track_ids = [track_ids[i] for i in order]
    elif N < max_persons:
        pad = max_persons - N
        kpts_xy = np.concatenate([kpts_xy, np.zeros((pad, 17, 2))], axis=0)
        kpts_conf = np.concatenate([kpts_conf, np.zeros((pad, 17))], axis=0)
        bboxes = np.concatenate([bboxes, np.zeros((pad, 4))], axis=0)
        track_ids = track_ids + [-1] * pad

    return (
        kpts_xy.astype(np.float32),
        kpts_conf.astype(np.float32),
        bboxes.astype(np.float32),
        track_ids,
    )


# ============================================================
# 保存到 pickle(供调试 / 离线推理)
# ============================================================
def save_pose_pkl(pose_data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pose_data, f)
    print(f"[save_pose_pkl] -> {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="用 YOLO26-Pose 提取视频骨骼")
    parser.add_argument("--video", required=True, help="视频文件路径(或摄像头索引如 0)")
    parser.add_argument("--out", required=True, help="输出 pickle 路径")
    parser.add_argument("--weights", default="yolo26x-pose.pt",
                        help="YOLO 权重 (yolo26n/s/m/l/x-pose.pt)")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu")
    parser.add_argument("--no-track", action="store_true", help="关闭 ByteTrack 跟踪")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-persons", type=int, default=1,
                        help="每帧保留几人(摔倒检测单人=1)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 摄像头索引?
    video_arg = args.video
    if video_arg.isdigit():
        video_arg = int(video_arg)

    pose_data = extract_video(
        video_path=video_arg,
        weights=args.weights,
        device=args.device,
        track=not args.no_track,
        conf=args.conf,
        imgsz=args.imgsz,
        max_persons=args.max_persons,
        verbose=args.verbose,
    )
    save_pose_pkl(pose_data, args.out)


if __name__ == "__main__":
    main()

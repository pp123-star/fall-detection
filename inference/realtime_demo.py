"""
inference/realtime_demo.py — 实时摄像头/视频摔倒检测演示

设计目标(论文 4.5 节"部署效率分析"用):
  - 输入:摄像头 / 视频文件 / RTSP 流
  - 输出:实时窗口或保存为带可视化的 mp4
  - 在画面上叠加:人体框、骨骼、track id、摔倒概率条、警报横幅
  - 记录 FPS / 各阶段耗时,论文里要的"端到端延迟"就是这里出

为什么不直接用 batch_predict?
  - batch_predict 是先提完整视频骨骼再切分,offline 用
  - 实时 demo 必须是滚动缓冲区:每来一帧推一次骨骼,缓冲区满 clip_len 帧就跑一次分类
  - 缓冲区用 collections.deque,O(1) 入队/出队,无内存增长

用法:
    # 摄像头(默认 0)
    python inference/realtime_demo.py \
        --source 0 \
        --config configs/posec3d_fall_binary.py \
        --ckpt work_dirs/posec3d_fall_binary/best.pth

    # 视频文件,保存结果
    python inference/realtime_demo.py \
        --source test.mp4 \
        --config configs/posec3d_fall_binary.py \
        --ckpt work_dirs/posec3d_fall_binary/best.pth \
        --save-out demo_out.mp4 \
        --no-show

    # 调低分类频率提升 FPS(每 8 帧才跑一次分类器,中间帧沿用最近一次概率)
    python inference/realtime_demo.py ... --infer-every 8
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.extract_pose_yolo26 import load_pose_model, _extract_one_frame
from inference.pose_to_pyskl_format import build_sample
from inference.batch_predict import load_action_model, predict_clip


# ============================================================
# COCO 17 点骨骼连线(画图用)
# ============================================================
COCO_SKELETON = [
    (5, 7), (7, 9),       # 左臂
    (6, 8), (8, 10),      # 右臂
    (5, 6),               # 肩
    (5, 11), (6, 12),     # 躯干
    (11, 12),             # 髋
    (11, 13), (13, 15),   # 左腿
    (12, 14), (14, 16),   # 右腿
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头
    (3, 5), (4, 6),       # 耳到肩
]


def draw_skeleton(img, kpts, scores, color, kpt_thr=0.3):
    """在 img 上画 17 点骨骼。"""
    # 关键点
    for j, (x, y) in enumerate(kpts):
        if scores[j] < kpt_thr:
            continue
        cv2.circle(img, (int(x), int(y)), 3, color, -1)
    # 连线
    for a, b in COCO_SKELETON:
        if scores[a] < kpt_thr or scores[b] < kpt_thr:
            continue
        pt1 = (int(kpts[a, 0]), int(kpts[a, 1]))
        pt2 = (int(kpts[b, 0]), int(kpts[b, 1]))
        cv2.line(img, pt1, pt2, color, 2)


def draw_hud(img, fall_prob, threshold, fps, infer_ms, alert=False):
    """画 HUD:概率条、FPS、警报横幅。"""
    H, W = img.shape[:2]

    # 半透明顶部条
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, 70), (40, 40, 40), -1)
    img[:] = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # 概率条
    bar_x, bar_y, bar_w, bar_h = 20, 25, 300, 22
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 90), -1)
    fill = int(bar_w * float(np.clip(fall_prob, 0, 1)))
    bar_color = (60, 60, 240) if fall_prob > threshold else (60, 200, 60)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    # 阈值线
    th_x = bar_x + int(bar_w * threshold)
    cv2.line(img, (th_x, bar_y - 4), (th_x, bar_y + bar_h + 4), (255, 255, 255), 1)
    cv2.putText(img, f"P(fall)={fall_prob:.2f}", (bar_x + bar_w + 12, bar_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS / 延迟
    info = f"FPS:{fps:5.1f}  Infer:{infer_ms:5.1f}ms"
    cv2.putText(img, info, (W - 290, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 警报横幅
    if alert:
        ov = img.copy()
        cv2.rectangle(ov, (0, H - 70), (W, H), (0, 0, 220), -1)
        img[:] = cv2.addWeighted(ov, 0.7, img, 0.3, 0)
        text = "!! FALL DETECTED !!"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(img, text, ((W - ts[0]) // 2, H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


# ============================================================
# 主循环
# ============================================================
def run_realtime(args):
    # 1. 模型
    pose_model = load_pose_model(args.pose_weights, args.device)
    action_model = load_action_model(args.config, args.ckpt, args.device)

    # 2. 视频源
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"无法打开视频源: {args.source}")

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    print(f"[源] {args.source}  {W}x{H}@{src_fps:.1f}fps")

    # 3. 输出 writer
    writer = None
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_out, fourcc, src_fps, (W, H))
        print(f"[写] 输出到 {args.save_out}")

    # 4. 滚动缓冲
    kpt_buf = deque(maxlen=args.clip_len)
    scr_buf = deque(maxlen=args.clip_len)

    # 5. 状态
    last_fall_prob = 0.0
    last_infer_ms = 0.0
    last_kpts = np.zeros((1, 17, 2), dtype=np.float32)
    last_scrs = np.zeros((1, 17), dtype=np.float32)
    last_box = np.zeros((1, 4), dtype=np.float32)
    last_id = -1
    fps_smoother = deque(maxlen=30)

    # 6. 跟踪状态(model.track 在 stream 模式才好用,这里手动逐帧 predict + 简单匹配也行)
    # 简化:用 model.track 配合 stream-of-images 不方便,这里逐帧 predict 然后让画面自然
    frame_idx = 0
    alert_frames_left = 0  # 警报横幅持续 N 帧,避免闪烁

    print("[开始] 按 q 退出")
    while True:
        t_frame = time.time()
        ok, frame = cap.read()
        if not ok:
            print("[源结束]")
            break

        # 6.1 姿态估计(逐帧 predict;.track 在循环外用 stream 才连续,这里简化)
        # 注:演示用,FPS 完全可接受;真要追求最高 FPS 可改成预先开 model.track 生成器
        r = pose_model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               verbose=False, device=args.device)[0]
        kpts_f, scrs_f, boxes_f, ids_f = _extract_one_frame(r, max_persons=1)

        last_kpts, last_scrs, last_box = kpts_f, scrs_f, boxes_f
        last_id = ids_f[0] if ids_f else -1

        kpt_buf.append(kpts_f)
        scr_buf.append(scrs_f)

        # 6.2 满 clip_len 帧时跑一次分类(可降频)
        if len(kpt_buf) == args.clip_len and frame_idx % args.infer_every == 0:
            t_inf = time.time()
            sample = build_sample(
                keypoints_seq=list(kpt_buf),
                scores_seq=list(scr_buf),
                img_shape=(H, W),
                frame_dir=f"live_{frame_idx}",
            )
            with torch.no_grad():
                last_fall_prob = predict_clip(action_model, sample, device=args.device)
            last_infer_ms = (time.time() - t_inf) * 1000

            # 触发警报
            if last_fall_prob > args.threshold:
                alert_frames_left = int(src_fps * args.alert_hold)

        # 6.3 画
        # 骨骼
        col = (60, 200, 60) if last_fall_prob <= args.threshold else (60, 60, 240)
        draw_skeleton(frame, last_kpts[0], last_scrs[0], color=col)
        # bbox
        if last_box[0].sum() > 0:
            x1, y1, x2, y2 = last_box[0].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"id:{last_id}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        # HUD
        alert = alert_frames_left > 0
        if alert:
            alert_frames_left -= 1
        loop_ms = (time.time() - t_frame) * 1000
        fps_smoother.append(1000.0 / max(loop_ms, 1e-6))
        cur_fps = float(np.mean(fps_smoother))
        draw_hud(frame, last_fall_prob, args.threshold, cur_fps, last_infer_ms, alert)

        if writer is not None:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("Fall Detection (q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print(f"\n[结束] 共 {frame_idx} 帧,平均 FPS={np.mean(fps_smoother):.1f}")


def main():
    parser = argparse.ArgumentParser(description="实时摔倒检测演示")
    parser.add_argument("--source", default="0",
                        help="视频路径 / RTSP / 摄像头编号(默认 0)")
    parser.add_argument("--config", required=True, help="MMAction2 config")
    parser.add_argument("--ckpt", required=True, help="训练好的 .pth")
    parser.add_argument("--pose-weights", default="yolo26x-pose.pt")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--clip-len", type=int, default=48,
                        help="滑动缓冲区帧数,应等于训练 config 的 clip_len")
    parser.add_argument("--infer-every", type=int, default=4,
                        help="每 N 帧跑一次分类器(降频以提升 FPS,中间帧沿用上次概率)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 人体框置信度")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--alert-hold", type=float, default=1.5,
                        help="检测到摔倒后警报横幅保持秒数")

    parser.add_argument("--save-out", default=None, help="保存可视化结果到 mp4")
    parser.add_argument("--no-show", action="store_true",
                        help="不开窗口(服务器/Headless 必加)")
    args = parser.parse_args()

    run_realtime(args)


if __name__ == "__main__":
    main()

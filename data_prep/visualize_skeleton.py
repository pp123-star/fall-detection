"""
data_prep/visualize_skeleton.py — 可视化关键点序列,确认头连头、脚连脚

★ 这是避坑核心!上一版项目踩过的最大坑就是关键点顺序错位,
  必须人工肉眼检查骨骼连线对不对。

COCO 17 点定义(0-indexed):
  0  nose            鼻子
  1  left_eye        左眼
  2  right_eye       右眼
  3  left_ear        左耳
  4  right_ear       右耳
  5  left_shoulder   左肩
  6  right_shoulder  右肩
  7  left_elbow      左肘
  8  right_elbow     右肘
  9  left_wrist      左腕
  10 right_wrist     右腕
  11 left_hip        左髋
  12 right_hip       右髋
  13 left_knee       左膝
  14 right_knee      右膝
  15 left_ankle      左踝
  16 right_ankle     右踝

骨骼连线(应该形成一个合理人形):
  0-1, 0-2          鼻-眼
  1-3, 2-4          眼-耳
  5-7, 7-9          左肩-肘-腕
  6-8, 8-10         右肩-肘-腕
  5-6               肩
  5-11, 6-12        肩-髋
  11-12             髋
  11-13, 13-15      左髋-膝-踝
  12-14, 14-16      右髋-膝-踝

输出:
  vis/skeleton_<样本名>.mp4 或 .gif

用法:
    python data_prep/visualize_skeleton.py --src data/fall_binary_xsub.pkl --num 5
    python data_prep/visualize_skeleton.py --src data/fall_binary_xsub.pkl --pos-only --num 3
"""
import argparse
import pickle
import random
from pathlib import Path

import cv2
import numpy as np


# COCO 17 骨骼连线(start_idx, end_idx, color BGR)
COCO_SKELETON = [
    (0, 1, (255, 0, 0)), (0, 2, (255, 0, 0)),
    (1, 3, (255, 100, 0)), (2, 4, (255, 100, 0)),
    (5, 7, (0, 255, 0)), (7, 9, (0, 255, 0)),       # 左臂(绿)
    (6, 8, (0, 255, 255)), (8, 10, (0, 255, 255)),  # 右臂(黄)
    (5, 6, (200, 200, 200)),                         # 肩
    (5, 11, (255, 0, 255)), (6, 12, (255, 0, 255)),  # 躯干
    (11, 12, (200, 200, 200)),
    (11, 13, (0, 0, 255)), (13, 15, (0, 0, 255)),    # 左腿(红)
    (12, 14, (255, 0, 200)), (14, 16, (255, 0, 200)),  # 右腿(紫)
]

# 关键点颜色(头部蓝、躯干灰、左肢绿/红、右肢黄/紫)
KP_COLORS = [
    (255, 0, 0),    # 0 nose
    (255, 50, 0), (255, 50, 0),       # 1,2 eyes
    (255, 100, 0), (255, 100, 0),     # 3,4 ears
    (0, 255, 0), (0, 255, 255),       # 5,6 shoulders L/R
    (0, 255, 0), (0, 255, 255),       # 7,8 elbows
    (0, 255, 0), (0, 255, 255),       # 9,10 wrists
    (255, 0, 255), (255, 0, 255),     # 11,12 hips
    (0, 0, 255), (255, 0, 200),       # 13,14 knees
    (0, 0, 255), (255, 0, 200),       # 15,16 ankles
]


def draw_skeleton_on_canvas(canvas, keypoints, scores=None, thre=0.3):
    """
    在 canvas 上画一帧的骨骼。
    
    keypoints: (V=17, C=2),已经归一化或像素坐标
    scores:    (V=17,) 或 None
    """
    if scores is None:
        scores = np.ones(keypoints.shape[0])

    # 先画连线
    for s, e, color in COCO_SKELETON:
        if scores[s] < thre or scores[e] < thre:
            continue
        x1, y1 = keypoints[s].astype(int)
        x2, y2 = keypoints[e].astype(int)
        cv2.line(canvas, (x1, y1), (x2, y2), color, 2)

    # 再画点(覆盖在线上)
    for i in range(keypoints.shape[0]):
        if scores[i] < thre:
            continue
        x, y = keypoints[i].astype(int)
        cv2.circle(canvas, (x, y), 4, KP_COLORS[i], -1)
        # 标关键点编号(便于排查错位)
        cv2.putText(canvas, str(i), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas


def normalize_keypoints_for_vis(kpt, img_shape, out_size=(640, 480)):
    """
    把关键点归一到指定画布大小。
    
    kpt: (T, V, 2)
    img_shape: 原图 (H, W),来自 annotation
    """
    h_orig, w_orig = img_shape
    h_out, w_out = out_size[1], out_size[0]
    scale = min(w_out / w_orig, h_out / h_orig)

    kpt_out = kpt * scale
    # 居中
    dx = (w_out - w_orig * scale) / 2
    dy = (h_out - h_orig * scale) / 2
    kpt_out[..., 0] += dx
    kpt_out[..., 1] += dy
    return kpt_out


def render_sample_to_video(annotation, out_path, fps=15, out_size=(640, 480)):
    """
    把一个 annotation 的骨骼序列渲染成 mp4 视频。
    
    annotation:
        keypoint:       (M, T, V, C)
        keypoint_score: (M, T, V)
        img_shape:      (H, W)
        label:          0 or 1
        frame_dir:      str
    """
    kpt = annotation["keypoint"]                # (M, T, V, C)
    score = annotation.get("keypoint_score")     # (M, T, V) or None
    M, T, V, C = kpt.shape

    img_shape = annotation.get("img_shape") or annotation.get("original_shape")
    if img_shape is None:
        img_shape = (1080, 1920)  # NTU 默认

    # 归一化坐标到画布
    kpt_vis = np.stack(
        [normalize_keypoints_for_vis(kpt[m], img_shape, out_size) for m in range(M)],
        axis=0,
    )  # (M, T, V, C)

    label = int(annotation["label"])
    name = annotation["frame_dir"]
    label_text = "★ FALL ★" if label == 1 else "non-fall"
    label_color = (0, 0, 255) if label == 1 else (0, 200, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, out_size)

    for t in range(T):
        canvas = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
        for m in range(M):
            sc = score[m, t] if score is not None else None
            # 跳过全 0 的"占位"人
            if np.all(kpt_vis[m, t] == 0):
                continue
            draw_skeleton_on_canvas(canvas, kpt_vis[m, t], sc)

        # 写元信息
        cv2.putText(canvas, f"{name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, f"label = {label}  {label_text}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
        cv2.putText(canvas, f"frame {t+1}/{T}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(canvas)

    writer.release()


def main():
    parser = argparse.ArgumentParser(description="骨骼可视化校验")
    parser.add_argument("--src", default="data/fall_binary_xsub.pkl")
    parser.add_argument("--out-dir", default="vis")
    parser.add_argument("--num", type=int, default=4,
                        help="可视化样本数")
    parser.add_argument("--pos-only", action="store_true",
                        help="只可视化正样本(摔倒)")
    parser.add_argument("--neg-only", action="store_true",
                        help="只可视化负样本")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    random.seed(args.seed)

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取 {src}")
    with open(src, "rb") as f:
        data = pickle.load(f)

    anns = data["annotations"]
    if args.pos_only:
        anns = [a for a in anns if a["label"] == 1]
    elif args.neg_only:
        anns = [a for a in anns if a["label"] == 0]

    samples = random.sample(anns, min(args.num, len(anns)))

    print(f"\n将可视化 {len(samples)} 个样本到 {out_dir}/")
    for i, ann in enumerate(samples):
        out_path = out_dir / f"sample_{i:02d}_{ann['frame_dir']}_label{ann['label']}.mp4"
        render_sample_to_video(ann, out_path, fps=args.fps)
        print(f"  [{i+1}/{len(samples)}] {out_path.name}  "
              f"({ann['keypoint'].shape[1]} 帧)")

    print("\n" + "=" * 60)
    print("✓ 可视化完成,请打开视频肉眼检查:")
    print("  - 头(0号点)应该在身体顶端,不应该在脚下")
    print("  - 关节连线应该形成合理人形(头-躯干-四肢)")
    print("  - 摔倒样本最后几帧应该是躺在地上的姿态")
    print("  - 各编号关键点位置符合 COCO 17 点定义")
    print("=" * 60)


if __name__ == "__main__":
    main()

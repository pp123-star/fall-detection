"""
inference/batch_predict.py — 批量视频推理

主要用途:
  1. 单视频整体打分(论文 demo 录屏前先这样测)
  2. 批量跑测试集(URFD/Kaggle 摔倒视频),给论文 4.x 节做"跨数据集泛化"实验

整体流程:
    video.mp4
       │
       ▼  YOLO26-Pose 提取(extract_pose_yolo26)
       │
       ▼  COCO 17 点 + 滑窗 (clip_len=48, stride=16)
       │
       ▼  PoseConv3D / ST-GCN++ 推理(load 训练好的 ckpt)
       │
       ▼  每个 clip 输出 P(fall),聚合为视频级判定
       │
       ▼  输出 JSON / CSV(供论文)

聚合策略:
  - max:  视频内任一 clip P(fall) > 阈值 → 判为摔倒(最敏感,适合摔倒检测)
  - mean: clip 概率取均值 vs 阈值
  - vote: 超阈值 clip 数 > 总数一半 → 判为摔倒(最保守)

用法:
    # 单视频
    python inference/batch_predict.py \
        --video data/raw/urfd_001.mp4 \
        --config configs/posec3d_fall_binary.py \
        --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18.pth \
        --out preds/urfd_001.json

    # 批量(指定一个视频文件夹 + 标签 CSV)
    python inference/batch_predict.py \
        --video-dir data/raw/urfd/ \
        --label-csv data/raw/urfd_labels.csv \
        --config configs/posec3d_fall_binary.py \
        --ckpt work_dirs/posec3d_fall_binary/best.pth \
        --out preds/urfd_results.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 让 import 找到本包
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.extract_pose_yolo26 import extract_video
from inference.pose_to_pyskl_format import build_sample, split_into_clips


# ============================================================
# 加载 MMAction2 模型
# ============================================================
def load_action_model(config_path, ckpt_path, device="cuda:0"):
    """加载训练好的动作识别模型。"""
    from mmaction.apis import init_recognizer
    model = init_recognizer(str(config_path), str(ckpt_path), device=device)
    model.eval()
    print(f"[load_action_model] {ckpt_path} -> {device}")
    return model


# ============================================================
# 单 clip 推理
# ============================================================
@torch.no_grad()
def predict_clip(model, clip_sample, device="cuda:0"):
    """给定单个 clip dict,返回 P(fall) ∈ [0, 1]。

    走 MMAction2 PoseDataset 的 pipeline,从 dict → torch 输入。
    不同模型(PoseConv3D vs ST-GCN++)的 pipeline 不同,这里通过 model.cfg 自动选。
    """
    from mmengine.dataset import Compose, pseudo_collate

    # 从 cfg 取 val pipeline(确保与训练时一致)
    val_pipeline_cfg = model.cfg.val_pipeline if hasattr(model.cfg, "val_pipeline") \
        else model.cfg.test_pipeline if hasattr(model.cfg, "test_pipeline") \
        else model.cfg.val_dataloader.dataset.pipeline

    pipeline = Compose(val_pipeline_cfg)
    data = pseudo_collate([pipeline(clip_sample.copy())])
    result = model.test_step(data)[0]

    # pred_score: 通常是 (num_classes,) 的 tensor
    score = result.pred_score if hasattr(result, "pred_score") else result.get("pred_score")
    if torch.is_tensor(score):
        score = score.cpu().numpy()
    # 二分类:类别 1 = 摔倒
    return float(score[1])


# ============================================================
# 视频级聚合
# ============================================================
def aggregate(clip_probs, strategy="max", threshold=0.5):
    """把 clip 级概率聚合为视频级判定。

    Returns:
        (is_fall: bool, agg_prob: float, n_clips: int)
    """
    probs = np.asarray(clip_probs)
    if probs.size == 0:
        return False, 0.0, 0

    if strategy == "max":
        agg = float(probs.max())
        return agg > threshold, agg, len(probs)
    if strategy == "mean":
        agg = float(probs.mean())
        return agg > threshold, agg, len(probs)
    if strategy == "vote":
        n_over = int((probs > threshold).sum())
        return n_over > len(probs) / 2, float(n_over / len(probs)), len(probs)
    raise ValueError(f"未知 aggregate strategy: {strategy}")


# ============================================================
# 单视频完整流程
# ============================================================
def predict_video(
    video_path,
    action_model,
    pose_weights="yolo26x-pose.pt",
    device="cuda:0",
    clip_len=48,
    stride=16,
    threshold=0.5,
    aggregate_strategy="max",
):
    """完整地预测一个视频是否含摔倒。

    Returns:
        dict(video, n_frames, n_clips, clip_probs, agg_prob, is_fall, elapsed_s, ...)
    """
    t0 = time.time()

    # 1. 提取骨骼
    pose = extract_video(
        video_path=str(video_path),
        weights=pose_weights,
        device=device,
        track=True,
        max_persons=1,  # 摔倒检测取最大目标
    )

    # 2. 构造 MMAction2 样本
    sample = build_sample(
        keypoints_seq=pose["keypoints"],
        scores_seq=pose["scores"],
        img_shape=pose["img_shape"],
        frame_dir=Path(video_path).stem,
    )

    # 3. 滑窗
    clips = split_into_clips(sample, clip_len=clip_len, stride=stride)

    # 4. 逐 clip 推理
    clip_probs = []
    for clip in clips:
        p = predict_clip(action_model, clip, device=device)
        clip_probs.append(p)

    # 5. 聚合
    is_fall, agg_prob, n_clips = aggregate(clip_probs, aggregate_strategy, threshold)
    elapsed = time.time() - t0

    return dict(
        video=str(video_path),
        n_frames=pose["num_frames"],
        n_clips=n_clips,
        clip_probs=[round(p, 4) for p in clip_probs],
        aggregate_strategy=aggregate_strategy,
        threshold=threshold,
        agg_prob=round(agg_prob, 4),
        is_fall=bool(is_fall),
        elapsed_s=round(elapsed, 2),
        fps=round(pose["num_frames"] / max(elapsed, 1e-6), 1),
    )


# ============================================================
# 批量
# ============================================================
def predict_batch(
    video_files,
    labels,
    action_model,
    pose_weights,
    device,
    clip_len,
    stride,
    threshold,
    aggregate_strategy,
):
    """跑一组视频,返回每个的结果 + 整体指标。"""
    results = []
    for i, vp in enumerate(video_files):
        print(f"\n--- [{i+1}/{len(video_files)}] {vp} ---")
        try:
            r = predict_video(
                video_path=vp,
                action_model=action_model,
                pose_weights=pose_weights,
                device=device,
                clip_len=clip_len,
                stride=stride,
                threshold=threshold,
                aggregate_strategy=aggregate_strategy,
            )
        except Exception as e:
            print(f"[ERROR] {vp}: {e}")
            r = dict(video=str(vp), error=str(e), is_fall=None, agg_prob=None)
        if labels is not None:
            r["gt_label"] = int(labels[i])
        results.append(r)

    # 汇总(有 GT 时)
    if labels is not None:
        valid = [r for r in results if r.get("is_fall") is not None]
        tp = sum(1 for r in valid if r["is_fall"] and r["gt_label"] == 1)
        fp = sum(1 for r in valid if r["is_fall"] and r["gt_label"] == 0)
        fn = sum(1 for r in valid if not r["is_fall"] and r["gt_label"] == 1)
        tn = sum(1 for r in valid if not r["is_fall"] and r["gt_label"] == 0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (tp + tn) / max(len(valid), 1)
        print("\n=== 批量汇总 ===")
        print(f"  样本数:{len(valid)}  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    return results


# ============================================================
# IO
# ============================================================
def save_results(results, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".csv":
        keys = ["video", "n_frames", "n_clips", "agg_prob", "is_fall",
                "gt_label", "elapsed_s", "fps", "aggregate_strategy", "threshold"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow(r)
    else:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[save_results] -> {out_path}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="批量视频摔倒检测推理")
    parser.add_argument("--video", help="单个视频路径")
    parser.add_argument("--video-dir", help="批量模式:视频文件夹")
    parser.add_argument("--label-csv", help="批量模式可选:CSV 含两列 video,label(0/1)")
    parser.add_argument("--video-ext", default=".mp4,.avi,.mov,.mkv",
                        help="批量模式视频后缀(逗号分隔)")

    parser.add_argument("--config", required=True, help="MMAction2 config")
    parser.add_argument("--ckpt", required=True, help="训练好的 checkpoint .pth")
    parser.add_argument("--pose-weights", default="yolo26x-pose.pt")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--clip-len", type=int, default=48)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--aggregate", default="max", choices=["max", "mean", "vote"])

    parser.add_argument("--out", required=True, help="结果输出路径(.json / .csv)")
    args = parser.parse_args()

    assert args.video or args.video_dir, "至少给一个 --video 或 --video-dir"

    # 加载动作模型(只加载一次)
    action_model = load_action_model(args.config, args.ckpt, args.device)

    # 单视频
    if args.video:
        r = predict_video(
            video_path=args.video,
            action_model=action_model,
            pose_weights=args.pose_weights,
            device=args.device,
            clip_len=args.clip_len,
            stride=args.stride,
            threshold=args.threshold,
            aggregate_strategy=args.aggregate,
        )
        print("\n=== 结果 ===")
        print(json.dumps(r, indent=2, ensure_ascii=False))
        save_results([r], args.out)
        return

    # 批量
    video_dir = Path(args.video_dir)
    exts = set(args.video_ext.split(","))
    video_files = sorted([
        p for p in video_dir.rglob("*") if p.suffix.lower() in exts
    ])
    print(f"[batch] 在 {video_dir} 找到 {len(video_files)} 个视频")
    assert video_files, "目录里没找到视频"

    # 标签
    labels = None
    if args.label_csv:
        labels_map = {}
        with open(args.label_csv) as f:
            for row in csv.DictReader(f):
                labels_map[row["video"]] = int(row["label"])
        labels = []
        for vp in video_files:
            key = vp.name if vp.name in labels_map else str(vp)
            if key not in labels_map:
                key = vp.stem
            labels.append(labels_map.get(key, -1))
        if -1 in labels:
            print(f"[警告] 有 {labels.count(-1)} 个视频在 label-csv 里找不到,gt 记为 -1")

    results = predict_batch(
        video_files=video_files,
        labels=labels,
        action_model=action_model,
        pose_weights=args.pose_weights,
        device=args.device,
        clip_len=args.clip_len,
        stride=args.stride,
        threshold=args.threshold,
        aggregate_strategy=args.aggregate,
    )
    save_results(results, args.out)


if __name__ == "__main__":
    main()

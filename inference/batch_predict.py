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

    NOTE: 已修复 v1 的两个 bug,与 multitarget_realtime_demo 保持一致:
      • Compose 改用 mmengine.dataset.Compose(MMAction2 v1.x 的 transforms 已被
        合并到 mmengine 的 BaseTransform 体系)
      • test_step 输入用 pseudo_collate 包装单样本,避免手动 unsqueeze + dict
    """
    from mmengine.dataset import Compose, pseudo_collate

    # 从 cfg 取 val pipeline
    val_pipeline_cfg = model.cfg.val_pipeline if hasattr(model.cfg, "val_pipeline") \
        else model.cfg.test_pipeline if hasattr(model.cfg, "test_pipeline") \
        else model.cfg.val_dataloader.dataset.pipeline

    pipeline = Compose(val_pipeline_cfg)
    data = pipeline(clip_sample.copy())

    # 用 pseudo_collate 把单样本封装成 batch
    batch = pseudo_collate([data])
    result = model.test_step(batch)[0]

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
    # === 新增:真实视频友好参数 ===
    target_fps=0.0,        # >0 且未给 time_window_sec 时,用 clip_len/target_fps 推导窗口秒数
    time_window_sec=0.0,   # >0 时启用 time-window 模式:每段窗口覆盖该秒数,均匀采 clip_len 帧
    window_stride_sec=0.0, # 滑窗步长(秒);仅 time_window_sec>0 时生效;<=0 → 默认 = time_window/3
    topk=5,                # 输出 top-k 概率,用于诊断
    prob_log_jsonl=None,   # >0 时把每个 clip 的概率写到 JSONL
    ground_truth=None,     # 0/1/None,仅用于 summary 诊断
):
    """完整地预测一个视频是否含摔倒。

    两种 clip 切分模式(互斥,二选一):

      A) **stride 模式**(默认,原行为):
         先用 build_sample + split_into_clips(clip_len, stride) 在所有原始帧上切窗。
         适合训练分布相近的视频。

      B) **time-window 模式**(time_window_sec > 0):
         按真实时间窗口切多个 clip,每个 clip 内部从原始帧中均匀采样 clip_len 帧。
         这才是"60fps + 1.6s 窗口 + 48 帧采样"的正确做法,适合手机视频。

    Returns:
        dict 包含:
          • video, n_frames, n_clips, clip_probs, agg_prob, is_fall, elapsed_s, fps
          • max_pfall, top{topk}_pfall, mean_top{topk}_pfall, mean_pfall, median_pfall
          • mode ('stride' or 'time_window'), source_fps, time_window_sec
          • ground_truth, diagnosis
    """
    t0 = time.time()

    # 1. 提取骨骼(全片)
    pose = extract_video(
        video_path=str(video_path),
        weights=pose_weights,
        device=device,
        track=True,
        max_persons=1,
    )
    source_fps = float(pose["fps"])

    # 2. 切 clips:根据模式选不同切法
    effective_time_window_sec = time_window_sec
    if effective_time_window_sec <= 0 and target_fps > 0:
        effective_time_window_sec = clip_len / target_fps

    if effective_time_window_sec > 0:
        # B) time-window 模式
        mode = "time_window"
        window_frames = max(clip_len, int(round(source_fps * effective_time_window_sec)))
        stride_sec = window_stride_sec if window_stride_sec > 0 else effective_time_window_sec / 3.0
        stride_frames = max(1, int(round(source_fps * stride_sec)))

        clips = _build_time_window_clips(
            pose=pose,
            clip_len=clip_len,
            window_frames=window_frames,
            stride_frames=stride_frames,
            video_stem=Path(video_path).stem,
        )
        print(f"[predict_video] time_window 模式:source_fps={source_fps:.1f} "
              f"window={window_frames}帧({effective_time_window_sec:.2f}s) "
              f"stride={stride_frames}帧({stride_sec:.2f}s) → {len(clips)} clips")
    else:
        # A) stride 模式(原行为)
        mode = "stride"
        sample = build_sample(
            keypoints_seq=pose["keypoints"],
            scores_seq=pose["scores"],
            img_shape=pose["img_shape"],
            frame_dir=Path(video_path).stem,
        )
        clips = split_into_clips(sample, clip_len=clip_len, stride=stride)
        print(f"[predict_video] stride 模式:source_fps={source_fps:.1f} "
              f"clip_len={clip_len} stride={stride} → {len(clips)} clips")

    # 3. 逐 clip 推理
    clip_probs = []
    prob_log_fh = None
    if prob_log_jsonl:
        Path(prob_log_jsonl).parent.mkdir(parents=True, exist_ok=True)
        prob_log_fh = open(prob_log_jsonl, "w", encoding="utf-8")

    for i, clip in enumerate(clips):
        try:
            p = predict_clip(action_model, clip, device=device)
        except Exception as e:  # noqa: BLE001
            print(f"  [clip {i}] 推理失败,跳过:{e}")
            continue
        clip_probs.append(p)
        if prob_log_fh:
            prob_log_fh.write(json.dumps({
                "video": str(video_path),
                "clip_idx": i, "frame_dir": clip.get("frame_dir", ""),
                "p_fall": round(p, 6),
            }, ensure_ascii=False) + "\n")
    if prob_log_fh:
        prob_log_fh.close()

    # 4. 聚合 + 诊断
    is_fall, agg_prob, n_clips = aggregate(clip_probs, aggregate_strategy, threshold)
    probs_arr = np.asarray(clip_probs) if clip_probs else np.zeros(0)
    topk_vals = sorted(clip_probs, reverse=True)[:topk]

    # diagnosis label
    if not clip_probs:
        diag = "no_inference"
    elif ground_truth == 0:
        diag = "false_alarm" if is_fall else "true_negative"
    elif is_fall:
        diag = "detected"
    else:
        m = float(probs_arr.max())
        if m >= threshold:
            diag = "just_below_threshold"
        elif m >= 0.3:
            diag = "partial_signal"
        else:
            diag = "model_unaware"

    elapsed = time.time() - t0
    return dict(
        video=str(video_path),
        mode=mode,
        source_fps=round(source_fps, 2),
        time_window_sec=effective_time_window_sec,
        n_frames=pose["num_frames"],
        n_clips=n_clips,
        clip_probs=[round(p, 4) for p in clip_probs],
        aggregate_strategy=aggregate_strategy,
        threshold=threshold,
        agg_prob=round(agg_prob, 4),
        is_fall=bool(is_fall),
        # 诊断字段
        max_pfall=round(float(probs_arr.max()), 4) if probs_arr.size else 0.0,
        mean_pfall=round(float(probs_arr.mean()), 4) if probs_arr.size else 0.0,
        median_pfall=round(float(np.median(probs_arr)), 4) if probs_arr.size else 0.0,
        **{f"top{topk}_pfall": [round(v, 4) for v in topk_vals]},
        **{f"mean_top{topk}_pfall": round(float(np.mean(topk_vals)), 4) if topk_vals else 0.0},
        ground_truth=ground_truth,
        diagnosis=diag,
        elapsed_s=round(elapsed, 2),
        fps=round(pose["num_frames"] / max(elapsed, 1e-6), 1),
    )


# ============================================================
# time-window 模式的 clip 构造
# ============================================================
def _build_time_window_clips(pose, clip_len, window_frames, stride_frames, video_stem):
    """从 pose["keypoints"](逐帧 (1,17,2))中按 window_frames 切窗,
    每段从中均匀采样 clip_len 帧,组成多个 MMAction2 PoseDataset 样本。

    Returns: list[dict],每个 dict 就是 build_sample 的输出。
    """
    H, W = pose["img_shape"]
    kpts = pose["keypoints"]
    scrs = pose["scores"]
    n = len(kpts)
    if n == 0:
        return []

    # 滑动窗口起点
    starts = list(range(0, max(1, n - window_frames + 1), stride_frames))
    if not starts or (n >= window_frames and starts[-1] + window_frames < n):
        starts.append(max(0, n - window_frames))

    clips = []
    for s in starts:
        e = min(s + window_frames, n)
        win_kpts = kpts[s:e]
        win_scrs = scrs[s:e]
        m = len(win_kpts)

        # 均匀采样 clip_len 帧
        if m >= clip_len:
            idx = np.linspace(0, m - 1, clip_len)
            idx = np.round(idx).astype(int)
            idx = np.clip(idx, 0, m - 1)
            sub_kpts = [win_kpts[i] for i in idx]
            sub_scrs = [win_scrs[i] for i in idx]
        else:
            # 窗口不足:循环补满到 clip_len
            reps = (clip_len + m - 1) // m
            sub_kpts = (win_kpts * reps)[:clip_len]
            sub_scrs = (win_scrs * reps)[:clip_len]

        sample = build_sample(
            keypoints_seq=sub_kpts,
            scores_seq=sub_scrs,
            img_shape=(H, W),
            frame_dir=f"{video_stem}_win{s}-{e}",
        )
        clips.append(sample)
    return clips


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
    # 新增
    target_fps=0.0,
    time_window_sec=0.0,
    window_stride_sec=0.0,
    topk=5,
    prob_log_dir=None,
):
    """跑一组视频,返回每个的结果 + 整体指标。"""
    results = []
    for i, vp in enumerate(video_files):
        print(f"\n--- [{i+1}/{len(video_files)}] {vp} ---")
        gt = int(labels[i]) if labels is not None else None
        prob_log_path = None
        if prob_log_dir:
            prob_log_path = str(Path(prob_log_dir) / f"{Path(vp).stem}_prob.jsonl")
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
                target_fps=target_fps,
                time_window_sec=time_window_sec,
                window_stride_sec=window_stride_sec,
                topk=topk,
                prob_log_jsonl=prob_log_path,
                ground_truth=gt,
            )
        except Exception as e:
            print(f"[ERROR] {vp}: {e}")
            r = dict(video=str(vp), error=str(e), is_fall=None, agg_prob=None,
                     max_pfall=None, diagnosis="error")
        if labels is not None:
            r["gt_label"] = gt
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
        # 按 diagnosis 分组
        from collections import Counter
        diag_counts = Counter(r.get("diagnosis", "?") for r in valid)
        print(f"  诊断分布:{dict(diag_counts)}")

    return results


# ============================================================
# IO
# ============================================================
def save_results(results, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".csv":
        keys = ["video", "mode", "source_fps", "n_frames", "n_clips",
                "max_pfall", "mean_pfall", "median_pfall",
                "agg_prob", "is_fall", "gt_label", "diagnosis",
                "elapsed_s", "fps", "aggregate_strategy", "threshold",
                "time_window_sec"]
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
    parser.add_argument("--stride", type=int, default=16,
                        help="stride 模式滑窗步长(帧),仅 --time-window-sec=0 时生效")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--aggregate", default="max", choices=["max", "mean", "vote"])

    # 新增:真实视频友好参数
    parser.add_argument("--target-fps", type=float, default=0.0,
                        help="训练等效目标 fps。若未显式给 --time-window-sec,用 clip_len/target_fps 推导窗口")
    parser.add_argument("--time-window-sec", type=float, default=0.0,
                        help=">0 启用 time-window 模式(推荐手机视频用 1.6 或 2.0)")
    parser.add_argument("--window-stride-sec", type=float, default=0.0,
                        help="time-window 模式的滑窗步长(秒),0 = window/3")
    parser.add_argument("--topk", type=int, default=5,
                        help="输出 top-k 概率,诊断用")
    parser.add_argument("--prob-log-dir", default=None,
                        help="批量模式:每个视频的 clip 概率日志输出目录")

    parser.add_argument("--out", required=True, help="结果输出路径(.json / .csv)")
    args = parser.parse_args()

    assert args.video or args.video_dir, "至少给一个 --video 或 --video-dir"

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
            target_fps=args.target_fps,
            time_window_sec=args.time_window_sec,
            window_stride_sec=args.window_stride_sec,
            topk=args.topk,
            prob_log_jsonl=(str(Path(args.prob_log_dir) / f"{Path(args.video).stem}_prob.jsonl")
                            if args.prob_log_dir else None),
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
        target_fps=args.target_fps,
        time_window_sec=args.time_window_sec,
        window_stride_sec=args.window_stride_sec,
        topk=args.topk,
        prob_log_dir=args.prob_log_dir,
    )
    save_results(results, args.out)


if __name__ == "__main__":
    main()

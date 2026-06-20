# Artifact Manifest

This repository now tracks the useful training artifacts needed to restore the current trained model state:

```text
work_dirs/posec3d_fall_binary/
```

Tracked model files:

```text
work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth
work_dirs/posec3d_fall_binary/epoch_22.pth
work_dirs/posec3d_fall_binary/epoch_23.pth
work_dirs/posec3d_fall_binary/epoch_24.pth
```

The same directory also includes training logs, test logs, prediction pickles, config snapshots, and evaluation plots from the completed PoseConv3D run.

Not tracked in normal Git:

```text
data/
outputs/
yolo26x-pose.pt
mmaction2_src/
vis/
```

Reasons:

* `data/ntu120_2d.pkl` and `data/ntu60_2d.pkl` are large upstream dataset artifacts.
* `outputs/` contains reproducible overlay videos and generated logs.
* `yolo26x-pose.pt` is a downloadable third-party pose model and exceeds GitHub's normal 100 MB single-file limit.
* `mmaction2_src/` is external source code that can be restored from its upstream project.

Local backup note:

```text
D:\AAA\基于深度学习的视频动作识别技术研究\fall_detection_git_artifacts_20260620.tar.gz
```

This local tarball was copied from the server on 2026-06-20. It contains the selected training run, real test videos, the binary training pkl, YOLO pose weights, and key JSON/JSONL inference results. It is intentionally not committed to Git.

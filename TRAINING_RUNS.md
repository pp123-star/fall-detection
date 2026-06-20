# Training Runs

This file records completed training runs and the model artifacts that must be kept for later maintenance, comparison, and deployment.

## Artifact Retention Rule

After each training run finishes, do not delete the final saved model artifacts unless there is an explicit replacement/cleanup decision recorded here.

For each run, keep at least:

* The `best_*.pth` checkpoint selected by validation metrics.
* The final saved epoch checkpoints kept by the training config.
* The prediction pickle files generated during testing, such as `*_pred.pkl`.
* The metric summaries, confusion matrix images, ROC/PR outputs, and error lists.
* The training log directory and the config copy under `work_dirs/`.

Current training config uses:

```python
checkpoint=dict(
    interval=1,
    save_best="acc/top1",
    rule="greater",
    max_keep_ckpts=3,
    save_last=True,
)
```

This means normal `epoch_*.pth` files are automatically limited to the latest 3 checkpoints, while the best checkpoint is kept separately. If future runs need to preserve more than the final 3 normal checkpoints, archive them immediately after training before starting another run.

## Run Template

```text
Run ID:
Date/time:
Operator:
Server/workspace:
Code commit:
Model/config:
Dataset:
Train split:
Validation/test split:
Command:
Environment notes:
Checkpoints kept:
Evaluation command:
Evaluation result:
Artifacts:
Conclusion:
Follow-up:
```

## 2026-06-20 - PoseConv3D Fall Binary Baseline

Run ID: `20260620_posec3d_fall_binary`

Date/time:

* Started: 2026-06-20 00:52 server time
* Finished: 2026-06-20 03:30 server time
* Evaluation finished: 2026-06-20 03:39 server time

Operator: Codex

Server/workspace:

```text
/root/autodl-tmp/fall-detection
```

Code commit after local/GitHub worklog update:

```text
14f27fa Record PoseConv3D training evaluation
```

Model/config:

```text
configs/posec3d_fall_binary.py
```

Main model:

```text
Recognizer3D + ResNet3dSlowOnly backbone + I3DHead
```

Dataset:

```text
data/fall_binary_xsub.pkl
```

Train split:

```text
xsub_train
```

Validation/test split used by current project config:

```text
xsub_val
```

Training command actually used:

```bash
cd /root/autodl-tmp/fall-detection
source /root/miniconda3/etc/profile.d/conda.sh
conda activate falldet
export OMP_NUM_THREADS=1
export PYTHONPATH=/root/autodl-tmp/fall-detection/mmaction2_src:$PYTHONPATH
python mmaction2_src/tools/train.py configs/posec3d_fall_binary.py --seed 42
```

Environment notes:

* GPU: NVIDIA GeForce RTX 4090.
* Environment: conda `falldet`.
* `importlib-metadata` was installed before training because the environment missed it.
* Training kept `--seed 42`.
* `--deterministic` was not used because the current PyTorch/CUDA stack reported unsupported deterministic CUDA behavior for PoseConv3D pooling backward.
* No technical documentation, model config, dataset file, or training logic was modified for this run.

Checkpoints kept:

```text
work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth
work_dirs/posec3d_fall_binary/epoch_22.pth
work_dirs/posec3d_fall_binary/epoch_23.pth
work_dirs/posec3d_fall_binary/epoch_24.pth
```

Checkpoint verification:

```bash
python tools/verify_best_ckpt.py work_dirs/posec3d_fall_binary
```

Verification result:

* `best_acc_top1_epoch_5.pth` matches the first highest validation `acc/top1=1.0000`.
* Epoch 22, 23, and 24 also reached validation `acc/top1=1.0000`, but `save_best` uses `rule="greater"`, so tied results did not replace the epoch 5 best checkpoint.

Evaluation commands:

```bash
bash tools/test.sh configs/posec3d_fall_binary.py <checkpoint>
python tools/eval_binary_metrics.py --pred <pred.pkl> --config configs/posec3d_fall_binary.py --out-dir <eval_dir> --save-errors
```

Evaluation dataset size:

```text
Total samples: 825
Fall samples: 275
Non-fall samples: 550
```

Evaluation result summary:

| Checkpoint | Accuracy | Precision | Recall | Specificity | F1 | ROC AUC | PR AUC | Default-threshold errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `best_acc_top1_epoch_5.pth` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `epoch_22.pth` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| `epoch_23.pth` | 0.9988 | 1.0000 | 0.9964 | 1.0000 | 0.9982 | 1.0000 | 1.0000 | 1 FN |
| `epoch_24.pth` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |

Epoch 23 default-threshold error:

```text
sample_130, gt=1, pred=0, fall_score=0.4759, FN
```

Artifacts:

```text
work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5_pred.pkl
work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5_metrics.txt
work_dirs/posec3d_fall_binary/eval_best_acc_top1_epoch_5/

work_dirs/posec3d_fall_binary/epoch_22_pred.pkl
work_dirs/posec3d_fall_binary/epoch_22_metrics.txt
work_dirs/posec3d_fall_binary/eval_epoch_22/

work_dirs/posec3d_fall_binary/epoch_23_pred.pkl
work_dirs/posec3d_fall_binary/epoch_23_metrics.txt
work_dirs/posec3d_fall_binary/eval_epoch_23/

work_dirs/posec3d_fall_binary/epoch_24_pred.pkl
work_dirs/posec3d_fall_binary/epoch_24_metrics.txt
work_dirs/posec3d_fall_binary/eval_epoch_24/
```

Conclusion:

* This run is usable as the current PoseConv3D baseline.
* Recommended deployment/testing checkpoint: `work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth`.
* Also keep `work_dirs/posec3d_fall_binary/epoch_24.pth` as the final-epoch comparison checkpoint.
* The result is excellent on the current project split, but it should not be treated as full real-world generalization proof because evaluation still uses `xsub_val` from the current dataset package.

External real-video smoke test:

* Date: 2026-06-20.
* Videos: `data/real_test/test4.mp4` to `data/real_test/test7.mp4`.
* User confirmation: all 4 videos contain fall actions.
* Inference output directory on server:

```text
/root/autodl-tmp/fall-detection/outputs/real_test_overlay_test4567_20260620_044418
```

| Video | Ground truth | Detection | Max/alert P(fall) | Note |
| --- | --- | --- | ---: | --- |
| `test4.mp4` | fall | missed | NA | No alert; max probability was not logged by the current summary |
| `test5.mp4` | fall | detected | 0.7350 | Alert at frame 186, track 2 |
| `test6.mp4` | fall | detected | 0.6244 | Alert at frame 223, track 1 |
| `test7.mp4` | fall | missed | NA | No alert; max probability was not logged by the current summary |

Real-video smoke-test result:

```text
Detected: 2 / 4
Missed:   2 / 4
```

Interpretation:

* The trained checkpoint works technically, but real phone videos reveal deployment/generalization gaps.
* This does not invalidate the training result; it identifies the next improvement target.
* Likely factors include short fall duration, about-60fps source videos making `clip_len=48` cover only about 0.8 seconds, vertical high-resolution framing, pose quality, and track stability.
* Visual inspection confirms that `test4.mp4` and `test7.mp4` are hard positive samples:
  * `test4.mp4` is an ice-slip fall with black winter clothing, hood occlusion, hand-support/half-sitting motion, reflective ice, and a fall pattern that differs from a standard indoor fall.
  * `test7.mp4` is a very short night snow-scene fall, mostly back-facing, with bulky clothing, motion blur, and limited stable post-fall visibility.
* `test5.mp4` and `test6.mp4` are easier positives because the person remains more visible and the fall/post-fall posture is clearer.
* Current real-video summary logs only alert events. For missed videos, `NA` means the current summary did not record max probability, not that the model necessarily assigned zero fall probability.

Follow-up:

* Add independent real videos or camera-captured samples for external testing.
* Add hard negative samples: sitting down quickly, lying down normally, bending, squatting, jumping, exercising, occlusion, poor lighting, camera shake, and multi-person scenes.
* If external testing exposes false positives or false negatives, fine-tune or retrain with those samples included.
* Before future training runs, decide whether to archive more than the latest 3 normal checkpoints. The current config only keeps `best` plus the latest 3 normal epoch checkpoints.
* For real-video inference, test 30fps resampling or time-based clip sampling so the action window covers enough real time.
* Try more sensitive deployment thresholds/alert settings before retraining, then use confirmed misses such as `test4.mp4` and `test7.mp4` as real hard-positive samples for fine-tuning.
* Add per-inference probability logging for real videos, including max/mean/top-k `P(fall)`, so missed videos can be diagnosed quantitatively instead of relying only on alert events.

Real-video diagnostic rerun after inference upgrade:

* Date: 2026-06-20.
* Code commit on server:

```text
e56cccd Use local MMAction2 source for inference
```

* Command family: `tools/run_real_video_eval.py` with time-window buffer, track merge, multi-policy alerting, probability logging, and summary output.
* Parameters:

```text
checkpoint: work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth
time_window_sec: 1.6
track_merge: true
threshold: 0.45
high_thr: 0.7
topk_mean_thr: 0.5
infer_every: 2
max_persons: 5
```

* Server output directory:

```text
/root/autodl-tmp/fall-detection/outputs/real_eval/test4567_recommended_20260620_154749
```

* Output artifacts include `summary.csv`, `failure_cases.csv`, `metrics.json`, per-video overlays, per-inference probability logs, per-video summaries, snapshots, and probability curve PNGs.

| Video | Ground truth | Diagnosis | Detection | Max P(fall) | Mean top5 P(fall) | Event |
| --- | --- | --- | --- | ---: | ---: | --- |
| `test4.mp4` | fall | `model_unaware` | missed | 0.1134 | 0.0496 | none |
| `test5.mp4` | fall | `detected` | detected | 0.9987 | 0.9979 | frame 182, track 2, event P=0.7711 |
| `test6.mp4` | fall | `detected` | detected | 0.9998 | 0.9997 | frame 213, track 1, event P=0.5675 |
| `test7.mp4` | fall | `model_unaware` | missed | 0.0569 | 0.0504 | none |

Metric summary:

```text
TP=2
FP=0
TN=0
FN=2
accuracy=0.5000
precision=1.0000
recall=0.5000
f1=0.6667
```

Interpretation:

* The upgraded inference pipeline technically works and now logs quantitative probabilities for missed videos.
* `test5.mp4` and `test6.mp4` remain clear detections, with max `P(fall)` near 1.0.
* `test4.mp4` and `test7.mp4` remain missed even after a 1.6-second time window and more sensitive alert policy; their max probabilities are very low, so they should be treated as `model_unaware` hard positives rather than simple threshold misses.
* Next model-improvement step should be fine-tuning with real hard-positive samples such as `test4.mp4` and `test7.mp4`, plus hard negatives. When starting any future training/fine-tuning run on the server, use `screen` so the web terminal can attach to the progress, for example `screen -S falldet-finetune` and `screen -x falldet-finetune`.

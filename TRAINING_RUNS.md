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

Follow-up:

* Add independent real videos or camera-captured samples for external testing.
* Add hard negative samples: sitting down quickly, lying down normally, bending, squatting, jumping, exercising, occlusion, poor lighting, camera shake, and multi-person scenes.
* If external testing exposes false positives or false negatives, fine-tune or retrain with those samples included.
* Before future training runs, decide whether to archive more than the latest 3 normal checkpoints. The current config only keeps `best` plus the latest 3 normal epoch checkpoints.

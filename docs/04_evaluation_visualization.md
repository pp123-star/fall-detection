# 04 评估与可视化

> 这份文档目标:**论文 4.x 节的每张图、每个表都能从这里查到怎么出**。

## 一、二分类核心指标(论文 4.1 表 X)

`tools/eval_binary_metrics.py` 是一站式工具,跑一次得到:

| 输出 | 文件 | 论文位置 |
|---|---|---|
| Acc / Precision / Recall / F1 | `metrics.json` | 4.1 主对比表 |
| ROC 曲线 + AUC | `roc_curve.png` | 4.1 图 |
| PR 曲线 + AP | `pr_curve.png` | 4.1 图 |
| 混淆矩阵 | `confusion_matrix.png` | 4.1 图 |
| 阈值扫描(找最佳 F1 阈值) | `threshold_sweep.png` + `metrics.json["best_threshold"]` | 4.1 图 |
| 错误样本列表 | `errors.csv`(False Positive + False Negative) | 4.6 失败案例分析 |

跑法(假设 test.py 已 dump 了 `pred.pkl`):

```bash
python tools/eval_binary_metrics.py \
    --pred work_dirs/posec3d_fall_binary/pred.pkl \
    --config configs/posec3d_fall_binary.py \
    --out-dir work_dirs/posec3d_fall_binary/eval
```

`tools/test.sh` 第三个参数就是 dump 路径:

```bash
bash tools/test.sh configs/posec3d_fall_binary.py \
    work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18.pth \
    work_dirs/posec3d_fall_binary/pred.pkl
```

---

## 二、训练曲线(论文 4.x 图)

```bash
mkdir -p figs

# 4 合 1 图(train_loss + val_loss + val_acc + lr),论文附录用
python tools/plot_curves.py \
    --work-dirs work_dirs/posec3d_fall_binary work_dirs/stgcnpp_fall_binary \
    --labels PoseConv3D ST-GCN++ \
    --out figs/training_curves_all.png

# 论文正文用:就一张 val acc 对比图
python tools/plot_curves.py \
    --work-dirs work_dirs/posec3d_fall_binary work_dirs/stgcnpp_fall_binary \
    --labels PoseConv3D ST-GCN++ \
    --metric acc \
    --out figs/val_acc_compare.pdf
```

输出最后会打印一份"最佳 acc 摘要",可直接抄进表。

---

## 三、主对比实验(E1:PoseConv3D vs ST-GCN++)

```bash
# 两个模型都训完后:
for tag in posec3d_fall_binary stgcnpp_fall_binary; do
    CKPT=$(ls -t work_dirs/${tag}/best_acc_top1_*.pth | head -1)
    bash tools/test.sh configs/${tag/_fall_binary/}_fall_binary.py "$CKPT" \
        work_dirs/${tag}/pred.pkl
    python tools/eval_binary_metrics.py \
        --pred work_dirs/${tag}/pred.pkl \
        --config configs/${tag/_fall_binary/}_fall_binary.py \
        --out-dir work_dirs/${tag}/eval
done

# 用一个小脚本把两个 metrics.json 合并成对比表:
python - <<'PY'
import json, glob
rows = []
for f in sorted(glob.glob("work_dirs/*/eval/metrics.json")):
    m = json.load(open(f))
    name = f.split("/")[1]
    rows.append((name, m["acc"], m["precision"], m["recall"], m["f1"],
                 m["roc_auc"], m["pr_auc"]))
print(f"{'Model':30s} {'Acc':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'ROC':>7s} {'PR':>7s}")
for r in rows:
    print(f"{r[0]:30s} " + " ".join(f"{x:7.4f}" for x in r[1:]))
PY
```

这个表直接抄到论文 4.1 即可。

---

## 四、消融实验(E2/E3/E4)

每个消融跑完都有独立 work_dir,用 `plot_curves.py` 把曲线画在一起。

### E2:负样本策略对比

```bash
# 三种策略各训一份(见 docs/03 第六节)
# 评估时不仅看主指标,还要单独算"对易混淆类的召回":
python tools/eval_binary_metrics.py \
    --pred work_dirs/posec3d_fall_hard/pred.pkl \
    --config configs/posec3d_fall_binary.py \
    --out-dir work_dirs/posec3d_fall_hard/eval

python tools/plot_curves.py \
    --work-dirs work_dirs/posec3d_fall_hard \
                work_dirs/posec3d_fall_random \
                work_dirs/posec3d_fall_mixed \
    --labels "Hard Negatives" "Random Negatives" "Mixed" \
    --metric acc \
    --out figs/ablation_negatives.pdf
```

### E3:数据量消融

```bash
python tools/plot_curves.py \
    --work-dirs work_dirs/posec3d_fall_sub{1.0,0.5,0.25,0.1} \
    --labels 100% 50% 25% 10% \
    --metric acc \
    --out figs/ablation_data.pdf
```

论文里一般还配一张 "data ratio vs final F1" 的折线图,自己用 metrics.json 出:

```python
import json, matplotlib.pyplot as plt
ratios = [1.0, 0.5, 0.25, 0.1]
f1s = []
for r in ratios:
    m = json.load(open(f"work_dirs/posec3d_fall_sub{r}/eval/metrics.json"))
    f1s.append(m["f1"])
plt.plot([r*100 for r in ratios], f1s, marker="o")
plt.xlabel("Training Data Ratio (%)")
plt.ylabel("F1 Score")
plt.grid(alpha=0.3)
plt.savefig("figs/ablation_data_f1.pdf", dpi=200, bbox_inches="tight")
```

### E4:输入帧数消融

```bash
python tools/plot_curves.py \
    --work-dirs work_dirs/posec3d_clip{32,48,64} \
    --labels "T=32" "T=48" "T=64" \
    --metric acc \
    --out figs/ablation_cliplen.pdf
```

---

## 五、跨数据集泛化(E5:NTU → URFD)

**为什么要做**:NTU 是演员摆拍的实验室数据,真实场景遮挡多、相机角度乱、跌倒方式随机。论文里这一项是泛化能力的硬证据。

### 5.1 准备 URFD 数据

```bash
mkdir -p data/raw/urfd && cd data/raw/urfd
# URFD 摔倒序列(30 段)
wget http://fenix.ur.edu.pl/mkepski/ds/uf/fall-01-cam0-rgb.zip
# ...其余 29 段
# 非摔倒(ADL,40 段)
wget http://fenix.ur.edu.pl/mkepski/ds/uf/adl-01-cam0-rgb.zip
# ...
# 解压后是 PNG 帧序列,需先合成 mp4
for d in fall-*-cam0-rgb adl-*-cam0-rgb; do
    ffmpeg -framerate 30 -i "${d}/%06d.png" -c:v libx264 -pix_fmt yuv420p "${d}.mp4"
done
cd -
```

### 5.2 准备标签 CSV

```csv
video,label
fall-01-cam0-rgb.mp4,1
fall-02-cam0-rgb.mp4,1
...
adl-01-cam0-rgb.mp4,0
adl-02-cam0-rgb.mp4,0
...
```

存为 `data/raw/urfd_labels.csv`。

### 5.3 批量推理(直接套训练好的模型)

```bash
python inference/batch_predict.py \
    --video-dir data/raw/urfd/ \
    --label-csv data/raw/urfd_labels.csv \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18.pth \
    --aggregate max \
    --threshold 0.5 \
    --out preds/urfd_posec3d.csv
```

结束时会自动打印 `Acc/P/R/F1`。论文这部分通常是:
- NTU val:Acc 0.96, F1 0.95(实验室)
- URFD:Acc 0.78, F1 0.75(真实场景下掉)
- 这个 gap 本身就是论文里的讨论点 → 6.x 节"局限与未来工作"

### 5.4 阈值敏感性分析(可选)

跑一遍把所有 clip 概率存下来,在 numpy 里扫不同阈值:

```python
import csv, numpy as np
from collections import defaultdict

# 加载 preds/urfd_posec3d.csv 里的 clip 概率(若 batch_predict 输出 JSON 形式)
# 在多个阈值下重算 F1,画图
```

---

## 六、姿态估计源对比(E6,可选加分项)

论文亮点之一:**训练时用 OpenMMLab HRNet 提取的骨骼,部署时用 YOLO26-Pose 提取**——这两套关键点是否一致?

```bash
# 6.1 用 YOLO26-Pose 重新提一遍 NTU 验证集(只跑 1000 段省时间)
python inference/extract_pose_yolo26.py \
    --video data/ntu_videos/<某段视频>.avi \
    --out poses/<某段视频>.pkl

# 6.2 跑一遍批量预测(把 YOLO26-Pose 提取的骨骼喂给 PoseConv3D)
# (此场景需要扩展 batch_predict 接受 pose pkl,这个 TODO 留给你按需写)
```

论文里:
- 训练时骨骼源 = HRNet(高精度但慢)
- 推理时骨骼源 = YOLO26-Pose(略低精度但快 5×+)
- 跨骨骼源测试 F1 下降几个百分点 → "工程权衡"讨论点

---

## 七、失败案例分析(论文 4.6)

`eval_binary_metrics.py` 会输出 `errors.csv`,每行是一个错分样本:
- `frame_dir`(NTU 命名能反推动作类:S001C001P001R001A007 → A007 = sit down)
- `gt_label` / `pred_label`
- `prob` 模型给出的摔倒概率

跑下面这段把错分按"原始动作类别"分组,看模型最容易把哪个类误判成摔倒:

```python
import pandas as pd
df = pd.read_csv("work_dirs/posec3d_fall_binary/eval/errors.csv")
# NTU 视频名末尾 A### 是动作类
df["action"] = df["frame_dir"].str.extract(r"A(\d+)").astype(int)
print(df.groupby(["action", "pred_label"]).size().unstack(fill_value=0))
```

在论文里写一段:"模型最常把 A007(sit down)、A041(staggering)误判为摔倒,这与人类标注难度一致……"

---

## 八、可视化:成功 / 失败案例的可视频帧

论文 4.6 节通常要放几张图,演示模型在啥情况下成 / 啥情况下败。

```bash
# 给一个具体视频片段画上骨骼 + 概率条 + 警报横幅
python inference/realtime_demo.py \
    --source data/raw/urfd/fall-05-cam0-rgb.mp4 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18.pth \
    --save-out figs/case_success.mp4 \
    --no-show

# 失败案例同理,挑一个 errors.csv 里的样本
```

录完 mp4 后用 ffmpeg 截关键帧:

```bash
ffmpeg -i figs/case_success.mp4 -vf "select=eq(n\,0)+eq(n\,30)+eq(n\,60)+eq(n\,90)" \
       -vsync vfr figs/case_success_%02d.png
```

---

## 九、效率分析(论文 4.5)

```bash
# 录制 realtime_demo,过程中会持续打印 FPS
python inference/realtime_demo.py \
    --source 你的测试视频.mp4 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best.pth \
    --save-out figs/demo.mp4 --no-show 2>&1 | tee figs/demo_log.txt

# 末尾打印平均 FPS 和 infer ms,即可填论文表:
#   端到端 FPS:30+(RTX 4090)
#   单 clip 分类延迟:~25ms
#   YOLO26-Pose 单帧延迟:~12ms
```

不同分类频率(`--infer-every`)下记 FPS,画一张"延迟 vs 准确率"折中图。

---

## 十、论文图表清单速查

| 论文位置 | 数据来源 | 出图命令 |
|---|---|---|
| 4.1 主对比表 | `work_dirs/*/eval/metrics.json` | 见本文第三节 |
| 4.1 ROC/PR/混淆矩阵 | `eval_binary_metrics.py` 自动出 | — |
| 4.1 训练曲线 | `plot_curves.py` | 第二节 |
| 4.2 负样本消融 | E2 三个 work_dir | 第四节 E2 |
| 4.3 数据量消融 | E3 多个 work_dir | 第四节 E3 |
| 4.4 帧数消融 | E4 多个 work_dir | 第四节 E4 |
| 4.5 跨数据集 | URFD 批量推理 | 第五节 |
| 4.5 效率分析 | realtime_demo 日志 | 第九节 |
| 4.6 失败案例 | `errors.csv` + realtime_demo | 第七、八节 |

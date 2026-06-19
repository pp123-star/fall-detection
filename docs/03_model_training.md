# 03 模型训练

## 一、训练前必做的最后三件事

> 跳过任何一步都可能让训练白跑,上一版项目踩过的坑都从这里开始。

### 1. 关键点对齐人工核验

```bash
python data_prep/visualize_skeleton.py --pkl data/fall_binary_xsub.pkl --num 5
# 输出在 vis/skeleton_check/ 下,5 个 mp4
```

打开 mp4 用肉眼检查:
- 头部三角形(鼻、左右眼、左右耳)是否连成倒三角
- 肩到肘到腕这条线是否对应人臂走向
- 髋到膝到踝是否对应腿

**如果出现"鼻子连到脚踝"之类的乱线,关键点顺序就错了,立刻停下来检查**。
COCO 17 点顺序参考 `inference/pose_to_pyskl_format.py` 文件头。

### 2. 训练/验证划分泄漏检查

```bash
python data_prep/split_check.py --pkl data/fall_binary_xsub.pkl
```

输出 5 项 PASS / FAIL:
- 样本名是否重叠
- 受试者 P 字段是否重叠 ← 这一项 FAIL 就是泄漏
- 标签分布是否合理(避免 99%/1% 极端不平衡)
- 关键点坐标范围是否在 img_shape 内
- 摔倒/非摔倒动作类别分布

**任何一项 FAIL 都不要训**。X-Sub 标准划分:训练 = P001 P002 ... 共 20 人,验证 = 剩下 20 人。

### 3. 显存预算确认

| 模型 | batch_size | clip_len | 显存占用 | 推荐显卡 |
|---|---|---|---|---|
| PoseConv3D | 16 | 48 | ~14 GB | RTX 4090 / 3090 |
| PoseConv3D | 8 | 48 | ~8 GB | RTX 4070 / 3080 |
| ST-GCN++ | 32 | 100 | ~6 GB | 任何 8 GB+ 卡 |

显存不够时:在 config 里把 `train_dataloader.batch_size` 减半,把 `optimizer.lr` 也减半,基本无损。

---

## 二、训练命令

### 主线:PoseConv3D

```bash
# 单卡
bash tools/train.sh configs/posec3d_fall_binary.py 1

# 4 卡(线性放大 lr 时同时改 config 里的 lr)
bash tools/train.sh configs/posec3d_fall_binary.py 4
```

预期时长:RTX 4090 单卡 ~1.5-2h,24 epochs。

输出:`work_dirs/posec3d_fall_binary/`
- `epoch_*.pth` — 每 epoch ckpt(最多保留 3 个,省盘)
- `best_acc_top1_epoch_*.pth` — 验证集 top-1 acc 最高的那个 ← **论文里用这个**
- `last_checkpoint` — 文本文件,记录最后一个 ckpt 路径(断点续训用)
- `<timestamp>.log` — 训练日志
- `<timestamp>/vis_data/scalars.json` — 用于 plot_curves.py

### 对比:ST-GCN++

```bash
bash tools/train.sh configs/stgcnpp_fall_binary.py 1
```

预期时长:RTX 4090 单卡 ~30min,16 epochs。

### 一键串联

```bash
# 跑完整流程(数据 → 训练两个模型 → 评估 → 出图)
bash scripts/run_all.sh

# 已搭好环境,只想重跑训练
bash scripts/run_all.sh --skip-env

# 只跑 ST-GCN++ 先看个数(快)
bash scripts/run_all.sh --skip-env --stgcn-only
```

---

## 三、训练过程监控

### 3.1 实时看日志

```bash
# 在另一个 SSH 窗口
tail -f work_dirs/posec3d_fall_binary/$(ls -t work_dirs/posec3d_fall_binary | grep -E '^[0-9]+_[0-9]+' | head -1).log
```

关键指标:
- `loss`:稳定下降,**第一个 epoch 末应该已经 < 0.5**;不下降基本是数据/学习率有问题
- `acc/top1`:验证准确率,二分类基线 0.5,**最终应该 > 0.95**;如果只到 0.6-0.7 说明负样本太难或模型没收敛
- `lr`:CosineAnnealing,从初始 lr 慢慢降到接近 0

### 3.2 TensorBoard(可选)

在 `configs/_base_/default_runtime.py` 里取消注释 `TensorboardVisBackend`,然后:

```bash
pip install tensorboard
tensorboard --logdir work_dirs/ --bind_all --port 6006
# 浏览器开 http://云GPU实例IP:6006
```

### 3.3 GPU 利用率

```bash
watch -n 1 nvidia-smi
```

PoseConv3D 训练时 GPU util 应该稳定在 90%+;
如果反复掉到 30-50%,大概率是数据加载瓶颈,改 config:
```python
train_dataloader = dict(
    num_workers=8,           # 4090 实例通常 8 核,设 8
    persistent_workers=True, # 加这个
    ...
)
```

---

## 四、断点续训

云 GPU 实例被强制关机或抢占时:

```bash
# 自动从最新 ckpt 续训
bash tools/train.sh configs/posec3d_fall_binary.py 1 --resume

# 或手动指定
bash tools/train.sh configs/posec3d_fall_binary.py 1 \
    --cfg-options "resume_from=work_dirs/posec3d_fall_binary/epoch_12.pth"
```

`train.sh` 内部用了 `--resume`,会读 `last_checkpoint` 文件,优化器状态、lr scheduler、epoch 全部恢复。

---

## 五、超参调优(对最终结果敏感的几个旋钮)

| 参数 | 默认 | 调高 | 调低 |
|---|---|---|---|
| `total_epochs` | 24 (PoseConv3D) / 16 (ST-GCN++) | 训练更充分,但易过拟合 | 欠拟合 |
| `train_dataloader.batch_size` | 16 / 32 | 提速,需更多显存,lr 同步放大 | 显存够用时建议保持 |
| `optimizer.lr` | 0.2 / 0.1 | 收敛快但易震荡 | 稳但慢 |
| `train_pipeline` 的 `clip_len` | 48 | 时序信息更全,显存涨 | 显存省,但摔倒动作可能截不全 |
| `train_dataset.times` (RepeatDataset) | 10 | 每 epoch 看更多数据,降低 IO 抖动 | 训练快但不稳 |
| `cls_head.dropout_ratio` | 0.5 | 过拟合时调高 | 欠拟合时调低 |

**论文里要做消融的旋钮(我已经预留接口)**:
- `--clip-len 32/48/64`:输入帧数 → 论文 E4
- `--neg-strategy hard/random/mixed`:负样本策略 → 论文 E2
- `--subsample-ratio 0.5/0.25/0.1`:训练数据量 → 论文 E3

---

## 六、消融实验配置(论文 4.x 节)

每个消融跑完都换个 `work_dir`,免得覆盖主结果。

### E2:困难负样本 vs 随机负样本

```bash
# 困难负样本(默认,推荐)
python data_prep/build_binary_pkl.py \
    --src data/ntu60_2d.pkl \
    --dst data/fall_binary_hard.pkl \
    --neg-strategy hard --neg-pos-ratio 3

# 纯随机负样本
python data_prep/build_binary_pkl.py \
    --src data/ntu60_2d.pkl \
    --dst data/fall_binary_random.pkl \
    --neg-strategy random --neg-pos-ratio 3

# 各训一次,对比两个 work_dir 的 val acc
# 训练时用 --cfg-options 覆盖 ann_file 路径:
bash tools/train.sh configs/posec3d_fall_binary.py 1 \
    --cfg-options "ann_file=data/fall_binary_random.pkl" \
                  "work_dir=work_dirs/posec3d_fall_random"
```

### E3:训练数据量消融

```bash
for ratio in 1.0 0.5 0.25 0.1; do
    python data_prep/build_binary_pkl.py \
        --src data/ntu60_2d.pkl \
        --dst "data/fall_binary_sub${ratio}.pkl" \
        --subsample-ratio "$ratio"
    bash tools/train.sh configs/posec3d_fall_binary.py 1 \
        --cfg-options "ann_file=data/fall_binary_sub${ratio}.pkl" \
                      "work_dir=work_dirs/posec3d_fall_sub${ratio}"
done
```

### E4:输入帧数消融

```bash
for clen in 32 48 64; do
    bash tools/train.sh configs/posec3d_fall_binary.py 1 \
        --cfg-options "train_pipeline.0.clip_len=${clen}" \
                      "val_pipeline.0.clip_len=${clen}" \
                      "test_pipeline.0.clip_len=${clen}" \
                      "work_dir=work_dirs/posec3d_clip${clen}"
done
```

---

## 七、训练出问题的快速诊断

| 现象 | 可能原因 | 排查 |
|---|---|---|
| Loss 不下降,一直 0.69 左右 | 数据全是 0(关键点未加载) | 跑 `data_prep/split_check.py` |
| 第 1 epoch acc 就 0.95+,但 val 测试很差 | 训练/验证集泄漏 | 跑 `split_check.py`,看 P-overlap |
| 训到一半 OOM | 显存不够 / 数据 worker 内存爆炸 | 减 batch / 减 num_workers |
| Loss 跳到 NaN | lr 太高 | 把 lr 减半重训 |
| 第 N epoch 后 val acc 不再涨 | 学习率太高 / 过拟合 | 让 cosine 自然衰减;或减 epochs |
| 单卡训练慢得离谱(< 50% GPU util) | 数据加载瓶颈 | num_workers=8, persistent_workers=True |
| 多卡比单卡还慢 | NCCL / 通信瓶颈 | 检查 `dist_cfg` 后端 |

---

## 八、训练完成后的"自检三连"

```bash
# 1. checkpoint 完整性(防止上一版的"best 没保存"bug 复发)
python tools/verify_best_ckpt.py --work-dir work_dirs/posec3d_fall_binary

# 2. 验证集上跑 test 拿到 pred pickle
bash tools/test.sh configs/posec3d_fall_binary.py \
    work_dirs/posec3d_fall_binary/best_acc_top1_epoch_*.pth \
    work_dirs/posec3d_fall_binary/pred.pkl

# 3. 出二分类细致指标(F1 / ROC / PR / 混淆矩阵)
python tools/eval_binary_metrics.py \
    --pred work_dirs/posec3d_fall_binary/pred.pkl \
    --config configs/posec3d_fall_binary.py \
    --out-dir work_dirs/posec3d_fall_binary/eval
```

下一步去看 `04_evaluation_visualization.md`,把论文图表都出齐。

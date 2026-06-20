# 08 真实视频推理与诊断

> 本文是项目继 `docs/07` 之后的新增章节。`07` 回答"数据够不够"——结论是 NTU 数据量足够,但真实视频会有 domain gap。本章给出对付那个 gap 的**推理侧改进**与**诊断流程**,以及如果需要**微调**怎么微调。

---

## 0. TL;DR

| 问题(test4/test7 漏检) | 改进 | 默认是否启用 |
|---|---|---|
| 60fps 手机视频,clip_len=48 只覆盖 0.8s(训练时 1.6s) | **时间感知缓冲**:`--time-window-sec 1.6` 让 buffer 跨更多原始帧,均匀采样到 48 | 关 |
| ByteTrack 在快速摔倒发生 ID 切换 | **Track 合并**:`--track-merge` 把刚消失 + 位置接近的新 track 缝合 | 关 |
| 单阈值 0.5 + 连续 K 次的去抖,对短促摔倒不敏感 | **多策略报警**:`--high-thr 0.8`(单次高分即报)+ `--topk-mean-thr 0.55`(最近 5 次 top-3 均值) | 关 |
| 漏检视频不知道是"差 0.01 过阈值"还是"模型全程低分" | **全量概率日志**:`--prob-log xxx.jsonl` 每次推理都记 raw/smoothed | 关 |
| 视频结束只看到报警事件,没诊断信息 | **视频级 summary**:`--summary-json xxx.json` 输出 max/top-k/诊断标签 | 关 |

所有新参数**默认关闭**,不传时行为与旧版 demo 一致。

---

## 1. 为什么 test4/test7 漏检 —— 系统性原因分析

### 1.1 时间窗口不匹配(最关键)

```
训练分布:  NTU60 ≈ 30 fps,clip_len=48  →  1.6 秒动作窗口
推理分布:  手机 60 fps,clip_len=48  →  0.8 秒动作窗口
                                       ↑
                                  只看到摔倒的前一半
```

模型在 NTU 上学到的是"1.6 秒内完成的摔倒模式"。当输入 60fps 视频的 48 帧时,实际只输入了 0.8 秒——摔倒动作还没结束。

**这是分布偏移,不是模型能力问题**。在 NTU val 上得 100% 也救不了这个。

**对应改进**:`TimeAwareBuffer`。buffer 长度按 `source_fps × time_window_sec` 设定(60fps×1.6s=96 帧),推理时再从 96 帧中均匀采样 48 帧喂模型——模型看到的是和训练分布一致的 1.6 秒动作。

### 1.2 ByteTrack 在快速摔倒下的 ID 碎片化(test7)

test7 摔倒瞬间出现 3 个不同 track_id。原因:
- 人形变化快(站立→倾倒→落地),YOLO 框尺寸剧变
- ByteTrack 用 IoU+卡尔曼跟踪,**大形变 IoU 会断**
- 断了之后下一帧 detect 出来,被分配新 ID
- 结果:每个 ID 只持有摔倒的几帧,**buffer 永远不满 48 帧**,根本没机会触发分类

**对应改进**:`TrackMerger`。track 消失时进 tombstones(等待 ≤15 帧),新 track 出现且与 tombstone 的 IoU ≥ 0.3 或中心点归一化距离 ≤ 0.15 时,**继承旧 track 的 buffer + 概率状态**。新 ID 一出现就已经"知道"前面发生了什么。

### 1.3 单阈值 + 去抖对短促事件不友好

旧逻辑:`smoothed_prob > 0.5` 且**连续 2 次**才报警。

考察 test5(成功): P=0.7350 → 连续多次都高,触发
考察 test7(失败): 假设有 1 次 P=0.85,但之前 buffer 不连续,前后都低 → 平滑成 ~0.5 → 不连续 → **被吃掉**

**对应改进**:`AlertPolicy` 三策略并联:
- **high_single**: 单次 raw_prob ≥ 0.8 立即报(快速摔倒救命)
- **consec_mid**: 旧逻辑保留(去抖)
- **topk_mean**: 最近 5 次推理中 top-3 平均 ≥ 0.55 报(短促摔倒)

### 1.4 训练数据形态偏差(test4)

test4 的"半摔、撑地、再滑、上半身画面"在 NTU A43 (falling down) 中几乎不存在。NTU 摔倒是"标准、单次、完整"的。这不是阈值能解决的,是**领域差异**——可能必须微调。

但**有概率日志才能判断**到底是"模型给了 0.4 想到了但没说出口"还是"模型全程低分根本不识别"。前者改阈值/策略,后者必须微调。

### 1.5 诊断能力缺失(根因)

旧 demo 只在报警时记事件。test4/test7 没报警 → 日志里啥都没有 → 我们不知道是哪种情况。

**对应改进**:`ProbabilityLogger`。每次推理都写一行:`{frame_idx, track_id, raw_prob, smoothed_prob, buffer_len, alerted, alert_reason}`。事后用 `tools/plot_prob_curves.py` 画曲线立刻看清。

`VideoSummaryBuilder` 进一步自动给诊断标签:

| diagnosis | 含义 | 改进路径 |
|---|---|---|
| `detected` | 正常报警(GT=1) | ✓ 完成 |
| `just_below_threshold` | max ≥ 0.5 但没触发(策略问题) | 调阈值/启 high_thr |
| `partial_signal` | max ∈ [0.3, 0.5) | 微调或放宽策略 |
| `model_unaware` | max < 0.3,模型全程低分 | **必须微调** |
| `false_alarm` | GT=0 但报警了 | 加困难负样本 |
| `true_negative` | GT=0 没报警 | ✓ 正确 |
| `no_inference` | buffer 始终不满 | 视频太短或姿态全失败 |

---

## 2. 改动概览

```
inference/realtime_core.py            (新增) 复用核心:5 个组件
  ├─ TimeAwareBuffer                  时间感知缓冲
  ├─ TrackMerger                      track 合并(IoU+距离 双信号)
  ├─ AlertPolicy                      三策略报警
  ├─ ProbabilityLogger                每次推理都记
  └─ VideoSummaryBuilder              视频级聚合 + 自动诊断

inference/multitarget_realtime_demo.py  (重写)
  • 集成上面 5 个组件,新参数默认关闭(向后兼容)
  • TrackState 底层换 TimeAwareBuffer(--time-window-sec=0 时等价旧行为)
  • _infer_one 改用 sample_clip 均匀采样
  • 全部修了 Compose / pseudo_collate 两个 bug
  • 备份在 .bak

inference/batch_predict.py             (修改)
  • predict_clip 同步修 bug
  • predict_video 新参数 target_fps / time_window_sec / window_stride_sec
                 / topk / prob_log_jsonl / ground_truth
  • 新增 _build_time_window_clips 离线 time-window 切窗
  • 返回字段新增 max_pfall / top5_pfall / mean_top5_pfall / diagnosis / mode

tools/run_real_video_eval.py           (新增)
  • 一键扫一个目录的所有视频
  • 输出 overlay / events / probs / summaries / snapshots / summary.csv
  • failure_cases.csv 自动给改进建议
  • 支持 labels.csv 算 P/R/F1

tools/plot_prob_curves.py              (新增)
  • 把 prob log 画成 raw + smoothed 曲线
  • 标阈值线、报警点、max 值

docs/08_真实视频推理与诊断.md          (新增,本文)
```

---

## 3. 实验计划(明天就能跑)

建议用 `tools/run_real_video_eval.py` 把每个实验作为独立 out 目录,跑完直接对比 summary.csv。

### 3.0 准备 labels.csv

```bash
cd /root/autodl-tmp/fall-detection
cat > data/real_test/labels.csv <<EOF
video,label
test4.mp4,1
test5.mp4,1
test6.mp4,1
test7.mp4,1
EOF
```

### 3.1 实验 A — Baseline(原参数,作对照)

```bash
python tools/run_real_video_eval.py \
    --video-dir data/real_test --labels-csv data/real_test/labels.csv \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --out-dir outputs/real_eval/A_baseline
```

预期:test5/test6 detected,test4/test7 missed(复现已知结果),但现在 prob log 能让你看到漏检视频实际最高概率多少。

### 3.2 实验 B — 仅时间窗口

```bash
python tools/run_real_video_eval.py \
    --video-dir data/real_test --labels-csv data/real_test/labels.csv \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --out-dir outputs/real_eval/B_timewindow16 \
    --time-window-sec 1.6
```

测试是否仅靠让 buffer 覆盖 1.6 秒就能救回 test4/test7。

### 3.3 实验 C — 推荐组合(time-window + track-merge + 多策略)

```bash
python tools/run_real_video_eval.py \
    --video-dir data/real_test --labels-csv data/real_test/labels.csv \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth \
    --out-dir outputs/real_eval/C_recommended \
    --time-window-sec 1.6 \
    --track-merge \
    --threshold 0.45 --high-thr 0.7 --topk-mean-thr 0.5
```

应该挡住大部分漏检。

### 3.4 实验 D — 更宽窗口(2.0 秒)

```bash
python tools/run_real_video_eval.py \
    ...  --out-dir outputs/real_eval/D_window20 \
    --time-window-sec 2.0 \
    --track-merge \
    --threshold 0.45 --high-thr 0.7 --topk-mean-thr 0.5
```

测试更长窗口是否更好。注意太长可能把摔倒后的"躺地"过多稀释。

### 3.5 实验 E — 阈值扫描

```bash
for THR in 0.3 0.4 0.5; do
    python tools/run_real_video_eval.py \
        ... --out-dir outputs/real_eval/E_thr${THR} \
        --time-window-sec 1.6 --threshold $THR
done
```

跑完看每个 out_dir 的 `metrics.json`,横向对比 Recall vs FP。

### 3.6 实验 F — 单一策略 vs 组合策略

```bash
# F1: 只用 consec_mid (旧默认)
... --time-window-sec 1.6 --out-dir outputs/real_eval/F1_consec_only

# F2: + high_thr
... --time-window-sec 1.6 --high-thr 0.7 --out-dir outputs/real_eval/F2_high

# F3: + topk_mean
... --time-window-sec 1.6 --high-thr 0.7 --topk-mean-thr 0.5 \
    --out-dir outputs/real_eval/F3_full
```

消融 AlertPolicy 三种策略的贡献。

### 3.7 实验 G — 漏检视频画概率曲线(诊断核心)

跑完上面任一实验后:

```bash
python tools/plot_prob_curves.py \
    --prob-log-dir outputs/real_eval/A_baseline/probs \
    --out-dir outputs/real_eval/A_baseline/curves \
    --threshold 0.5 --high-thr 0.7
```

打开 `curves/test4.png` 和 `curves/test7.png`,关键看曲线最高点:

```
case 1: 曲线最高点 ≥ 0.5
        → "差一口气"  → 调阈值/启 topk_mean 或 high_thr 就能救
case 2: 曲线最高点 ∈ [0.3, 0.5)
        → "模型识别到了一点"  → 多策略 + 微调
case 3: 曲线最高点 < 0.3
        → "模型不识别"        → 必须微调(test4 大概率属此类)
case 4: 曲线突然中断,多条曲线交替
        → ID switch          → --track-merge
```

### 3.8 推荐参数表(完整跑过实验之后,起点)

| 场景 | 参数 |
|---|---|
| **真实手机视频(默认推荐)** | `--time-window-sec 1.6 --track-merge --threshold 0.45 --high-thr 0.7 --topk-mean-thr 0.5` |
| 已知 60fps 但摔倒慢 | 加 `--time-window-sec 2.0` |
| 多人场景 | 加 `--max-persons 8` |
| 强误报(怕扰民) | `--threshold 0.5 --high-thr 0.85 --topk-mean-thr 0.6 --alert-k 3` |
| 强查全(医院场景) | `--threshold 0.35 --high-thr 0.6 --topk-mean-thr 0.45 --alert-k 1` |

---

## 4. 诊断流程图

```
test_xxx.mp4 漏检
      │
      ▼
查 outputs/.../summary.csv 中该行
      │
      ├── diagnosis = detected         ✓ 已修复
      │
      ├── diagnosis = just_below_threshold
      │     │  说明:max_pfall ≥ threshold,但报警状态机没过(连续次数不够)
      │     │
      │     ▼  解法:加 --high-thr 或 --topk-mean-thr,无需重训
      │
      ├── diagnosis = partial_signal
      │     │  说明:max_pfall ∈ [0.3, 0.5),模型嗅到一点但不够强
      │     │
      │     ▼  解法(先后顺序):
      │            1. 加 --time-window-sec(若还没加)
      │            2. 加 --track-merge
      │            3. 加 --topk-mean-thr 0.45
      │            4. 还不够 → 微调
      │
      ├── diagnosis = model_unaware
      │     │  说明:max_pfall < 0.3,模型全程低分
      │     │
      │     ▼  解法:必须微调。把这段视频纳入微调集
      │
      ├── diagnosis = no_inference
      │     │  说明:buffer 从没满过 clip_len
      │     │
      │     ▼  原因:视频过短 / 姿态全失败 / target_fps 设置过低
      │
      └── diagnosis = error
            │
            ▼  看 failure_cases.csv 的 stderr_tail
```

---

## 5. 数据集补充建议(按"最容易落地"排序)

### 5.1 ⭐⭐⭐ URFD(Urszula Rzepka Fall Detection)— 最易上手

- **规模**:30 段摔倒 + 40 段日常动作
- **链接**:`http://fenix.ur.edu.pl/mkepski/ds/uf.html`
- **优点**:RGB-D + 加速度计可选,但**只用 RGB 也够**;每段视频独立 mp4,文件名直接给标签(`fall-` vs `adl-`)
- **缺点**:相对老的实验室环境,但已经比 NTU 真实多
- **接入步骤**:
  ```bash
  # 1. 下载 URFD 的 RGB 视频
  # 2. 用项目已有脚本提骨骼
  python inference/extract_pose_yolo26.py --video URFD/fall-01-cam0-rgb.mp4 ...
  # 3. 转项目格式
  python inference/pose_to_pyskl_format.py ...  # 复用现有
  ```

### 5.2 ⭐⭐ Le2i Fall Detection — 中等难度

- **规模**:191 段,4 个场景(home / coffee_room / lecture_room / office)
- **链接**:`http://le2i.cnrs.fr/Fall-detection-Dataset?lang=en`
- **优点**:多场景、真实室内、标注规范
- **缺点**:旧 mp4 编码,有时需要 ffmpeg 转码

### 5.3 ⭐⭐ Multiple Cameras Fall Dataset(Auvinet et al.)

- **规模**:24 个场景,每个 8 个摄像头同步
- **优点**:多视角(可练 view-invariant)
- **缺点**:同一动作 ×8 个视角,**视频数虚高**

### 5.4 ⚠️ SisFall — 视频缺失

- 真实老人摔倒数据,但**只有 IMU 加速度计,没有视频**。骨骼建模项目用不上。

### 5.5 自采集(test4/test7 这种)

每段单独录手机视频,标注成本极低(看一眼就知道是不是摔)。把 test4/test7 直接收为困难正样本

### 5.6 接入流程(任何带视频的数据集)

```
公开数据集 mp4
   │
   ▼  YOLO26-Pose 提取(已有脚本)
   │
   ▼  转换为 MMAction2 PoseDataset 格式(已有 build_sample)
   │
   ▼  打 label(自动:文件名含 'fall'/'adl';或手工 labels.csv)
   │
   ▼  混入 data/fall_binary_xsub.pkl 的 annotations
   │
   ▼  用 docs/03 提到的训练命令微调
```

---

## 6. 真实视频采集规范

| 项目 | 推荐 | 原因 |
|---|---|---|
| **格式** | `.mp4` H.264 | 兼容性最好,OpenCV 直读 |
| **帧率** | **30 fps**(理想) / 60 fps 也可(配 `--time-window-sec`) | 与 NTU 训练分布一致 |
| **时长** | 6–15 秒 | 摔倒前 1-2s + 过程 0.5-2s + 后 2-3s(给模型看到落地静止) |
| **角度** | 侧视 30°-60°,**腰部高度**或略低 | NTU 多侧视;过头顶/过低都偏移 |
| **距离** | 人占画面高度 50-80% | 太小:骨骼噪声;太大:bbox 抖 |
| **入镜** | 全身,头到脚都在框内 | clip_len 短时若上半身缺失会损失关键信息 |
| **方向** | **横屏**优先 | 训练数据基本横屏;竖屏会让人物在框中占比奇怪 |
| **必含** | 摔倒前正常动作 + 摔倒过程 + 落地后静止 ≥ 2s | 模型靠"动作变化"判断,缺前/缺后都不利 |
| **光线** | 均匀,避免逆光 | 不强制要求,YOLO 现在抗光性不错,但夜间会让骨骼噪声变大 |
| **场景** | 单人优先;多人时间隔 ≥ 1m | 多人需要 ID 跟踪稳定 |

**不适合作测试样本的视频**:
- 过曝/严重逆光,人体完全剪影
- 严重遮挡(>50% 身体被障碍物挡)
- 镜头剧烈晃动(走路拍摄)
- 极短(<3 秒) —— buffer 都填不满
- 人体出框严重(超过 1 秒只剩半身)

**如果只能用手机 60fps 拍**:

推理时加这两个参数即可,**不必预先 ffmpeg 降帧**:
```bash
--time-window-sec 1.6
# 如果想模拟 30fps 的等效喂入,buffer 内部已经会均匀采 48 帧覆盖 1.6s
```

如果你想强制走 30fps 视角:
```bash
ffmpeg -i input_60fps.mp4 -r 30 -c:v libx264 -crf 23 input_30fps.mp4
```
但这一步**不是必需**,新版 demo 内部已经做了等效处理。

---

## 7. 何时该微调 — 优先级建议

### 7.1 **先改推理,不立刻重训**

理由:推理改动是 0 训练成本的纯软件修复;如果推理就能救回 test4/test7,根本不需要碰模型。

**做完实验 A → C → G**(参数扫描 + 概率曲线),你会清楚地分类每个漏检视频:

| 分类 | 是否需要微调 |
|---|---|
| `just_below_threshold` / `partial_signal` | **不需要**,调参数即可 |
| `model_unaware` | **需要**,且这就是微调集核心 |

### 7.2 如果决定微调 — 用 best_acc_top1_epoch_5.pth 热启动

> 千万别从头训。NTU 上的特征学习成果别浪费。

**数据组织**:

```
data/fall_finetune.pkl:
  annotations:
    - NTU60 训练集 ×0.8(随机采样 80%,保留语义先验)
    - URFD fall ×30 + ADL ×40(主要 domain gap 填充)
    - test4/test7 ×10(每段重复采样 10 个 clip,做困难正样本)
    - 自录困难负样本(快速坐下/弯腰/捡东西等)×20-50
  splits:
    xsub_train: ...
    xsub_val:   保留 NTU val(让模型证明没破坏 NTU 能力) + 留 20% 真实数据
```

**微调命令**(在 AutoDL):

```bash
cd /root/autodl-tmp/fall-detection
source /root/miniconda3/etc/profile.d/conda.sh
conda activate falldet

python mmaction2_src/tools/train.py configs/posec3d_fall_binary.py \
    --seed 42 \
    --cfg-options \
        "load_from=work_dirs/posec3d_fall_binary/best_acc_top1_epoch_5.pth" \
        "data.train_dataloader.dataset.ann_file=data/fall_finetune.pkl" \
        "optim_wrapper.optimizer.lr=0.01" \
        "param_scheduler.0.eta_min=0.001" \
        "train_cfg.max_epochs=8" \
        "work_dir=work_dirs/posec3d_finetune_real"
```

**参数说明**:
- `load_from=...best_5.pth`:**热启动**,不是 `--resume`。会加载权重但 epoch 计数从 0 开始。
- `lr=0.01`:比从头训(0.4)小一个数量级。已经训好的模型只需小步修正。
- `max_epochs=8`:5-8 即可,再多就是过拟合微调集。
- `work_dir=...posec3d_finetune_real`:输出到新目录,**不要覆盖原 baseline**。

**预期效果**:
- xsub_val 仍能 ≥ 0.95(不破坏 NTU 能力)
- 真实视频 detected 率从 50% → 80%+
- test4/test7 概率曲线明显抬高(就算还不到 0.8,通常 0.5+ 就够 `--high-thr 0.5 --topk-mean-thr 0.45` 救回)

**论文加分**:这一步变成 `4.7 域自适应微调` 章节,你的论文从"NTU baseline + 推理工程"升级为"NTU 预训练 + 真实数据微调 + 工程化部署",层级更高。

### 7.3 如果时间不够,**先不微调**也行

仅靠推理改进(实验 C 的参数)在 NTU 95%+ 不变的前提下,真实视频 recall 通常能从 50% → 75%+。论文里完整讲清楚 domain gap + 推理工程改进,**已经是合格的毕设**。

---

## 8. CSV 字段说明速查

### 8.1 `summary.csv`(`run_real_video_eval.py` 输出)

| 字段 | 含义 |
|---|---|
| `video_name` | 视频文件名 |
| `gt_label` | 0=非摔倒 1=摔倒 (来自 labels.csv) |
| `ok` | 子进程是否成功跑完 |
| `diagnosis` | 自动诊断(见 §4) |
| `num_alerts` | 报警次数(0 = 未报) |
| `max_pfall` | 整段最高 raw 概率(**核心诊断字段**) |
| `mean_top5_pfall` | top-5 平均(辅助诊断"偶发高分 vs 持续中分") |
| `mean_pfall` | 全部推理的平均概率 |
| `num_unique_tracks` | 出现过的 track 数 |
| `num_id_switches_handled` | track 合并次数(>0 = 真实发生过 ID 切换) |
| `suspected_id_switch` | 同上的 bool |
| `total_inferences` | 总推理次数 |
| `total_frames` | 总帧数 |
| `overlay` | overlay mp4 路径 |
| `prob_log` | prob log jsonl 路径 |

### 8.2 `failure_cases.csv`

只列 `diagnosis` 非 `detected` / `true_negative` 的视频。多了 `recommendation` 一列直接给改进建议。

### 8.3 prob log JSONL 每行字段

```json
{"frame_idx": 186, "timestamp": "2026-06-20T05:12:34.567",
 "source": "test5.mp4", "track_id": 2,
 "raw_prob": 0.7350, "smoothed_prob": 0.6821,
 "buffer_len": 48, "bbox_x1": 120.0, "bbox_y1": 50.0,
 "bbox_x2": 380.0, "bbox_y2": 620.0,
 "alerted": true, "alert_reason": "high_single"}
```

---

## 9. 常见问题

**Q: --time-window-sec 设多少合适?**
A: 默认 1.6(等于训练 clip_len/30)。如果摔倒类型比较慢(老人慢慢倒下)可以试 2.0。比 1.6 大不少时要警惕"摔倒后的躺地"占比过多反而稀释信号。

**Q: --track-merge 会不会把两个真不同的人合成一个?**
A: 概率低。默认要求 IoU ≥ 0.3 或归一化距离 ≤ 0.15,且 gap ≤ 15 帧。两个不同的人在 15 帧(0.5 秒@30fps)内位置重合到这个程度,本身就是 ByteTrack 会困惑的场景。可以把 `--track-merge-iou-thr` 调严(如 0.5)缓解。

**Q: --high-thr 设多少?**
A: 0.7-0.85。设 0.7 灵敏但有 FP 风险;设 0.85 保守但可能错过一些。建议从 0.7 起步,看 false_alarm 数。

**Q: prob log 为什么会同时有 raw 和 smoothed?**
A: `raw` 是单次推理直接输出,用来观察"瞬间峰值"——short window 摔倒的关键信号。`smoothed` 是 EMA 平滑,用来做 consec_mid 判定,稳定但滞后。两个都看才能完整诊断。

**Q: 实时摄像头(--source 0)也能用这些新参数吗?**
A: 能。`--time-window-sec 1.6 --track-merge --high-thr 0.7` 对实时摄像头同样有效。摄像头通常 30fps,所以 buffer 就是 48 帧,行为与原版一致。

---

## 10. 下一步

1. **今天剩余时间**:跑实验 A 看看 prob log,直接知道 test4/test7 漏检属于哪种 diagnosis。
2. **明天**:跑实验 C(推荐参数)+ 实验 E(阈值扫描)+ 实验 G(画曲线)。
3. **本周内**:决定是否微调(看实验 G 的 model_unaware 比例)。
4. **下周**:如果决定微调,按 §7.2 走;论文 4.7 节同步成型。

文档结束。代码改动具体见 `inference/realtime_core.py` 顶部注释。

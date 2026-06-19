# 02 数据准备

## 一、整体思路

我们**不自己跑姿态估计提取 NTU 视频的骨骼**。

原因:
- NTU60 有 56880 个视频,跑一次 HRNet 姿态估计要好几个小时,且每个视频要先用 mmdet 检测人体再 mmpose 估姿态,链路长易出错
- **OpenMMLab 官方已经发布预提取好的 pickle 文件**,直接下载即可
- 文件结构清晰,可被 MMAction2 / pyskl 直接消费

完整数据流:

```
ntu60_2d.pkl                      ← 下载(900MB)
   │
   ▼
build_binary_pkl.py               ← 筛选 + 重标签 + 划分检查
   │
   ▼
fall_binary_xsub.pkl              ← 训练用文件
   │
   ▼
visualize_skeleton.py             ← 肉眼校验骨骼连线
   │
   ▼
split_check.py                    ← 自动化检查
   │
   ▼
开始训练
```

## 二、Step 1:下载预提取骨骼

```bash
conda activate falldet
cd ~/autodl-tmp/fall-detection

# 默认下载到 ./data/
python data_prep/download_pkl.py

# 输出预期:
# ✓ ntu60_2d.pkl: 56880 个样本,60 个类别
#   split: ['xsub_train', 'xsub_val', 'xview_train', 'xview_val']
#   第一个样本: frame_dir=S001C001P001R001A001, label=0, keypoint shape=(2, 103, 17, 2)
```

文件大约 900 MB,首次下载需 5-15 分钟。

## 三、Step 2:构建摔倒二分类数据集

### 3.1 默认策略(推荐起步)

```bash
python data_prep/build_binary_pkl.py \
    --src data/ntu60_2d.pkl \
    --dst data/fall_binary_xsub.pkl \
    --neg-strategy mixed \
    --neg-pos-ratio 2.0
```

**输出**:`data/fall_binary_xsub.pkl`,内容:
- 正样本(摔倒,label=1):约 950 个(原 NTU60 摔倒类的全部样本)
- 负样本(非摔倒,label=0):约 1900 个,其中:
  - 困难负样本 ≈ 1700(sit down, stand up, staggering 等 9 类)
  - 随机负样本 ≈ 200(随机抽 10 个其他类各少量)

### 3.2 策略说明(论文消融用)

| `--neg-strategy` | 用途 | 论文章节 |
|---|---|---|
| `hard`   | 仅困难负 → 模型对 sit/stand 区分能力 | 4.2 |
| `random` | 仅随机负 → 看缺少困难负的 baseline 效果 | 4.2 |
| `mixed`  | 推荐主线训练用 | 主对比实验 |

`--neg-pos-ratio` 控制负:正比例,默认 2.0。如果想训练更"敏感"(高 Recall)的模型,降到 1.0;想训练更"精确"(低 FPR)的模型,升到 3.0-4.0。

### 3.3 论文数据量消融

```bash
# 100% 训练数据(默认)
python data_prep/build_binary_pkl.py --subsample-ratio 1.0 \
    --dst data/fall_binary_100pct.pkl

# 50% 训练数据
python data_prep/build_binary_pkl.py --subsample-ratio 0.5 \
    --dst data/fall_binary_50pct.pkl

# 25% 训练数据
python data_prep/build_binary_pkl.py --subsample-ratio 0.25 \
    --dst data/fall_binary_25pct.pkl

# 10% 训练数据
python data_prep/build_binary_pkl.py --subsample-ratio 0.1 \
    --dst data/fall_binary_10pct.pkl
```

注意:**只在 train 上下采样,val 保持完整**(否则评估不准)。

## 四、Step 3:可视化校验关键点对齐 ★

这是上一版项目踩过的最大坑,**必须做**:

```bash
python data_prep/visualize_skeleton.py \
    --src data/fall_binary_xsub.pkl \
    --num 5 \
    --out-dir vis/

# 输出 mp4 视频到 vis/ 目录,把它们下载到本地用任意播放器打开
```

**检查清单**:打开生成的 mp4,确认:

- [ ] 0 号点(nose)在身体顶端,**不应该出现在脚下**
- [ ] 头部三角形(0-1-3-0, 0-2-4-0)的连线像一张脸
- [ ] 5-6(双肩连线)在脖子下方
- [ ] 5-11、6-12 连线是躯干两侧
- [ ] 13-15、14-16 是小腿
- [ ] 摔倒样本(label=1)最后几帧应该看到人在地面平躺/侧倒
- [ ] 非摔倒样本应该看到 sit down / stand up 等动作

如果有一项不对,**说明关键点顺序错位**,需要在 build_binary_pkl.py 里加 permutation 修正(NTU 的 HRNet pickle 通常无此问题,但部署时用 YOLO26-Pose 提取可能要做)。

## 五、Step 4:自动化划分检查

```bash
python data_prep/split_check.py --src data/fall_binary_xsub.pkl
```

**通过标准**:输出最后一行必须是 `✓ 全部检查通过`。

检查内容:
1. **样本名重叠** — train 和 val 不能有同名样本
2. **受试者重叠** — X-Sub 划分下,train 和 val 的受试者必须完全分离(NTU 默认按 20 训练 / 20 验证拆人)
3. **标签分布** — 各 split 必须同时包含正负样本
4. **关键点合理性** — 无 NaN、无全 0、形状正确 (V=17, C=2)
5. **NTU 动作分布** — 看 A43 (falling) 在 train/val 各有多少

如果第 2 项不通过,说明源 pickle 划分有问题,不要硬训。

## 六、Step 5:[可选]获取真实场景测试集

NTU 是实验室摆拍,真实场景泛化必须用另外的数据。

### URFD (推荐,免费下载)

```bash
mkdir -p data/urfd && cd data/urfd
# 下载 URFD RGB 视频 + 加速度数据
wget http://fenix.ur.edu.pl/mkepski/ds/data/fall-01-cam0-rgb.zip
# (URFD 总共 30 个 fall + 40 个 ADL,逐个下,或者用 for 循环)
for i in $(seq 1 30); do
    wget "http://fenix.ur.edu.pl/mkepski/ds/data/fall-$(printf '%02d' $i)-cam0-rgb.zip" || true
done
for i in $(seq 1 40); do
    wget "http://fenix.ur.edu.pl/mkepski/ds/data/adl-$(printf '%02d' $i)-cam0-rgb.zip" || true
done
unzip "*.zip" && rm *.zip
cd ../..
```

之后在 `data/urfd/` 下会有一堆图像帧序列,每个序列对应一段视频(由帧拼成)。

### Le2i

```
http://le2i.cnrs.fr/Fall-detection-Dataset
```

需邮件申请,有规范的 fall 标注帧。

### 把真实视频转为骨骼

真实视频不能直接用 NTU 的预提取 pickle,需要自己跑姿态估计:

```bash
# 用 YOLO26-Pose 一站式提取
python inference/extract_pose_yolo26.py \
    --video_dir data/urfd \
    --out data/urfd_skeleton.pkl
```

(见 `docs/05_inference_deployment.md`)

## 七、文件总览(数据准备完成后)

```
data/
├── ntu60_2d.pkl                ← 原始下载(900 MB,保留备份)
├── fall_binary_xsub.pkl        ← 主线训练用
├── fall_binary_50pct.pkl       ← 消融实验:50% 数据
├── fall_binary_25pct.pkl       ← 消融实验:25% 数据
├── fall_binary_hard_only.pkl   ← 消融实验:仅困难负样本
├── fall_binary_random_only.pkl ← 消融实验:仅随机负样本
└── urfd/                       ← (可选)真实场景测试集
    └── ...
```

下面这些命令可以一次性生成所有消融实验数据:

```bash
# 主线
python data_prep/build_binary_pkl.py \
    --neg-strategy mixed --neg-pos-ratio 2.0 \
    --dst data/fall_binary_xsub.pkl

# 消融:负样本策略
python data_prep/build_binary_pkl.py \
    --neg-strategy hard \
    --dst data/fall_binary_hard_only.pkl
python data_prep/build_binary_pkl.py \
    --neg-strategy random \
    --dst data/fall_binary_random_only.pkl

# 消融:训练数据量
for r in 0.5 0.25 0.1; do
    python data_prep/build_binary_pkl.py \
        --subsample-ratio $r \
        --dst data/fall_binary_${r/0./}pct.pkl
done
```

---

下一篇:`03_model_training.md`

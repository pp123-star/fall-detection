# 00 技术选型与方案设计

## 一、为什么不延续上一版方案

上一版:YOLO 人体检测 + OpenPose 骨骼 + ByteTrack 跟踪 + CTR-GCN 图卷积。

主要问题:
1. **OpenPose 已严重过时**:依赖 Caffe / CMake / 特定 CUDA,与现代 Python 生态冲突,环境搭建占用了大量时间
2. **CTR-GCN 对姿态估计噪声敏感**:OpenPose 在遮挡/快速运动下关键点跳变,GCN 把每个关节当成图节点,坐标抖动直接传到分类头
3. **任务定义粗糙**:NTU 60 类多分类训练用于"摔倒检测"二分类目标,模型在类间长尾上分配的容量被浪费
4. **数据划分疏忽**:滑动窗口随机切分导致验证集泄漏,虚高的分数掩盖了真实问题

## 二、新方案的关键决策

### 决策 1:用 PoseConv3D 替代 CTR-GCN 作为主线

**依据**:论文《Revisiting Skeleton-based Action Recognition》(Duan 等,CVPR 2022 Oral)证明在 NTU60 X-Sub 上:
- PoseConv3D(SlowOnly-R50):**94.1%**
- CTR-GCN: 92.4%
- ST-GCN: 88.3%

PoseConv3D 的核心优势:
- 把关键点坐标 (x, y) 渲染成高斯热图,**空间维度上"抗噪"**——关键点偏几个像素不会让热图发生质变
- 时间维度堆叠成 3D 数据体后用 3D CNN 处理,**继承了视频领域 3D CNN 的成熟设计**(SlowOnly/X3D backbone 都可)
- 输入分辨率可控(默认 48×48 或 56×56),计算量比 RGB 视频 3D CNN 小一个量级

**代价**:训练比 GCN 慢约 2-3 倍,显存占用大一些(单卡 RTX 3090/4090 完全够用)。

### 决策 2:ST-GCN++ 作为对比/消融

**依据**:同作者团队提出,在论文里和 PoseConv3D 同台对比。选 ST-GCN++ 而不是老 ST-GCN 或 CTR-GCN 的原因:
- 比 ST-GCN 高约 2-3 个点,但代码同样简单
- 比 CTR-GCN 快、参数少,更适合做"轻量对比方案"
- 论文里写"骨骼表征方式对比"时变量更干净(同作者团队的两个方法)

### 决策 3:把任务从 60 类多分类改成二分类

**摔倒 = NTU 类 A43(falling)+ 真实场景摔倒补充**  
**非摔倒** 不是用其他 59 类全部,而是**重点构造困难负样本**:
- A8 (sit down)
- A9 (stand up)  
- A14 (put on jacket) — 动作幅度大、躯干前倾,容易混淆
- A24 (kicking something) — 单脚抬起,易误判
- A26 (hopping) — 突然垂直运动
- A27 (jump up) — 全身腾空
- A41 (sneeze/cough) — 上半身突然弯曲
- A42 (staggering) — **极易混淆**,踉跄
- 加少量随机抽取的其他类作平衡负样本

**为什么这样设计**:
- 部署时真实误报最多来自"坐下/躺下/蹲下/绊到没倒"这类动作
- 把负样本做窄、做难,模型学到的"摔倒特征"才稳健
- 二分类输出概率更适合工程上的阈值调整

### 决策 4:用 MMAction2 v1.x 而不是 pyskl

- pyskl 最后实质性更新在 2023 年 3 月,依赖 `mmcv-full`(已废弃)
- MMAction2 v1.x **吸收了 pyskl 的全部成果**:PoseConv3D 和 ST-GCN++ 都在 model zoo 里
- MMAction2 v1.x 用 `mmcv >= 2.0`(纯 Python 安装,无需编译 CUDA op,这是和 mmcv-full 的关键差别)
- 配套 `mmengine` 提供统一的 Runner,checkpoint 保存逻辑更清晰

### 决策 5:部署端用 YOLO26-Pose

YOLO26 是 Ultralytics 2026 年 1 月发布的最新版本,关键升级:
- 引入 **Residual Log-Likelihood Estimation (RLE)** 提升关键点定位精度
- 端到端 NMS-free 推理,延迟更低
- COCO 17 点输出,**与训练用的 NTU 2D 骨骼格式可对齐**(都是 17 点 COCO)
- 单 pip 包 `ultralytics`,集成检测+关键点+ByteTrack 一站式

> ⚠️ 关键点格式对齐:训练用的 `ntu60_2d.pkl` 是 HRNet 在 NTU 视频上提取的 COCO 17 关键点,部署用 YOLO26-Pose 也是 COCO 17,**直接复用,无需做关键点重映射**。这是这次选型避坑的重要细节。

## 三、数据策略

### 3.1 训练数据 — 不自己跑姿态估计

OpenMMLab 官方已经发布了 NTU60/NTU120 用 HRNet 预提取好的 2D 骨骼 pickle 文件(每个文件约几百 MB)。**直接下载,跳过自己用 mmpose 跑姿态估计这一步**(原来这步要花几个小时跑 56880 个视频)。

文件下载地址:
- NTU60 2D: `https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl`
- NTU120 2D: `https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_2d.pkl`

文件结构(pickle 反序列化后):
```python
{
    'split': {
        'xsub_train': ['S001C001P001R001A001', ...],   # 训练集样本名列表
        'xsub_val':   ['S001C003P001R001A001', ...],
        ...
    },
    'annotations': [
        {
            'frame_dir': 'S001C001P001R001A001',
            'label': 0,                                  # 0-59 类
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 103,
            'keypoint': np.ndarray (M, T, V=17, C=2),   # M最大人数, T帧数, V关节数, C坐标
            'keypoint_score': np.ndarray (M, T, V=17),  # 关键点置信度
        },
        ...
    ]
}
```

### 3.2 训练集/验证集划分 — 严格按受试者(X-Sub)

NTU 官方 X-Sub 划分:
- 训练受试者 ID:1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
- 验证受试者 ID:其余 20 人

**重要**:同一个受试者的所有视频要么全在训练集要么全在验证集,**不能按视频随机切分**。这正是上一版项目踩过的坑——"原始视频为单位划分"对应到 NTU 上就是"原始受试者为单位划分"。下载的 pickle 已经按官方 X-Sub 切好,我们的脚本只需要保留这个切分。

### 3.3 真实场景测试集

NTU 是演员摆拍,跨场景泛化不能光靠 NTU 验证集说话。推荐用以下数据集做跨数据集测试:

- **URFD (UR Fall Detection)**:30 个真实场景摔倒视频 + 40 个 ADL,有标注精确帧
- **Le2i Fall Detection**:191 个视频,真实居家场景
- **Multicam**:24 序列,多视角,有遮挡

至少补 URFD,作为论文 4.4 跨数据集泛化章节的数据。

## 四、整体数据流图

```
┌─────────────────────────────────────────────────────────────┐
│ 训练阶段                                                     │
│                                                              │
│  ntu60_2d.pkl  ──filter──>  fall_binary_xsub.pkl            │
│  (60类HRNet骨骼)            (摔倒 vs 困难负样本,二分类)      │
│                                  │                           │
│                                  ▼                           │
│                          ┌───────────────┐                   │
│                          │ PoseConv3D    │ (主线)            │
│                          │ ST-GCN++      │ (对比)            │
│                          └───────┬───────┘                   │
│                                  │                           │
│                                  ▼                           │
│                          best_*.pth                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 推理阶段                                                     │
│                                                              │
│  任意视频  ──YOLO26-Pose──>  COCO 17 关键点序列              │
│                  │                                           │
│                  └──ByteTrack──>  按人ID切分                 │
│                                       │                      │
│                                       ▼                      │
│                              滑动窗口(48帧/16步)             │
│                                       │                      │
│                                       ▼                      │
│                               PoseConv3D 分类                │
│                                       │                      │
│                                       ▼                      │
│                            摔倒概率 > 阈值 → 报警             │
└─────────────────────────────────────────────────────────────┘
```

## 五、显存与时长估算(单卡 RTX 4090,24GB)

| 模型 | batch | 显存 | 1 epoch | 总训练时长(120 epoch) |
|---|---|---|---|---|
| PoseConv3D-R50 | 16 | ~14 GB | 3-5 min | 6-10 h |
| ST-GCN++ | 32 | ~6 GB | 1-2 min | 2-4 h |

若用 RTX 3090(同 24GB)或 A40,时长接近;若用 4090×2 多卡,可减半。AutoDL 上 RTX 4090 单卡按 ¥2-3/小时计费,一次完整训练成本约 ¥15-30。

## 六、消融实验设计(论文用)

| 实验 | 改动 | 看什么 |
|---|---|---|
| E1 主对比 | PoseConv3D vs ST-GCN++ | F1, Recall, FPR, 推理速度 |
| E2 负样本策略 | 仅随机负 vs 困难负 vs 困难+随机混合 | FPR下降幅度,Recall是否被牺牲 |
| E3 训练数据量 | 100% / 50% / 25% / 10% | 数据效率曲线 |
| E4 输入帧数 | 32 / 48 / 64 帧窗口 | 时序长度影响 |
| E5 跨数据集 | NTU训→URFD测 | 真实场景泛化能力 |
| E6 姿态估计源 | HRNet提取 vs YOLO26-Pose提取 | 部署一致性 |

E1-E3 是必做,E4-E6 是加分项。

---

下一篇:`01_environment_setup.md`

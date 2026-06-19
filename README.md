# 摔倒检测 / 视频动作识别毕设项目

## 一、项目定位

**论文题目**:基于深度学习的视频动作识别技术研究  
**核心子任务**:摔倒检测  
**论文价值点**:骨骼3D热图(PoseConv3D) vs 骨骼图卷积(ST-GCN++)对比,困难负样本消融,跨数据集泛化测试

## 二、技术选型一览

| 模块 | 选型 | 替代 |
|---|---|---|
| 训练框架 | MMAction2 v1.x | pyskl(停更) |
| 主线模型 | **PoseConv3D**(SlowOnly + 3D高斯热图) | - |
| 对比模型 | **ST-GCN++** | CTR-GCN/AAGCN |
| 训练数据 | NTU RGB+D 60 预提取2D骨骼 | NTU120/UCF101 |
| 部署姿态估计 | **YOLO26-Pose** (Ultralytics) | RTMPose / MediaPipe |
| 部署跟踪 | ByteTrack(Ultralytics内置) | BoT-SORT |
| 关键点格式 | **COCO 17 点** | NTU Kinect 25点 |
| 任务定义 | **二分类**(摔倒 / 非摔倒) | 60类多分类 |

## 三、目录结构

```
fall-detection/
├── README.md                       ← 你在这
├── docs/                           ← 详细文档
│   ├── 00_technical_design.md
│   ├── 01_environment_setup.md     ← 先看这个
│   ├── 02_data_preparation.md
│   ├── 03_model_training.md
│   ├── 04_evaluation_visualization.md
│   ├── 05_inference_deployment.md
│   ├── 06_multitarget_realtime_detection.md
│   └── 99_troubleshooting_checklist.md ← 强烈建议看
├── env/
│   ├── setup_autodl.sh             ← AutoDL 一键环境搭建
│   └── requirements_extra.txt
├── data_prep/
│   ├── download_pkl.py             ← 下载预提取2D骨骼
│   ├── build_binary_pkl.py         ← 构建摔倒二分类数据集
│   ├── visualize_skeleton.py       ← 关键点对齐校验(必跑)
│   └── split_check.py              ← 训练/验证划分泄漏检查
├── configs/
│   ├── _base_/
│   │   ├── default_runtime.py
│   │   └── schedule.py
│   ├── posec3d_fall_binary.py      ← 主线模型配置
│   └── stgcnpp_fall_binary.py      ← 对比模型配置
├── tools/
│   ├── train.sh
│   ├── test.sh
│   ├── eval_binary_metrics.py      ← 二分类精确指标(F1/P/R/混淆矩阵)
│   ├── plot_curves.py              ← 训练曲线绘制
│   └── verify_best_ckpt.py         ← checkpoint 保存逻辑验证
├── inference/
│   ├── extract_pose_yolo26.py      ← YOLO26-Pose 提取骨骼
│   ├── pose_to_pyskl_format.py     ← 关键点 → MMAction2 格式
│   ├── realtime_demo.py            ← 实时摄像头/视频演示
│   ├── multitarget_realtime_demo.py ← 多目标实时摔倒检测
│   └── batch_predict.py            ← 批量视频推理
└── scripts/
    └── run_all.sh                  ← 一键串联流程
```

## 四、最短路径(给"先跑通"的你)

```bash
# 1. 进入云GPU实例后
bash env/setup_autodl.sh

# 2. 下载预处理好的 NTU 2D 骨骼(免去自己跑姿态估计的麻烦)
python data_prep/download_pkl.py

# 3. 构建摔倒二分类数据集(摔倒 vs 困难负样本)
python data_prep/build_binary_pkl.py

# 4. 关键点对齐可视化(必跑!确保头连头、脚连脚)
python data_prep/visualize_skeleton.py --num 5

# 5. 训练主模型
bash tools/train.sh configs/posec3d_fall_binary.py 1   # 单卡 GPU

# 6. 评估
python tools/eval_binary_metrics.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_*.pth \
    --config configs/posec3d_fall_binary.py

# 7. (可选)推理任意视频
python inference/batch_predict.py --video your_test.mp4 \
    --pose_model yolo26x-pose.pt \
    --action_ckpt work_dirs/posec3d_fall_binary/best_*.pth
```

## 五、各文档阅读顺序

1. `docs/01_environment_setup.md` — 别跳过,生态版本对齐是这次项目避坑核心
2. `docs/02_data_preparation.md` — 解释为什么用预提取 pickle、为什么改二分类
3. `docs/03_model_training.md` — 含 PoseConv3D vs ST-GCN++ 两条命令
4. `docs/04_evaluation_visualization.md` — 论文里那些图表怎么出
5. `docs/05_inference_deployment.md` — YOLO26-Pose + 训练好的分类器串联
6. `docs/99_troubleshooting_checklist.md` — 把上次项目踩的坑都列了,新坑也列了

## 六、给毕业论文的章节建议(对应实验内容)

| 论文章节 | 对应代码/实验 |
|---|---|
| 第3章 方法 | `configs/posec3d_fall_binary.py`、`configs/stgcnpp_fall_binary.py` |
| 4.1 主对比实验 | `tools/eval_binary_metrics.py` 跑两个模型 |
| 4.2 困难负样本消融 | `data_prep/build_binary_pkl.py` 改 `--neg_strategy` 重训 |
| 4.3 数据量消融 | `data_prep/build_binary_pkl.py` 改 `--subsample_ratio` |
| 4.4 跨数据集泛化 | URFD 测试,见 `docs/04_evaluation_visualization.md` |
| 4.5 部署效率分析 | `inference/realtime_demo.py` 记 FPS |

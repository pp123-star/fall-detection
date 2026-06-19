# =============================================================================
# configs/stgcnpp_fall_binary.py
#
# 对比模型:ST-GCN++ (Spatio-Temporal Graph Convolutional Network)
# 任务:摔倒 vs 非摔倒(二分类)
# 数据:data/fall_binary_xsub.pkl
#
# 与 PoseConv3D 主线的区别:
#   - 不渲染热图,直接用关键点坐标作为图节点
#   - 训练快约 3 倍,显存占用低
#   - 论文里作为骨骼3D热图方案的对比基线
# =============================================================================

_base_ = ["_base_/default_runtime.py"]

# ============================================================
# 模型
# ============================================================
model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="STGCN",
        gcn_adaptive="init",                # ST-GCN++ 的自适应邻接矩阵
        gcn_with_res=True,                  # 残差连接
        tcn_type="mstcn",                   # ST-GCN++ 的多尺度 TCN
        graph_cfg=dict(layout="coco", mode="spatial"),  # ★ COCO 17 点骨架
    ),
    cls_head=dict(
        type="GCNHead",
        num_classes=2,                       # ★ 二分类
        in_channels=256,
    ),
    test_cfg=dict(average_clips="prob"),
)

# ============================================================
# 数据集
# ============================================================
dataset_type = "PoseDataset"
ann_file = "data/fall_binary_xsub.pkl"

split_train = "xsub_train"
split_val = "xsub_val"

clip_len = 100                              # GCN 系列默认用 100 帧

train_pipeline = [
    dict(type="PreNormalize2D"),            # 关键点 (x,y) 归一化到 [-1, 1]
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),  # joint 模态
    dict(type="UniformSampleFrames", clip_len=clip_len),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="PackActionInputs"),
]

val_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSampleFrames", clip_len=clip_len, num_clips=1, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="PackActionInputs"),
]

test_pipeline = [
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSampleFrames", clip_len=clip_len, num_clips=10, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=32,                          # GCN 模型小,可以更大 batch
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=10,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split=split_train,
        ),
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split=split_val,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split=split_val,
        test_mode=True,
    ),
)

# ============================================================
# 评估
# ============================================================
val_evaluator = dict(
    type="AccMetric",
    metric_options=dict(
        top_k_accuracy=dict(topk=(1,)),
        mean_class_accuracy=dict(),
    ),
)
test_evaluator = val_evaluator

# ============================================================
# 优化器与训练计划
# ============================================================
total_epochs = 16

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD",
        lr=0.1,                              # GCN 比 3D CNN 收敛快,lr 可小一些
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
    ),
    clip_grad=dict(max_norm=40, norm_type=2),
)

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        eta_min=0,
        T_max=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=total_epochs,
    val_begin=1,
    val_interval=1,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# ============================================================
# 输出目录
# ============================================================
work_dir = "work_dirs/stgcnpp_fall_binary"

# ============================================================
# 备注
# ============================================================
# 单卡 RTX 4090 上预期:
#   - 每个 epoch 约 30-60 秒
#   - 总训练约 30 分钟
#   - 收敛后 val acc 应该比 PoseConv3D 略低 1-3 个点
#     (这正是论文里要论证的"骨骼3D热图 > 骨骼GCN"的结论)

# =============================================================================
# configs/posec3d_fall_binary.py
#
# 主线模型:PoseConv3D (SlowOnly-R50, 3D 高斯热图)
# 任务:摔倒 vs 非摔倒(二分类)
# 数据:data/fall_binary_xsub.pkl
# =============================================================================

_base_ = ["_base_/default_runtime.py"]

# ============================================================
# 模型
# ============================================================
model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="ResNet3dSlowOnly",
        depth=50,
        pretrained=None,                 # 从零开始训(NTU 数据足够)
        in_channels=17,                  # 17 个关键点 -> 17 通道热图输入
        base_channels=32,                # SlowOnly 原配置
        num_stages=3,                    # 阶段数(默认 3 阶段够用)
        out_indices=(2,),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1),
    ),
    cls_head=dict(
        type="I3DHead",
        in_channels=512,                 # SlowOnly stage3 输出通道
        num_classes=2,                   # ★ 二分类
        spatial_type="avg",
        dropout_ratio=0.5,
        average_clips="prob",
    ),
    # 推理时的 normalization,关键点已经是像素坐标,不再额外归一
    test_cfg=dict(average_clips="prob"),
)

# ============================================================
# 数据集
# ============================================================
dataset_type = "PoseDataset"
ann_file = "data/fall_binary_xsub.pkl"

# X-Sub 训练/验证划分
split_train = "xsub_train"
split_val = "xsub_val"

# 输入设置
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]    # COCO 17 点左侧索引
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]  # 右侧索引(给数据增强 FlipKeypoint 用)
clip_len = 48                             # 时间维度长度(帧)
heatmap_size = 56                         # 热图边长

train_pipeline = [
    dict(type="UniformSampleFrames", clip_len=clip_len),
    dict(type="PoseDecode"),
    dict(type="PoseCompact", hw_ratio=1.0, allow_imgpad=True),
    dict(type="Resize", scale=(-1, 64)),
    dict(type="RandomResizedCrop", area_range=(0.56, 1.0)),
    dict(type="Resize", scale=(heatmap_size, heatmap_size), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type="GeneratePoseTarget",
         sigma=0.6,
         use_score=True,
         with_kp=True,
         with_limb=False),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="PackActionInputs"),
]

val_pipeline = [
    # 验证时不做随机增强,均匀采样 + 中心裁剪
    dict(type="UniformSampleFrames", clip_len=clip_len, num_clips=1, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="PoseCompact", hw_ratio=1.0, allow_imgpad=True),
    dict(type="Resize", scale=(-1, 64)),
    dict(type="CenterCrop", crop_size=64),
    dict(type="Resize", scale=(heatmap_size, heatmap_size), keep_ratio=False),
    dict(type="GeneratePoseTarget",
         sigma=0.6,
         use_score=True,
         with_kp=True,
         with_limb=False),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="PackActionInputs"),
]

test_pipeline = [
    # 测试时 10-clip TTA,提升评估稳定性
    dict(type="UniformSampleFrames", clip_len=clip_len, num_clips=10, test_mode=True),
    dict(type="PoseDecode"),
    dict(type="PoseCompact", hw_ratio=1.0, allow_imgpad=True),
    dict(type="Resize", scale=(-1, 64)),
    dict(type="CenterCrop", crop_size=64),
    dict(type="Resize", scale=(heatmap_size, heatmap_size), keep_ratio=False),
    dict(type="GeneratePoseTarget",
         sigma=0.6,
         use_score=True,
         with_kp=True,
         with_limb=False),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="RepeatDataset",
        times=10,                          # 每个 epoch 重复 10 次(NTU 摔倒样本仅 ~700,需要重复采样)
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split=split_train,
        ),
    ),
)

val_dataloader = dict(
    batch_size=16,
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
total_epochs = 24                          # 二分类数据少,epoch 不用太大
# RepeatDataset 已经把 epoch 拉长了 10 倍,所以实际等价 240 epoch

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD",
        lr=0.2,                            # 单卡 batch=16 时 lr=0.2(论文用 8 卡 batch=128 lr=0.4)
        momentum=0.9,
        weight_decay=0.0003,
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
work_dir = "work_dirs/posec3d_fall_binary"

# ============================================================
# 备注
# ============================================================
# 单卡 RTX 4090 上预期:
#   - 每个 epoch 约 3-5 分钟
#   - 总训练约 1.5-2 小时
#   - 收敛后 val acc 应该 >= 96%
#
# 若显存不够(< 16 GB),减半 batch_size 并相应减半 lr:
#   train_dataloader.batch_size = 8
#   optim_wrapper.optimizer.lr = 0.1

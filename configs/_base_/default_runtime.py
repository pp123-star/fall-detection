# configs/_base_/default_runtime.py
# MMAction2 v1.x 通用运行时设置

default_scope = "mmaction"

default_hooks = dict(
    runtime_info=dict(type="RuntimeInfoHook"),
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20, ignore_last=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    # ★ 关键:save_best 严格按 val acc top1 保存最佳 checkpoint
    # 防止上一版项目"checkpoint 保存逻辑 bug 导致训练白跑"
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,                  # 每个 epoch 保存一次
        save_best="acc/top1",        # 按 top1 acc 选最佳
        rule="greater",              # acc 越大越好
        max_keep_ckpts=3,            # 最多保留 3 个,省磁盘
        save_last=True,              # 总是保留最后一个
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    sync_buffers=dict(type="SyncBuffersHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=20, by_epoch=True)

vis_backends = [
    dict(type="LocalVisBackend"),
    # 如要用 TensorBoard,取消下行注释
    # dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="ActionVisualizer", vis_backends=vis_backends)

log_level = "INFO"
load_from = None
resume = False

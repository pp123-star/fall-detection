"""
inference/pose_to_pyskl_format.py — COCO 17 点 → MMAction2 PoseDataset 输入格式

【为什么需要这个文件】
训练阶段:用的是 OpenMMLab 预提取的 ntu60_2d.pkl(已是 MMAction2 格式)
推理阶段:用 YOLO26-Pose 提取新视频的关键点,输出格式不同
所以推理时需要做一次格式转换。

【MMAction2 PoseDataset 期望的格式】
单个样本是一个 dict,包含:
    {
        'keypoint':       np.ndarray, shape (M, T, V, C),  # M=人数, T=帧数, V=关键点数, C=2(x,y)
        'keypoint_score': np.ndarray, shape (M, T, V),     # 每个关键点的置信度
        'frame_dir':      str,                              # 样本标识(随便起)
        'img_shape':      tuple (H, W),                     # 原图尺寸,用于热图生成
        'original_shape': tuple (H, W),
        'total_frames':   int,                              # = T
        'label':          int,                              # 推理时不用,占位 0
    }

其中 M 通常 = 1(单人摔倒检测场景);多人场景 M 会变化,本模块支持。
关键点顺序必须是 COCO 17 点,且和模型 graph_cfg 的 layout='coco' 严格一致。

【COCO 17 点顺序】(必须严格按这个,否则准确率会莫名其妙地低)
    0: nose
    1: left_eye        2: right_eye
    3: left_ear        4: right_ear
    5: left_shoulder   6: right_shoulder
    7: left_elbow      8: right_elbow
    9: left_wrist     10: right_wrist
   11: left_hip       12: right_hip
   13: left_knee      14: right_knee
   15: left_ankle     16: right_ankle

YOLO-Pose / RTMPose 默认就是这个顺序(可放心),但仍建议用
data_prep/visualize_skeleton.py 抽样人工核验头连头脚连脚。
"""
import numpy as np


# COCO 17 点常量(给其他模块复用)
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
COCO_NUM_KEYPOINTS = 17


def build_sample(
    keypoints_seq,
    scores_seq,
    img_shape,
    frame_dir="unknown",
    label=0,
):
    """构造一个 MMAction2 PoseDataset 样本 dict。

    Args:
        keypoints_seq: 关键点序列。两种可接受形态:
            - list of np.ndarray, len=T, 每个 shape (M, V, 2)  ← 推荐(逐帧)
            - np.ndarray, shape (M, T, V, 2)                     ← 已经堆叠好
        scores_seq:    置信度序列,形态与上面对应,缺少最后一维:
            - list of np.ndarray, len=T, 每个 shape (M, V)
            - np.ndarray, shape (M, T, V)
        img_shape:     (H, W),原视频帧尺寸。**重要**:PoseConv3D 用这个生成高斯热图;
                       推理时如果做了 resize,这里要传 resize 后的尺寸。
        frame_dir:     样本名,推理时随便起一个;批量时建议用文件名+片段索引
        label:         占位用,推理时无所谓;评估时要给真实标签(0=非摔倒,1=摔倒)

    Returns:
        dict 形如 ntu60_2d.pkl 里的一条记录,可直接喂给 PoseDataset。
    """
    # 统一成 np.ndarray
    kpts = _to_mtvc(keypoints_seq)        # (M, T, V, 2)
    scrs = _to_mtv(scores_seq)            # (M, T, V)

    # 一致性检查(顺手挡住一次随机崩溃)
    M, T, V, C = kpts.shape
    assert C == 2, f"关键点维度必须是 (x,y),实际 C={C}"
    assert V == COCO_NUM_KEYPOINTS, (
        f"关键点数量必须是 17(COCO),实际 V={V}。"
        f"如果使用 BODY_25/Kinect 25 等其他格式,请先转换或重训模型。"
    )
    assert scrs.shape == (M, T, V), (
        f"keypoint_score 形状 {scrs.shape} 与 keypoint 形状 {kpts.shape[:3]} 不匹配"
    )

    H, W = img_shape

    sample = dict(
        keypoint=kpts.astype(np.float32),
        keypoint_score=scrs.astype(np.float32),
        frame_dir=str(frame_dir),
        img_shape=(int(H), int(W)),
        original_shape=(int(H), int(W)),
        total_frames=int(T),
        label=int(label),
    )
    return sample


def _to_mtvc(seq):
    """list of (M, V, 2) 或 (M, T, V, 2) -> (M, T, V, 2)"""
    if isinstance(seq, np.ndarray):
        if seq.ndim == 4:
            return seq
        raise ValueError(f"keypoints ndarray 维度必须是 4,实际 {seq.ndim}")

    # list of per-frame arrays
    if not seq:
        raise ValueError("keypoints 序列为空")
    arrs = [np.asarray(f) for f in seq]
    M = arrs[0].shape[0]
    V = arrs[0].shape[1]

    # 处理"某些帧没检测到人"的情况:用零填充到 M
    padded = []
    for a in arrs:
        if a.shape[0] < M:
            pad = np.zeros((M - a.shape[0], V, 2), dtype=a.dtype)
            a = np.concatenate([a, pad], axis=0)
        elif a.shape[0] > M:
            # 多检测到的人裁掉(摔倒检测主场景单人,简单处理)
            a = a[:M]
        padded.append(a)

    stacked = np.stack(padded, axis=1)  # (M, T, V, 2)
    return stacked


def _to_mtv(seq):
    """list of (M, V) 或 (M, T, V) -> (M, T, V)"""
    if isinstance(seq, np.ndarray):
        if seq.ndim == 3:
            return seq
        raise ValueError(f"scores ndarray 维度必须是 3,实际 {seq.ndim}")

    if not seq:
        raise ValueError("scores 序列为空")
    arrs = [np.asarray(f) for f in seq]
    M = arrs[0].shape[0]
    V = arrs[0].shape[1]

    padded = []
    for a in arrs:
        if a.shape[0] < M:
            pad = np.zeros((M - a.shape[0], V), dtype=a.dtype)
            a = np.concatenate([a, pad], axis=0)
        elif a.shape[0] > M:
            a = a[:M]
        padded.append(a)

    return np.stack(padded, axis=1)


# ============================================================
# 滑窗切分
# ============================================================
def split_into_clips(sample, clip_len=48, stride=16):
    """把一个长视频样本切成多个滑窗 clip。

    推理时摔倒可能发生在视频中任何位置,需要滑窗扫一遍。

    Args:
        sample:   build_sample 返回的 dict
        clip_len: 每个 clip 长度(帧),应与训练 config 的 clip_len 一致(默认 48)
        stride:   滑动步长,默认 16(75% 重叠,论文里常用)

    Returns:
        list[dict],每个元素是一个新的 sample dict,frame_dir 后缀 _clipN
    """
    M, T, V, C = sample["keypoint"].shape
    if T < clip_len:
        # 视频比窗口短,用循环补帧到 clip_len(MMAction2 的做法)
        repeats = (clip_len + T - 1) // T
        kpts = np.tile(sample["keypoint"], (1, repeats, 1, 1))[:, :clip_len]
        scrs = np.tile(sample["keypoint_score"], (1, repeats, 1))[:, :clip_len]
        clip = sample.copy()
        clip["keypoint"] = kpts
        clip["keypoint_score"] = scrs
        clip["total_frames"] = clip_len
        clip["frame_dir"] = f"{sample['frame_dir']}_clip0"
        return [clip]

    clips = []
    starts = list(range(0, T - clip_len + 1, stride))
    # 确保覆盖到尾部
    if starts[-1] + clip_len < T:
        starts.append(T - clip_len)

    for i, s in enumerate(starts):
        clip = sample.copy()
        clip["keypoint"] = sample["keypoint"][:, s:s + clip_len]
        clip["keypoint_score"] = sample["keypoint_score"][:, s:s + clip_len]
        clip["total_frames"] = clip_len
        clip["frame_dir"] = f"{sample['frame_dir']}_clip{i}_f{s}-{s + clip_len}"
        clips.append(clip)

    return clips


if __name__ == "__main__":
    # 自测:造一个假样本看看构造和切分
    T = 100
    fake_kpts = [np.random.rand(1, 17, 2) * 720 for _ in range(T)]
    fake_scrs = [np.random.rand(1, 17) for _ in range(T)]
    sample = build_sample(fake_kpts, fake_scrs, img_shape=(720, 1280), frame_dir="test")
    print("[build_sample] 样本:")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
    clips = split_into_clips(sample, clip_len=48, stride=16)
    print(f"\n[split_into_clips] 切出 {len(clips)} 个 clip")
    for c in clips[:3]:
        print(f"  frame_dir={c['frame_dir']:50s}  keypoint.shape={c['keypoint'].shape}")

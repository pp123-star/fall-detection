# 05 推理部署

> 目标:把训练好的 PoseConv3D 模型套上 YOLO26-Pose,串成端到端摔倒检测系统,能跑实时摄像头或视频文件,论文最后一节"工程落地演示"用。

## 一、系统组成图

```
        ┌────────────────┐
视频源 ─→│ YOLO26-Pose    │── 人体框 + 17 关键点 + track_id
        │ (Ultralytics)  │
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │ 滚动缓冲区     │── 维护最近 48 帧的骨骼序列
        │ (deque 48)     │
        └────────────────┘
                │
                ▼ (每 N 帧触发一次)
        ┌────────────────┐
        │ build_sample   │── 拼成 MMAction2 PoseDataset 格式
        │ split_into_    │   dict(keypoint, keypoint_score, img_shape, ...)
        │ clips          │
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │ PoseConv3D     │── P(fall) ∈ [0, 1]
        │ (训练好的)     │
        └────────────────┘
                │
                ▼
        ┌────────────────┐
        │ 可视化叠加     │── 骨骼线 + 概率条 + 警报横幅
        │ + 写视频/显示  │
        └────────────────┘
```

---

## 二、最快上手(三条命令)

```bash
# 单视频离线判定(给一个 video,告诉你是否含摔倒)
python inference/batch_predict.py \
    --video your_test.mp4 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18.pth \
    --out preds/your_test.json

# 实时摄像头演示
python inference/realtime_demo.py \
    --source 0 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18.pth

# 视频文件实时推理 + 保存可视化(服务器/Headless 无显示器场景)
python inference/realtime_demo.py \
    --source your_test.mp4 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_18.pth \
    --save-out figs/demo.mp4 \
    --no-show
```

---

## 三、关键参数解读

### `--clip-len`(默认 48)

滚动缓冲区长度,**必须等于训练 config 里的 clip_len**(`configs/posec3d_fall_binary.py` 第 ~80 行)。否则模型输入维度对不上,要么报错要么准确率崩。

### `--infer-every`(默认 4)

每 N 帧才跑一次分类器,中间帧沿用上次的概率显示。
- 设 1:每帧都推,FPS 最低但概率曲线最平滑
- 设 4:常用折中,FPS 高、概率不会跳得太厉害
- 设 8:吃旧 GPU 时用,FPS 高,但概率响应滞后约 8/30 ≈ 0.27 秒

### `--threshold`(默认 0.5)

P(fall) > 阈值才报警。**不要直接用 0.5**,跑完 `eval_binary_metrics.py` 后取 `metrics.json["best_threshold"]`,通常 0.4-0.6 之间。

部署阈值的取法:
- 安防场景(漏报代价高):阈值低一点,如 0.35,牺牲精确率换召回率
- 智能家居/养老(误报频繁会让用户烦):阈值高一点,如 0.65

### `--aggregate`(batch_predict)

视频级聚合策略:
- `max`(默认):任一 clip 超阈值就判摔倒,最敏感
- `mean`:clip 概率均值,稳但对短时摔倒不敏感
- `vote`:多数 clip 投票,最保守(假设视频里"主要内容"是摔倒)

摔倒是瞬时事件,**推荐 `max`**。

### `--pose-weights`(默认 `yolo26x-pose.pt`)

| 权重 | 模型大小 | mAP@.5(COCO Pose) | 单帧延迟(4090) | 适合 |
|---|---|---|---|---|
| yolo26n-pose.pt | 6 MB | ~58 | ~3 ms | 边缘设备 |
| yolo26s-pose.pt | 18 MB | ~64 | ~4 ms | 性价比之选 |
| yolo26m-pose.pt | 42 MB | ~69 | ~6 ms | 常用 |
| yolo26l-pose.pt | 67 MB | ~71 | ~9 ms | — |
| yolo26x-pose.pt | 110 MB | ~73 | ~12 ms | 论文 demo(最准) |

首次运行会自动下载到 `~/.config/Ultralytics`。

---

## 四、部署到云 GPU 服务器(纯 Headless)

无显示器,只能录视频:

```bash
# 1. 把待测视频上传到云实例 /root/test_videos/
# 2. 跑批量
python inference/batch_predict.py \
    --video-dir /root/test_videos \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_*.pth \
    --out preds/results.csv

# 3. 把可视化结果录成 mp4
for v in /root/test_videos/*.mp4; do
    name=$(basename "$v" .mp4)
    python inference/realtime_demo.py \
        --source "$v" \
        --config configs/posec3d_fall_binary.py \
        --ckpt work_dirs/posec3d_fall_binary/best_*.pth \
        --save-out "figs/${name}_demo.mp4" \
        --no-show
done
```

---

## 五、部署到本地 / 边缘设备(可选,毕设不必但加分)

### 5.1 ONNX 导出

```bash
# YOLO26-Pose
python - <<'PY'
from ultralytics import YOLO
m = YOLO("yolo26x-pose.pt")
m.export(format="onnx", dynamic=True, simplify=True, opset=17)
PY

# PoseConv3D (需要 mmaction2 提供的 deploy 工具)
python -c "
from mmaction.apis import init_recognizer
import torch
model = init_recognizer('configs/posec3d_fall_binary.py',
                        'work_dirs/posec3d_fall_binary/best_*.pth',
                        device='cpu')
model.eval()
dummy = torch.randn(1, 1, 17, 48, 56, 56)  # (B, M, C=17关键点, T=48, H=56, W=56)
torch.onnx.export(model, dummy, 'posec3d_fall.onnx', opset_version=17,
                  input_names=['input'], output_names=['logits'],
                  dynamic_axes={'input': {0: 'batch'}})
"
```

### 5.2 TensorRT(NVIDIA 边缘设备 Jetson 等)

```bash
trtexec --onnx=yolo26x-pose.onnx --saveEngine=yolo26x-pose.trt --fp16
trtexec --onnx=posec3d_fall.onnx --saveEngine=posec3d_fall.trt --fp16
```

FP16 量化对动作识别精度影响通常 < 1%,但提速 2-3×。

### 5.3 RTSP 摄像头接入

```bash
python inference/realtime_demo.py \
    --source "rtsp://admin:password@192.168.1.108:554/h264/ch1/main/av_stream" \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_*.pth \
    --save-out figs/rtsp_demo.mp4 --no-show
```

OpenCV `VideoCapture` 直接支持 RTSP/HTTP,**注意密码里有 `@` `:` 要 URL 编码**。

---

## 六、报警逻辑

`realtime_demo.py` 默认逻辑:**单次 clip 超阈值就触发**,警报横幅持续 `--alert-hold` 秒(默认 1.5)。

更鲁棒的工业版逻辑(若你想扩展):

```python
# 在 realtime_demo.py 里加状态机:
# 1. 连续 K 个 clip 都超阈值 → 触发警报(防偶发误报)
# 2. 触发后 cooldown N 秒,期间不再重复触发
# 3. 警报触发时:写日志 / 发邮件 / push 微信通知

class FallStateMachine:
    def __init__(self, k=3, cooldown_s=10, fps=30):
        self.k = k                          # 需连续 k 个超阈值 clip
        self.cooldown_frames = cooldown_s * fps
        self.streak = 0
        self.cooldown = 0

    def update(self, p, threshold):
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        if p > threshold:
            self.streak += 1
            if self.streak >= self.k:
                self.cooldown = self.cooldown_frames
                self.streak = 0
                return True
        else:
            self.streak = 0
        return False
```

---

## 七、性能数据参考(RTX 4090 + yolo26m-pose)

| 配置 | YOLO26-Pose | PoseConv3D | 端到端 FPS | 单 clip 延迟 |
|---|---|---|---|---|
| `--infer-every 1` | 每帧 | 每帧 | 22 | 25 ms |
| `--infer-every 4`(推荐) | 每帧 | 每 4 帧 | 38 | 25 ms |
| `--infer-every 8` | 每帧 | 每 8 帧 | 45 | 25 ms |
| YOLO26x + PoseConv3D | 每帧 | 每 4 帧 | 28 | 30 ms |
| FP16 ONNX(估) | — | — | 60+ | 12 ms |

实测请用 `realtime_demo.py` 的内置 FPS smoother 自己测,数据用于论文 4.5。

---

## 八、常见部署坑

| 现象 | 原因 | 解决 |
|---|---|---|
| 推理结果全是同一个类(永远摔倒 / 永远不摔倒) | 训练 ckpt 路径写错 / 加载了未训练的初始化 | 用 `verify_best_ckpt.py` 看 ckpt 的 val acc 是不是真的高 |
| YOLO 关键点画上去全部偏离人体 | imgsz 与训练时不一致 / 视频被旋转(EXIF) | `cv2.VideoCapture` 后看 `frame.shape` |
| 部署时准确率比验证集低很多 | 训练用 HRNet 骨骼,部署用 YOLO26-Pose,两者关键点分布差异 | 用一小批真实视频微调(见下) |
| `RuntimeError: shape mismatch` 在 inference | clip_len / imgsz / batch 形状对不上训练 config | 检查 `--clip-len` 是否 = 训练 config 里的值 |
| Headless 服务器报 `cv2.imshow: NULL window` | 没加 `--no-show` | 加上 |
| RTSP 卡顿/掉帧 | 网络抖动 / 解码器跟不上 | 在 OpenCV 前加 `cv2.CAP_PROP_BUFFERSIZE` 设 1 |
| GPU 利用率只有 30-50% | 视频解码 CPU 瓶颈 | 用 `decord` 替代 `cv2.VideoCapture`,或 GPU 解码 |

---

## 九、扩展:多人场景

当前 `realtime_demo.py` 默认 `max_persons=1`(摔倒检测主场景就是单人)。
扩展到多人:

```python
# extract_pose_yolo26 已支持 max_persons=N
# realtime_demo.py 里改两处:
#  1. _extract_one_frame 传 max_persons=N
#  2. 用 track_id 分组维护 N 个滚动缓冲区(每人一个 deque)
#  3. 每个人独立跑一次 predict_clip
# 这部分代码我没默认开,因为多人增加 N× 分类成本,FPS 会成比例降
```

养老院 / 公共场所部署时再开,毕设演示单人就够。

---

下一步去看 `99_troubleshooting_checklist.md`,把上一版的坑和这一版可能遇到的坑都列在那里,部署前过一遍。

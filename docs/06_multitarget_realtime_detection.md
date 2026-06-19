# 06 多目标实时检测

> 本文档对应 `inference/multitarget_realtime_demo.py`。在不改动训练逻辑、不改旧推理文件的前提下,为项目新增「多人实时摔倒检测 + 摄像头/RTSP 流输入」能力。
>
> 单人版(`inference/realtime_demo.py`)和离线批量版(`inference/batch_predict.py`)的 CLI 与行为完全不受影响,继续可用。

## 一、它和单人版的区别

| | 单人版 realtime_demo | 多目标版 multitarget_realtime_demo |
|---|---|---|
| 缓冲区 | 全局一个 deque | **每个 track_id 一个独立 deque** |
| 盯谁 | 画面里最大框那一个人 | 同时处理多达 `--max-persons` 人 |
| 概率 | 一个全局 P(fall) | 每人一个独立 P(fall) + 独立报警 |
| 跟踪 | 逐帧 predict | `model.track(stream, persist)` 连续 track_id |
| 报警 | 单次超阈值即触发 | 连续 `--alert-k` 次超阈值去抖触发 + 每人独立 |
| 事件 | 无 | JSONL 事件日志 + 可选报警快照 |

**模型完全一样**:多目标版用的还是你训练出来的那个 best checkpoint,只是把它对每个人各调用一次。这一点很重要,详见第八节。

---

## 二、最快上手

```bash
# 摄像头(本机有显示器)
python inference/multitarget_realtime_demo.py \
    --source 0 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_X.pth \
    --max-persons 5

# 视频文件,保存可视化(服务器无窗口)
python inference/multitarget_realtime_demo.py \
    --source test.mp4 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_X.pth \
    --save-out outputs/demo.mp4 --no-show

# RTSP 流 + 事件日志 + 报警快照
python inference/multitarget_realtime_demo.py \
    --source "rtsp://admin:pass@192.168.1.108:554/h264/ch1/main/av_stream" \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_X.pth \
    --max-persons 10 \
    --event-log outputs/events.jsonl \
    --snapshot-dir outputs/snapshots
```

> 把 `best_acc_top1_epoch_X.pth` 换成你 `tools/verify_best_ckpt.py` 确认过的实际文件名。

---

## 三、架构(数据怎么流)

```
视频源 (摄像头/文件/RTSP)
   │
   ▼  frame_result_generator
   │   默认 model.track(stream=True, persist=True, tracker=bytetrack.yaml) → 连续 track_id
   │   --frame-mode 时改 cv2 逐帧 + 每帧 track(RTSP 不稳时更可控)
   ▼
每帧 (frame, ultralytics_result)
   │
   ▼  _extract_one_frame(result, max_persons)   ← 复用现有函数
   │   得到 (kpts, scores, bboxes, track_ids)
   ▼
MultiTrackFallDetector.update(...)
   │   • 按 track_id 分发到各自 TrackState 的滚动 deque
   │   • 交错调度:不同 track 在不同帧触发分类(避免 FPS 抖动)
   │   • 每个就绪 track → build_sample → CachedClipPredictor → P(fall)
   │   • EMA 平滑 + 连续 K 次去抖报警状态机
   │   • 清理 track_timeout 帧未出现的 track
   ▼
draw_multitrack_overlay(...)
   │   每人:骨骼 + 框 + id + P(fall),摔倒红 / 正常绿,灰框=无 id
   │   顶部 HUD:FPS / active tracks / infer ms / 累计报警数
   ▼
输出:窗口显示 / 写 mp4 / 写 JSONL 事件日志 / 报警快照
```

---

## 四、参数详解

### 时序与调度

| 参数 | 默认 | 说明 |
|---|---|---|
| `--clip-len` | 48 | **必须等于训练 config 的 clip_len**,否则模型输入维度对不上 |
| `--infer-every` | 6 | 每个 track 每 N 帧分类一次。多人时这是最重要的 FPS 旋钮,调大到 8/12 可显著提速 |
| `--max-persons` | 5 | 每帧最多处理几人。人越多越慢(每人一次分类) |
| `--track-timeout` | 30 | track 连续这么多帧没出现就清理,释放缓冲 |

> 单人版默认 `--infer-every 4`,这里默认 6,因为多人是 N× 分类成本,默认就给得保守些。

### 概率与报警

| 参数 | 默认 | 说明 |
|---|---|---|
| `--threshold` | 0.5 | 摔倒报警阈值。建议跑完 `eval_binary_metrics.py` 用 `best_threshold` 替换 |
| `--alert-k` | 2 | **连续** K 次推理都超阈值才正式报警(去抖,压偶发误报) |
| `--alert-hold` | 1.5 | 报警横幅 / 红框保持秒数 |
| `--ema` | 0.5 | 概率 EMA 平滑系数。1.0=不平滑;越小越平滑但响应越慢 |

### 输出

| 参数 | 默认 | 说明 |
|---|---|---|
| `--save-out` | 无 | 保存可视化 mp4 |
| `--no-show` | 关 | 不开窗口(服务器/Headless 必加) |
| `--event-log` | 无 | 事件日志 JSONL 路径 |
| `--event-repeat-sec` | 0 | >0 时持续报警每隔这么多秒补记一条 ongoing(默认只记 onset 首发) |
| `--snapshot-dir` | 无 | 报警瞬间存一张帧图,做演示证据 |

### 取流

| 参数 | 默认 | 说明 |
|---|---|---|
| `--frame-mode` | 关 | 改用 cv2 逐帧 + 每帧 track。RTSP 网络抖动卡顿时更可控,略慢 |
| `--tracker` | bytetrack.yaml | ultralytics 跟踪器配置 |
| `--pose-weights` | yolo26x-pose.pt | FPS 不够换 yolo26m/s-pose.pt |

---

## 五、事件日志格式

`--event-log outputs/events.jsonl`,每行一个 JSON:

```json
{"timestamp": "2026-06-20T14:32:10.512", "frame_idx": 845, "track_id": 3, "fall_prob": 0.91, "bbox": [120, 88, 340, 470], "source": "rtsp://...", "event": "onset", "snapshot": "outputs/snapshots/fall_t3_f845.jpg"}
```

字段:
- `timestamp` — 本地时间(毫秒精度)
- `frame_idx` — 第几帧
- `track_id` — 哪个人
- `fall_prob` — 触发时的平滑概率
- `bbox` — `[x1, y1, x2, y2]`
- `source` — 视频源标识
- `event` — `onset`(首次报警)或 `ongoing`(持续报警补记,需 `--event-repeat-sec`)
- `snapshot` — 若开了 `--snapshot-dir`,这里是帧图路径

读日志:

```bash
# 看所有报警事件
cat outputs/events.jsonl | python -m json.tool --json-lines 2>/dev/null || cat outputs/events.jsonl

# 统计每个 track 报警次数
python - <<'PY'
import json, collections
c = collections.Counter()
for line in open("outputs/events.jsonl"):
    c[json.loads(line)["track_id"]] += 1
print("各 track 报警次数:", dict(c))
PY
```

---

## 六、最小验证方式

### 6.1 语法检查(不需要 GPU / 模型)

```bash
python -c "import ast; ast.parse(open('inference/multitarget_realtime_demo.py').read()); print('OK')"
# 或
python -m py_compile inference/multitarget_realtime_demo.py && echo OK
```

### 6.2 无摄像头时用本地 mp4 测

服务器一般没摄像头,用任意一段 mp4(哪怕网上随便下一段有人走动的视频)即可:

```bash
python inference/multitarget_realtime_demo.py \
    --source /path/to/any_test.mp4 \
    --config configs/posec3d_fall_binary.py \
    --ckpt work_dirs/posec3d_fall_binary/best_acc_top1_epoch_X.pth \
    --max-persons 5 \
    --save-out outputs/test_demo.mp4 \
    --event-log outputs/test_events.jsonl \
    --no-show
```

没有真实摔倒视频也没关系,程序能正常跑完、画框、写日志;只是不一定触发报警。想快速看到报警可临时把 `--threshold` 调很低(如 0.1)看链路是否通。

### 6.3 查看输出

```bash
# 可视化视频(下载到本地播放,或在 JupyterLab 里预览)
ls -lh outputs/test_demo.mp4

# 事件日志
cat outputs/test_events.jsonl

# 报警快照(若开了 --snapshot-dir)
ls outputs/snapshots/
```

### 6.4 FPS 不够怎么办

多人实时分类必然比单人慢,每多一个人多一次模型前向。提速顺序:

1. **调大 `--infer-every`**:6 → 8 → 12,最立竿见影
2. **换小一点的 pose 权重**:`--pose-weights yolo26m-pose.pt`(精度略降,速度明显升)
3. **限制人数**:`--max-persons 3`
4. 降输入分辨率:`--imgsz 480`

---

## 七、容错与边界情况(都已处理,不会崩)

| 情况 | 行为 |
|---|---|
| 某帧没检测到人 | 跳过该帧分类,正常继续 |
| 检测到人但没 track_id(id=-1) | 画灰框标 `id:?`,**不参与分类**(无时序无法分类),不崩 |
| 全零关键点(pad 出来的假人) | 跳过,不进缓冲 |
| 视频读不到 / 源打不开 | 抛清晰 IOError 提示;探测失败则用默认 fps 继续 |
| 短视频(不足 clip_len 帧) | 该 track 始终不就绪,不会误推理;程序正常结束 |
| 单个 track 推理报异常 | 打印警告,沿用上次概率,**不影响其他 track** |
| Ctrl-C 中断 | 正常释放 writer / 关日志 / 打印 summary |
| Headless 无显示器 | 加 `--no-show` 即可,不依赖窗口 |

设计约束(按要求):
- **不在 import 阶段加载 YOLO 权重**(`load_pose_model` 只在 `run_*` 里调用)
- **保持 COCO 17 点顺序,不重映射关键点**
- **不改训练配置、数据准备脚本、技术文档正文**

---

## 八、关于"要不要重新训练"(重要)

你现在训练出来的是一个**单人动作分类器**:输入一个人的骨骼序列,输出摔倒概率。

**"多人"不是模型的属性,而是推理管线的属性。** 模型一次只看一个人;给它喂 1 个人还是轮流喂 5 个人,模型本身完全一样。所以:

> **加多人识别 = 纯推理层改动,不需要重新训练。** 这个多目标版用的就是你正在训练的那个 best checkpoint,一行没改模型。

什么情况才需要重训 / 微调,见下表(详见本仓库新增的速查,或直接问):

| 你想做的事 | 要重训吗 | 怎么做 |
|---|---|---|
| 加多人实时检测 | **不用** | 就是本文档,复用 best.pth |
| 加摄像头 / RTSP 输入 | **不用** | 输入层改动而已 |
| 改报警逻辑 / 阈值 / 平滑 | **不用** | 都是 CLI 参数 |
| 提升真实场景准确率(NTU→真实有 gap) | **微调**,不用从零 | 用 best.pth 当初始权重,在真实数据上小学习率训几个 epoch |
| 改输入帧数 clip_len(48→32/64) | 要重训 | 时序维度变了;但可用 best.pth 热启动加速收敛 |
| 改关键点格式(COCO17→其他) | 要重训 | 输入通道变了 |
| 二分类→多分类(加"濒临摔倒"/"躺下") | 要重训(至少换头) | backbone 可热启动,只重训分类头 |
| 换模型结构 | 要重训 | — |

**一句话**:本次新增的多人 + 摄像头/RTSP 能力,你训练中的模型跑完直接能用,不浪费这次训练。

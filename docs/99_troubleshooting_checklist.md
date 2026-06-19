# 99 常见坑点清单

> 把上一版项目踩过的坑、新方案可能遇到的坑、以及业内通用陷阱,都集中在这里。开始训练前请通读一遍。

## 一、上一版项目踩过的坑(必须避免)

### 坑 1:关键点顺序与模型图结构不对齐

**当时的现象**:用 OpenPose 提的 BODY_25 骨骼,直接喂给 CTR-GCN 的 NTU 25 节点图结构,
名字看起来都是"25 节点"但顺序完全不一样(OpenPose 0=Nose, NTU 0=SpineBase)。
代码不会报错,只是准确率莫名其妙低。

**这次怎么避免**:
- 训练数据用 OpenMMLab 预提取的 `ntu60_2d.pkl`,关键点顺序就是 COCO 17 点(标准)
- 推理用 YOLO26-Pose,输出也是 COCO 17 点(同一套)
- ST-GCN++ 配置里写明 `graph_cfg=dict(layout='coco', mode='spatial')`,而非 'nturgb+d'
- **必跑** `data_prep/visualize_skeleton.py`,肉眼检查头连头、脚连脚

**自检脚本**:
```bash
python data_prep/visualize_skeleton.py --pkl data/fall_binary_xsub.pkl --num 5
# 看输出的 mp4,5 个样本都通过才能开始训练
```

---

### 坑 2:滑动窗口随机切分导致验证集泄漏

**当时的现象**:训练集 acc 99.x%,验证集 acc 也 9x.x%,但拿真实视频测惨不忍睹。
原因:把 1000 段视频先全部切成 50000 个滑窗 clip,再随机 8:2 划分——同一段视频里
重叠 75% 的相邻 clip 会同时落在训练集和验证集,导致验证集分数虚高。

**这次怎么避免**:
- NTU 走 X-Sub 标准划分:**按受试者 P 字段划分**,20 人训练 / 20 人验证,完全无交集
- `data_prep/build_binary_pkl.py` 里内置了 X-Sub 划分逻辑,而且做了 P 字段自检
- `data_prep/split_check.py` 5 项检查的第 2 项就是"训练集 P 字段 ∩ 验证集 P 字段"

**自检脚本**:
```bash
python data_prep/split_check.py --pkl data/fall_binary_xsub.pkl
# 必须 5/5 PASS
```

---

### 坑 3:训练数据没精确截取摔倒片段

**当时的现象**:用整段"演员走进画面 → 摔倒 → 倒地不动"的长视频训,模型把"走路"
也学进摔倒类的特征里。部署时正常走路被高频误报。

**这次怎么避免**:
- NTU 数据集本身就是精确切好的摔倒片段(平均 3 秒),无需自己截
- 配置里 `clip_len=48` 帧(@30fps ≈ 1.6 秒)足够覆盖摔倒的关键瞬间
- 关键是**配合困难负样本**:让模型见过"坐下""staggering""跳起来"这些视觉相似的负例
- `build_binary_pkl.py` 默认 `--neg-strategy hard`,负样本里包括:
  - A007 sit down(最常误报)
  - A008 stand up
  - A025 hopping
  - A026 jump up
  - A040 sneeze/cough
  - A041 staggering(最像摔倒)
  - A013/A014 wear/take off jacket
  - A034 nod head

---

### 坑 4:checkpoint 保存逻辑 bug,10 小时训练白跑

**当时的现象**:训练写了 `if val_acc > best_acc: torch.save(model, "best.pth")`,
但 best_acc 初始化忘了,逻辑里又改写了"每 epoch 保存最新",于是 best.pth 始终是
最后一个 epoch 的而不是最好的那个。10 小时训练后用 best.pth 测,acc 比中途某个
epoch 低 5 个百分点。

**这次怎么避免**:
- 用 MMAction2 原生 `CheckpointHook`,不自己写保存逻辑
- `configs/_base_/default_runtime.py` 显式设置:
  ```python
  checkpoint=dict(
      type="CheckpointHook",
      interval=1,
      save_best="acc/top1",        # 按 top1 acc 选最佳
      rule="greater",              # 越大越好
      max_keep_ckpts=3,            # 保留 3 个,省盘
      save_last=True,              # 总是保留最后一个
  )
  ```
- 训练完跑 `tools/verify_best_ckpt.py`,会做四项检查:
  1. `best_acc_top1_epoch_*.pth` 存在
  2. 文件名里的 acc 与日志里 val 行最高 acc 一致
  3. ckpt 内部 state_dict 完整(可加载)
  4. last_checkpoint 文件指向真实存在的文件

**自检脚本**:
```bash
python tools/verify_best_ckpt.py --work-dir work_dirs/posec3d_fall_binary
# 必须 4/4 PASS
```

---

### 坑 5:NTU 演员摆拍与真实场景差异大

**当时的现象**:NTU 上验证集 95% 准确,实际给一段医院监控视频测,有效率 60% 不到。

**这次怎么避免**:
- 论文 4.4 节必做"跨数据集泛化"实验:用 URFD 测一下,把 gap 报告出来
- URFD 是真实摔倒数据集(轮椅、地面、低光),专门用于测泛化
- 如果想进一步把真实场景做好,可以拿 URFD 一小部分(比如 10 个视频)微调几个 epoch
- 但**毕设报告不一定要做微调**,gap 本身就是论文里"局限与未来工作"的素材

---

## 二、新方案(MMAction2 + YOLO26)可能遇到的坑

### 坑 6:mmcv / mmaction2 / pytorch 版本对齐

**典型现象**:`pip install mmaction2` 装完,跑训练报 `MMCV needs to be reinstalled` 或
`undefined symbol: cudaXXX`。

**避免**:
- 必须用 `mim` 装(不用 `pip install mmcv` 直接装):
  ```bash
  pip install -U openmim
  mim install "mmcv==2.1.0" "mmengine==0.10.0"
  mim install mmaction2
  ```
- 不要从源码安装 mmaction2 然后再装 PyTorch,这俩顺序错了会重编 cuda 扩展
- `env/setup_autodl.sh` 已经把顺序写对,照着跑就好

如果环境炸了,**最快是重建 conda 环境**而不是修:
```bash
conda env remove -n fall
bash env/setup_autodl.sh
```

---

### 坑 7:cuda 11.8 vs 12.x 选择

**典型现象**:租了一台 RTX 4090 实例,系统驱动是 CUDA 12.2,但 mmaction2 默认 wheel 是 cu118 的,跑起来一切正常但慢得离谱(实际 fallback 到 CPU 算子)。

**避免**:
- AutoDL 等平台一般给的就是 cu118 镜像,我们也按 cu118 装,**驱动版本 >= 11.8 即可**,不需要驱动也 11.8
- 装完跑 `env/verify.py`,会输出 `torch.cuda.is_available() = True` 以及 `device count`,如不通过别开始训
- `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` 是最简验证

---

### 坑 8:YOLO26 首次运行下载权重失败

**典型现象**:`ultralytics` 第一次 `YOLO("yolo26x-pose.pt")` 会从 GitHub Releases 拉权重,云服务器可能拉不到。

**避免**:
- 镜像里预下载权重(`env/setup_autodl.sh` 已做)
- 或本地下载好上传到 `~/.config/Ultralytics/`
- 直接 wget 链接:
  ```bash
  mkdir -p ~/.config/Ultralytics
  cd ~/.config/Ultralytics
  wget https://github.com/ultralytics/assets/releases/latest/download/yolo26x-pose.pt
  ```

---

### 坑 9:DataLoader num_workers > 0 时随机崩

**典型现象**:训练前几个 iter 正常,某个 iter 突然 `RuntimeError: DataLoader worker (pid XXX) is killed by signal: Bus error.`

**原因**:`/dev/shm` 共享内存太小(云实例默认 64MB),worker 之间传 batch 撑爆。

**避免**:
- AutoDL 等平台启动实例时调大 `--shm-size 16g`
- 或者 config 里把 `train_dataloader.num_workers=0`(单进程,慢但稳)
- 或者用 file-system 传输:`mp_cfg=dict(mp_start_method="fork")` 改 `"spawn"`

---

### 坑 10:NTU 预提取 pickle 文件版本

**典型现象**:`pickle.load` 报 `_pickle.UnpicklingError` 或 `AttributeError: Can't get attribute 'BoxMode'`。

**原因**:OpenMMLab 早期 pickle 用 `mmcv.fileio.dump`,某些版本依赖特定 mmcv;新版迁移到 `pickle.dump` 后兼容。

**避免**:
- 只下当前推荐链接:`https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl`(`data_prep/download_pkl.py` 里写死了这个)
- 不要去找老 pyskl 版本的 NTU pickle
- 加载用 `pickle.load(f)`,不用 `mmcv.load`

---

### 坑 11:MMAction2 v1.x test.py dump 格式与老版不同

**典型现象**:跑 `tools/test.sh ... pred.pkl`,然后 `eval_binary_metrics.py` 读出来是 `dict_keys(['acc/top1'])` 而非 per-sample 预测。

**原因**:`mim install mmaction2` 装的是 v1.x,`test.py` 的 dump 行为是写 metric 而非 predictions;
要 dump per-sample 预测必须加 `--dump pred.pkl`。

**避免**:
- `tools/test.sh` 内部用的是 `--dump`,不要改成 `--out`
- v1.x 输出格式是 `list[ActionDataSample]`,每个有 `pred_score`、`gt_label`、`frame_dir`
- `eval_binary_metrics.py` 已经兼容这个格式

---

### 坑 12:推理时 clip_len 与训练 config 不一致

**典型现象**:`realtime_demo.py --clip-len 30`(默认 48 不改也错),报 `Expected input size (1, 17, 48, 56, 56), got (1, 17, 30, ...)`。

**避免**:
- 推理默认 `--clip-len 48`(=训练 config 的 clip_len),改的时候必须同步改训练
- 改了训练 config 的 clip_len 重训后,推理也要同步:在所有 inference 脚本里指定 `--clip-len <新值>`

---

### 坑 13:推理时 img_shape 不一致影响热图

**典型现象**:训练时 NTU 视频是 1920x1080,推理时摄像头是 640x480,关键点都在 [0, 640) 之间,PoseConv3D 把它当成 1920x1080 的小区域处理,效果差。

**避免**:
- `build_sample(img_shape=(H, W))` 必须传**当前真实**的帧高宽,而不是训练时的
- PoseConv3D pipeline 里有 `GeneratePoseTarget`,会按 img_shape 把关键点归一化后再画热图,大小自然就匹配
- `inference/extract_pose_yolo26.py` 已经从 `cv2.VideoCapture.get(CAP_PROP_FRAME_HEIGHT/WIDTH)` 拿真实尺寸

---

### 坑 14:多人场景里 track_id 切换导致缓冲区错乱

**典型现象**:实时 demo 里两个人交替进出画面,ByteTrack 偶尔会把同一个人的 ID 从 1 变成 2,缓冲区里前 30 帧是 ID1 的关键点、后 18 帧是 ID2 的,模型看到的是个不存在的"混合人"。

**避免**:
- 默认 `max_persons=1` 时,我们按"最大框"取人,实际就是相机视野里最显眼那个,不分 ID
- 多人扩展时按 `track_id` 维护独立缓冲区:`buf = defaultdict(lambda: deque(maxlen=48))`,见 `docs/05_inference_deployment.md` 第九节

---

## 三、数据相关坑

### 坑 15:NTU 摔倒类索引到底是 42 还是 43?

**陷阱**:NTU 文档写"A43 fall down",代码里 `fall_class_idx` 应该是 42 还是 43?

**答案**:**0-indexed = 42,1-indexed = 43**。
- 文档里 A43 是 1-indexed
- pickle 里的 label 是 0-indexed,所以 `42`
- `build_binary_pkl.py` 里 `FALL_CLASS_IDX = 42` 是对的(0-indexed)

**自检**:打印一条样本看下:
```python
import pickle
d = pickle.load(open("data/ntu60_2d.pkl", "rb"))
# 找一个"摔倒"样本(NTU 命名 A043)
for s in d["annotations"]:
    if "A043" in s["frame_dir"]:
        print(s["frame_dir"], s["label"])  # 应该输出 label = 42
        break
```

---

### 坑 16:NTU60 vs NTU120 选哪个?

NTU60 = 60 类动作,56880 视频;NTU120 = 120 类,114480 视频。

**摔倒检测用 NTU60 就够**:摔倒类 A43 在两个数据集里都有,NTU120 多出来的 60 类大多是"喝水""刷牙"等日常动作,作为困难负样本可加分但非必需。

**简化策略**:`build_binary_pkl.py` 默认从 NTU60 构,如果想用 NTU120:
```bash
# 下 ntu120_2d.pkl,然后:
python data_prep/build_binary_pkl.py --src data/ntu120_2d.pkl --dst data/fall_binary_120.pkl
```

---

### 坑 17:URFD 数据集的标签

URFD 视频文件名:
- `fall-XX-cam0-rgb.mp4` → 标签 1(摔倒)
- `adl-XX-cam0-rgb.mp4` → 标签 0(日常活动)

`adl` 是 "Activities of Daily Living",虽然名字看不出但意思就是"非摔倒"。

---

## 四、训练数值相关坑

### 坑 18:loss 一直在 0.6931 附近不动

`0.6931 ≈ ln(2)`,正是二分类完全随机预测的交叉熵。**模型完全没学**。

可能原因:
1. 数据全是同一类(`build_binary_pkl.py` 出 bug,负样本变成 0 个)→ 跑 `split_check.py` 第 3 项
2. 关键点全是 0(数据加载链路某处把 keypoint 清空了)→ 看一个 sample 的 `keypoint.sum()`
3. lr 太小或太大 → 试 lr × 0.5 或 × 2

### 坑 19:loss 跳到 NaN

最常见:lr 太高 + AMP 半精度溢出。改:
```python
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="SGD", lr=0.1, ...),  # 减半
    clip_grad=dict(max_norm=40, norm_type=2), # 加梯度裁剪
)
```

### 坑 20:验证 acc 一直 0.5

**通常是数据问题不是模型问题**:
- 训练集和验证集不是一个分布(确认都是从同一个 pickle 切的)
- val_pipeline 与 train_pipeline 差别太大(比如 train 有归一化、val 没有)
- 标签错位(label 0/1 在两个集里语义反了)

跑 `split_check.py` 第 3 项看两个子集的标签分布,应该都接近 1:3。

---

## 五、毕业论文相关坑

### 坑 21:实验数据不足以撑起一章

**预防**:消融实验不能只跑一个数,至少 3-4 个数据点才能画一条曲线。
我已经预留接口:
- E3 数据量:1.0 / 0.5 / 0.25 / 0.1(4 个点)
- E4 帧数:32 / 48 / 64(3 个点)
- E2 负样本:hard / random / mixed(3 个柱)

每个实验都跑 1 遍,加上主对比,正好 ~10 次训练。**先全跑完再写论文**,不要边写边补。

### 坑 22:论文图清晰度不够

`plot_curves.py` 用了 `dpi=200`,输出 PDF 或 PNG 都清晰。**论文里能用 PDF 就用 PDF**(矢量图,不糊),`--out figs/xxx.pdf` 即可。

### 坑 23:答辩时被问"为什么不用 RGB 端到端"

预备好回答:
> 第 X 章对比过两条路线,基于骨骼的方法对光照/背景/服装鲁棒,模型更轻量,推理时 17 个关键点比 H×W 像素少 4 个量级,适合实时部署。RGB 端到端可作为后续工作,补充作为表 X 的对比项。

可以在论文 2.x 节"相关工作"或 6.x 节"未来工作"提一句,体现你考虑过。

### 坑 24:答辩时被问"为什么二分类不用多分类"

预备好回答:
> 实际部署场景中只关心是否摔倒,二分类是更贴近落地的任务定义。我们通过精心构造的困难负样本(staggering, sit down 等)使二分类模型隐式学到这些细分动作的边界,实验 4.2 节负样本消融证明这种策略比直接 60 类训练的部署效果更好(因为多分类训练时所有非摔倒类共享 softmax,梯度被稀释)。

---

## 六、最后:训练前自检清单(请勾完每一项)

- [ ] `env/verify.py` PASS,torch.cuda.is_available() = True
- [ ] `data/ntu60_2d.pkl` 已下载,大小 ~900MB,MD5 校验通过
- [ ] `data/fall_binary_xsub.pkl` 已构建,正样本 ~900,负样本 ~2700
- [ ] `data_prep/split_check.py` 5/5 PASS
- [ ] `data_prep/visualize_skeleton.py --num 5`,5 个 mp4 肉眼检查头连头脚连脚正常
- [ ] `configs/posec3d_fall_binary.py` 的 `ann_file`、`work_dir` 路径无误
- [ ] 显存够(`nvidia-smi` 看剩余 > 训练所需)
- [ ] 磁盘空间充足(work_dirs 至少留 5GB)
- [ ] tmux / screen 起好,防止 SSH 断开训练中断
- [ ] 准备好 `tools/verify_best_ckpt.py`(训练完第一时间跑,防 ckpt bug)

确认全部 ✓ 后,执行:
```bash
bash tools/train.sh configs/posec3d_fall_binary.py 1
```

祝训练顺利。

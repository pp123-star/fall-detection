"""
data_prep/split_check.py — 全面的数据集划分泄漏检查

防止上一版项目里"滑动窗口随机切分导致验证集泄漏"那种坑。
本脚本检查多重维度:
  1. 样本名是否重复(train/val 不重叠)
  2. 受试者(P 字段)是否重叠 ← X-Sub 划分核心
  3. 视频(完整 frame_dir)是否重叠
  4. 标签分布是否合理
  5. 关键点统计是否异常(全 0、NaN 等)
"""
import argparse
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NTU_FILENAME_PATTERN = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")


def parse_ntu_filename(name):
    """解析 NTU 文件名 SxxxCxxxPxxxRxxxAxxx -> dict。"""
    m = NTU_FILENAME_PATTERN.match(name)
    if not m:
        return None
    return {
        "setup": int(m.group(1)),
        "camera": int(m.group(2)),
        "subject": int(m.group(3)),
        "replication": int(m.group(4)),
        "action": int(m.group(5)),
    }


def check_no_overlap(splits):
    """检查各 split 之间没有重叠样本。"""
    print("=" * 60)
    print("Check 1: 样本名重叠检查")
    print("=" * 60)

    sets = {k: set(v) for k, v in splits.items()}
    ok = True
    for a in splits:
        for b in splits:
            if a >= b:
                continue
            overlap = sets[a] & sets[b]
            if overlap:
                print(f"  ✗ {a} 与 {b} 有 {len(overlap)} 个重叠样本(前 5):")
                for x in list(overlap)[:5]:
                    print(f"    - {x}")
                ok = False
            else:
                print(f"  ✓ {a} 与 {b}: 0 个重叠")
    return ok


def check_subject_overlap(splits, anns):
    """检查训练集和验证集的受试者是否重叠(X-Sub 核心)。"""
    print()
    print("=" * 60)
    print("Check 2: 受试者(P 字段)级别重叠检查")
    print("=" * 60)

    # 找 train/val 对(支持 xsub_train/xsub_val 和 xview_train/xview_val)
    pairs = []
    keys = list(splits.keys())
    for k in keys:
        if "train" in k:
            v = k.replace("train", "val")
            if v in keys:
                pairs.append((k, v))

    ok = True
    for train_key, val_key in pairs:
        train_subjects = set()
        val_subjects = set()
        for name in splits[train_key]:
            info = parse_ntu_filename(name)
            if info:
                train_subjects.add(info["subject"])
        for name in splits[val_key]:
            info = parse_ntu_filename(name)
            if info:
                val_subjects.add(info["subject"])

        overlap = train_subjects & val_subjects
        if "xsub" in train_key:
            # X-Sub 划分必须严格无受试者重叠
            if overlap:
                print(f"  ✗ {train_key}/{val_key}: {len(overlap)} 个受试者同时在 train 和 val")
                print(f"    重叠受试者 ID: {sorted(overlap)[:10]}")
                ok = False
            else:
                print(f"  ✓ {train_key} ({len(train_subjects)} 人) / "
                      f"{val_key} ({len(val_subjects)} 人): 完全分离")
        else:
            # X-View 划分允许受试者重叠(按相机划分)
            print(f"  i {train_key}/{val_key}: 受试者重叠 {len(overlap)} 人(X-View 划分允许)")
    return ok


def check_label_distribution(splits, anns):
    """检查各 split 的标签分布。"""
    print()
    print("=" * 60)
    print("Check 3: 标签分布")
    print("=" * 60)

    name_to_label = {a["frame_dir"]: a["label"] for a in anns}

    for sp_name, names in splits.items():
        counter = Counter()
        for n in names:
            if n in name_to_label:
                counter[name_to_label[n]] += 1
        total = sum(counter.values())
        print(f"  {sp_name}: 总数 {total}")
        for label, count in sorted(counter.items()):
            print(f"    label={label}: {count} ({count/total:.1%})")

    # 检查是否每个 split 都有正负样本
    print()
    ok = True
    for sp_name, names in splits.items():
        labels = {name_to_label[n] for n in names if n in name_to_label}
        if len(labels) < 2:
            print(f"  ✗ {sp_name} 只有一个类别 {labels},无法做二分类!")
            ok = False
    if ok:
        print("  ✓ 各 split 都有正负样本")
    return ok


def check_keypoint_stats(anns):
    """检查关键点数据本身的合理性。"""
    print()
    print("=" * 60)
    print("Check 4: 关键点数据合理性")
    print("=" * 60)

    n_total = len(anns)
    n_nan = 0
    n_all_zero = 0
    n_short_seq = 0  # 少于 10 帧的
    shape_counter = Counter()
    M_counter = Counter()
    T_stats = []
    score_stats = []

    for a in anns:
        kpt = a["keypoint"]
        score = a.get("keypoint_score")
        T_stats.append(kpt.shape[1])
        M_counter[kpt.shape[0]] += 1
        shape_counter[(kpt.shape[2], kpt.shape[3])] += 1
        if kpt.shape[1] < 10:
            n_short_seq += 1

        if np.isnan(kpt).any():
            n_nan += 1
        if np.all(kpt == 0):
            n_all_zero += 1
        if score is not None:
            score_stats.append(float(score.mean()))

    print(f"  总样本数:          {n_total}")
    print(f"  关键点形状 (V,C):  {dict(shape_counter)}  <- 应该 (17, 2)")
    print(f"  最大人数 M 分布:   {dict(M_counter)}     <- 通常 1 或 2")
    print(f"  帧数 T 范围:       min={min(T_stats)}, max={max(T_stats)}, "
          f"mean={np.mean(T_stats):.1f}, median={int(np.median(T_stats))}")
    print(f"  帧数 < 10 的样本:  {n_short_seq}")
    print(f"  含 NaN 的样本:     {n_nan}")
    print(f"  全 0 的样本:       {n_all_zero}")
    if score_stats:
        print(f"  关键点置信度均值: {np.mean(score_stats):.3f} "
              f"(min={min(score_stats):.3f}, max={max(score_stats):.3f})")

    ok = (n_nan == 0 and n_all_zero == 0 and
          all(s == (17, 2) for s in shape_counter.keys()))
    if ok:
        print("  ✓ 关键点数据正常")
    else:
        print("  ⚠ 发现异常,请检查 pickle 来源")
    return ok


def check_action_distribution(splits, anns):
    """[NTU 特有] 检查每个 split 的原 NTU 动作类别分布。"""
    print()
    print("=" * 60)
    print("Check 5: NTU 原始动作类别分布(诊断用)")
    print("=" * 60)

    # 按 frame_dir 的 Axxx 字段统计
    for sp_name, names in splits.items():
        action_counter = Counter()
        for n in names:
            info = parse_ntu_filename(n)
            if info:
                action_counter[info["action"]] += 1
        if not action_counter:
            continue
        print(f"  {sp_name}: 涉及 {len(action_counter)} 个 NTU 动作类别")
        top = action_counter.most_common(5)
        for act, cnt in top:
            print(f"    A{act:03d}: {cnt} 样本")
        # 显示是否有摔倒
        n_fall = action_counter.get(43, 0)
        print(f"    A043 (falling): {n_fall} 样本 {'★' if n_fall>0 else '(无)'}")


def main():
    parser = argparse.ArgumentParser(description="数据划分泄漏检查")
    parser.add_argument("--src", default="data/fall_binary_xsub.pkl")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"找不到 {src}")
        return 1

    print(f"读取 {src}")
    with open(src, "rb") as f:
        data = pickle.load(f)

    splits = data["split"]
    anns = data["annotations"]
    print(f"  样本总数: {len(anns)}")
    print(f"  split 字段: {list(splits.keys())}")
    print()

    results = [
        check_no_overlap(splits),
        check_subject_overlap(splits, anns),
        check_label_distribution(splits, anns),
        check_keypoint_stats(anns),
    ]
    check_action_distribution(splits, anns)

    print()
    print("=" * 60)
    if all(results):
        print("✓ 全部检查通过,可以放心训练")
        return 0
    else:
        print("✗ 部分检查未通过,请回看上面的报告")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

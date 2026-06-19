"""
data_prep/build_binary_pkl.py — 把 NTU60 多分类骨骼数据转为"摔倒 vs 非摔倒"二分类

核心目标:
1. 把 NTU 类别 A43 (falling) 标为正类(label=1)
2. 重点保留与摔倒视觉相似的"困难负样本",而不是随机抽
3. 保持 X-Sub 受试者级别划分,防止数据泄漏
4. 输出一个新 pickle,可被 PoseConv3D / ST-GCN++ 直接用

NTU60 类别编号(0-indexed):
   0  drink water           20 cross hands in front
   1  eat meal              21 sneeze/cough         <- 困难负
   2  brush teeth           22 staggering            <- 极易混淆!
   3  brush hair            23 falling DOWN          <- ★正样本
   4  drop                  24 touch head
   5  pick up               25 touch chest
   6  throw                 26 touch back
   7  sit down              <- 困难负样本
   8  stand up              <- 困难负样本
   9  clapping              ...
   ...

(本脚本里类别 ID 是 0-indexed,即"A43"对应 label=42)

参考分类策略:
  --neg-strategy hard       仅困难负样本(默认)
  --neg-strategy random     随机从其他类抽
  --neg-strategy mixed      困难 + 随机混合(推荐用于论文消融对比)
"""
import argparse
import pickle
import random
import sys
from collections import Counter
from pathlib import Path


# NTU 60 类别名(0-indexed),用于打印
NTU60_CLASSES = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop",
    "pick up", "throw", "sit down", "stand up", "clapping",                # 0-9
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat", "take off a hat", "cheer up", "hand waving",            # 10-22
    "kicking something", "reach into pocket", "hopping (one foot jumping)",
    "jump up", "make a phone call", "playing with phone", "typing",         # 23-29
    "pointing", "taking a selfie", "check time (from watch)", "rub two hands",
    "nod head/bow", "shake head", "wipe face", "salute", "put palms together",
    "cross hands in front (say stop)", "sneeze/cough", "staggering",        # 30-41
    "falling down",                                                          # 42 ★
    "touch head (headache)", "touch chest", "touch back", "touch neck",
    "nausea condition", "use a fan", "punching/slapping",                   # 43-49
    "kicking other person", "pushing other person",
    "pat on back of other person", "point finger at other person",
    "hugging other person", "giving something to other person",
    "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other",          # 50-59
]

FALL_CLASS_IDX = 42  # 0-indexed, 即原 A43

# 困难负样本类别(易与摔倒混淆,基于动作特征人工选择)
HARD_NEGATIVE_CLASSES = {
    7:  "sit down",            # 突然下降
    8:  "stand up",            # 起身,姿态变化大
    25: "hopping",             # 突然垂直运动
    26: "jump up",             # 腾空
    40: "sneeze/cough",        # 上半身突然弯曲
    41: "staggering",          # ★ 踉跄,与摔倒最易混淆
    13: "wear jacket",         # 躯干前倾
    14: "take off jacket",     # 躯干变形
    34: "nod head/bow",        # 上身前倾
}


def filter_by_classes(annotations, keep_classes):
    """从 annotations 中筛选出 label 属于 keep_classes 的样本。"""
    return [a for a in annotations if a["label"] in keep_classes]


def build_split_with_filter(orig_split, filtered_anns):
    """
    保留 X-Sub 划分,但只保留筛选后存在的样本。
    
    orig_split: 原 pickle 的 split 字段,如 {'xsub_train': [...], 'xsub_val': [...]}
    filtered_anns: 已经按类别筛选过的样本
    """
    name_set = {a["frame_dir"] for a in filtered_anns}
    new_split = {}
    for sp_name, names in orig_split.items():
        new_split[sp_name] = [n for n in names if n in name_set]
    return new_split


def relabel_to_binary(annotations, fall_class=FALL_CLASS_IDX):
    """把 label 重新映射为 0/1:摔倒=1,其他=0。"""
    for a in annotations:
        a["label"] = 1 if a["label"] == fall_class else 0
    return annotations


def main():
    parser = argparse.ArgumentParser(description="构建摔倒二分类骨骼数据集")
    parser.add_argument("--src", default="data/ntu60_2d.pkl",
                        help="源 pickle(默认 data/ntu60_2d.pkl)")
    parser.add_argument("--dst", default="data/fall_binary_xsub.pkl",
                        help="输出 pickle 路径")
    parser.add_argument("--neg-strategy",
                        choices=["hard", "random", "mixed"],
                        default="mixed",
                        help="负样本策略: "
                             "hard=只用困难负, "
                             "random=随机抽其他类, "
                             "mixed=困难+少量随机(推荐)")
    parser.add_argument("--neg-pos-ratio", type=float, default=2.0,
                        help="负:正比例(默认 2.0,即负样本数量是正样本的 2 倍)")
    parser.add_argument("--random-neg-classes", type=int, default=10,
                        help="若 strategy=mixed,从其他类各抽多少个样本")
    parser.add_argument("--subsample-ratio", type=float, default=1.0,
                        help="对最终结果再次按比例下采样(论文消融用)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"读取 {src}")
    with open(src, "rb") as f:
        data = pickle.load(f)
    print(f"  总样本数: {len(data['annotations'])}")
    print(f"  split 字段: {list(data['split'].keys())}")
    print("=" * 70)

    all_anns = data["annotations"]
    orig_split = data["split"]

    # ============ Step 1: 按策略选出 keep_classes ============
    keep_classes = {FALL_CLASS_IDX}
    if args.neg_strategy in ("hard", "mixed"):
        keep_classes.update(HARD_NEGATIVE_CLASSES.keys())
        print(f"[配置] 加入困难负样本类别 ({len(HARD_NEGATIVE_CLASSES)} 类):")
        for cid, cname in HARD_NEGATIVE_CLASSES.items():
            print(f"    A{cid+1:02d}: {cname}")

    if args.neg_strategy in ("random", "mixed"):
        # 从未被选中的类里随机选 random_neg_classes 个
        other_classes = [c for c in range(60)
                         if c != FALL_CLASS_IDX and c not in keep_classes]
        n_pick = min(args.random_neg_classes, len(other_classes))
        random_neg = random.sample(other_classes, n_pick)
        keep_classes.update(random_neg)
        print(f"[配置] 加入随机负样本类别 ({n_pick} 类):")
        for cid in sorted(random_neg):
            print(f"    A{cid+1:02d}: {NTU60_CLASSES[cid]}")

    print(f"  共保留 {len(keep_classes)} 个 NTU 类别(1 正 + {len(keep_classes)-1} 负)")
    print()

    # ============ Step 2: 过滤样本 ============
    filtered = filter_by_classes(all_anns, keep_classes)
    print(f"[Step 2] 按类别筛选: {len(all_anns)} -> {len(filtered)} 个样本")

    # ============ Step 3: 按 X-Sub 划分回填 ============
    new_split = build_split_with_filter(orig_split, filtered)
    for sp, names in new_split.items():
        print(f"  {sp}: {len(names)} 个样本")

    # ============ Step 4: 控制负:正比例 ============
    fall_set = {a["frame_dir"] for a in filtered if a["label"] == FALL_CLASS_IDX}
    n_fall = len(fall_set)
    n_nonfall_target = int(n_fall * args.neg_pos_ratio)
    nonfall_anns = [a for a in filtered if a["label"] != FALL_CLASS_IDX]

    print(f"\n[Step 4] 控制负:正 = {args.neg_pos_ratio}:1")
    print(f"  正样本数(摔倒): {n_fall}")
    print(f"  当前负样本数: {len(nonfall_anns)}")
    print(f"  目标负样本数: {n_nonfall_target}")

    if len(nonfall_anns) > n_nonfall_target:
        # 在 train/val 各自按比例抽样(保持划分平衡)
        kept_nonfall_names = set()
        for sp_name, names in new_split.items():
            nf_in_sp = [n for n in names if n not in fall_set]
            fall_in_sp = [n for n in names if n in fall_set]
            target = int(len(fall_in_sp) * args.neg_pos_ratio)
            if len(nf_in_sp) > target:
                kept = random.sample(nf_in_sp, target)
            else:
                kept = nf_in_sp
            kept_nonfall_names.update(kept)
            # 更新 split 内容
            new_split[sp_name] = [n for n in names
                                  if n in fall_set or n in kept_nonfall_names]

        filtered = [a for a in filtered
                    if a["label"] == FALL_CLASS_IDX or
                    a["frame_dir"] in kept_nonfall_names]
        print(f"  采样后负样本数: {len(filtered) - n_fall}")

    # ============ Step 5: 论文消融用的总下采样 ============
    if args.subsample_ratio < 1.0:
        print(f"\n[Step 5] 总体下采样到 {args.subsample_ratio*100:.0f}%(论文消融用)")
        # 在 train 里下采样,val 保持完整(否则评估不准)
        train_sp_name = "xsub_train"
        train_names = new_split[train_sp_name]
        train_anns = [a for a in filtered if a["frame_dir"] in train_names]

        # 分别按类别下采样(保持类别比例)
        train_pos = [a for a in train_anns if a["label"] == FALL_CLASS_IDX]
        train_neg = [a for a in train_anns if a["label"] != FALL_CLASS_IDX]
        keep_pos = random.sample(train_pos, max(1, int(len(train_pos) * args.subsample_ratio)))
        keep_neg = random.sample(train_neg, max(1, int(len(train_neg) * args.subsample_ratio)))
        keep_train_names = {a["frame_dir"] for a in (keep_pos + keep_neg)}

        val_names_set = set(new_split["xsub_val"])
        filtered = [a for a in filtered
                    if a["frame_dir"] in keep_train_names or
                    a["frame_dir"] in val_names_set]
        new_split[train_sp_name] = [n for n in train_names if n in keep_train_names]

        print(f"  下采样后 train: {len(new_split['xsub_train'])}")
        print(f"  val 保持: {len(new_split['xsub_val'])}")

    # ============ Step 6: 重新打标签为 0/1 ============
    print("\n[Step 6] 重新打标签为 0/1 (1=摔倒, 0=非摔倒)")
    filtered = relabel_to_binary(filtered)

    # 统计最终分布
    label_counter = Counter([a["label"] for a in filtered])
    print(f"  最终标签分布: {dict(label_counter)}")
    print(f"  正类比例: {label_counter[1] / sum(label_counter.values()):.2%}")

    for sp_name, names in new_split.items():
        name_set = set(names)
        sp_pos = sum(1 for a in filtered
                     if a["frame_dir"] in name_set and a["label"] == 1)
        sp_neg = sum(1 for a in filtered
                     if a["frame_dir"] in name_set and a["label"] == 0)
        print(f"  {sp_name}: {sp_pos} 摔倒 + {sp_neg} 非摔倒 = {sp_pos+sp_neg}")

    # ============ Step 7: 数据泄漏检查 ============
    print("\n[Step 7] X-Sub 划分泄漏自检")
    train_set = set(new_split.get("xsub_train", []))
    val_set = set(new_split.get("xsub_val", []))
    leak = train_set & val_set
    assert len(leak) == 0, f"❌ 泄漏!{len(leak)} 个样本同时在 train 和 val:{list(leak)[:5]}"
    print(f"  ✓ 训练集 ({len(train_set)}) 和验证集 ({len(val_set)}) 无重叠样本")

    # X-Sub 受试者编号检查
    train_subjects = set()
    val_subjects = set()
    for a in filtered:
        fd = a["frame_dir"]
        sid = int(fd[8:12].lstrip("P"))  # SxxxCxxxPxxxRxxxAxxx 取 P 后面 3 位
        if fd in train_set:
            train_subjects.add(sid)
        elif fd in val_set:
            val_subjects.add(sid)
    overlap_subjects = train_subjects & val_subjects
    if overlap_subjects:
        print(f"  ⚠ 警告:有 {len(overlap_subjects)} 个受试者同时在 train/val,"
              f"这是 NTU X-Sub 默认行为(不应发生),请检查源 pickle")
    else:
        print(f"  ✓ {len(train_subjects)} 个训练受试者与 {len(val_subjects)} 个验证受试者完全分离")

    # ============ Step 8: 保存 ============
    out = {
        "split": new_split,
        "annotations": filtered,
    }
    with open(dst, "wb") as f:
        pickle.dump(out, f)

    print("\n" + "=" * 70)
    print(f"✓ 保存到 {dst}")
    print(f"  正样本: {label_counter[1]}, 负样本: {label_counter[0]}")
    print(f"  下一步: python data_prep/visualize_skeleton.py --src {dst}")
    print("=" * 70)


if __name__ == "__main__":
    main()

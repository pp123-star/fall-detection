"""
tools/verify_best_ckpt.py — 验证 checkpoint 保存逻辑正确

防止上一版项目踩过的"checkpoint 保存逻辑 bug 导致 10 小时训练白跑"。

本脚本扫描 work_dir,检查:
1. 是否存在 best_*.pth(用于评估的最佳模型)
2. best 是否对应于训练日志里的最高 val_acc
3. 是否有意外丢失的 checkpoint(例如只剩 latest 没有 best)

用法:
    python tools/verify_best_ckpt.py work_dirs/posec3d_fall_binary
"""
import argparse
import json
import re
from pathlib import Path


def find_log_file(work_dir):
    """找最新一次训练的 vis_data/scalars.json。"""
    candidates = sorted(work_dir.glob("*/vis_data/scalars.json"), reverse=True)
    if not candidates:
        # 旧版日志格式
        candidates = sorted(work_dir.glob("*.log.json"), reverse=True)
    return candidates[0] if candidates else None


def parse_val_acc_from_scalars(log_file):
    """从 mmaction2 v1.x 的 scalars.json 提取每个 epoch 的 val acc/top1。"""
    epoch_acc = {}
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            # 验证日志记录里同时有 step (epoch) 和 acc/top1
            if "acc/top1" in d and "step" in d:
                epoch_acc[int(d["step"])] = float(d["acc/top1"])
    return epoch_acc


def parse_val_acc_from_legacy_log(log_file):
    """旧版 log.json (mmaction 0.x)。"""
    epoch_acc = {}
    with open(log_file, "r") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
            except Exception:
                continue
            if d.get("mode") == "val" and "top1_acc" in d:
                ep = d.get("epoch", -1)
                epoch_acc[ep] = float(d["top1_acc"])
    return epoch_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", help="训练输出目录 work_dirs/xxx")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        print(f"❌ 目录不存在: {work_dir}")
        return 1

    print(f"=" * 60)
    print(f"扫描 {work_dir}")
    print(f"=" * 60)

    # ============ 1. 列出所有 checkpoint ============
    pths = sorted(work_dir.glob("*.pth"))
    print(f"\n[1/4] 找到 {len(pths)} 个 checkpoint:")
    for p in pths:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"    {p.name}  ({size_mb:.1f} MB)")

    best_ckpts = [p for p in pths if p.name.startswith("best_")]
    epoch_ckpts = [p for p in pths if re.match(r"epoch_\d+\.pth", p.name)]
    latest_ckpts = [p for p in pths if p.name == "last_checkpoint" or p.name.endswith("last.pth")]

    print(f"\n  - best_*.pth (按指标筛的): {len(best_ckpts)}")
    print(f"  - epoch_*.pth (每个 epoch 的): {len(epoch_ckpts)}")
    print(f"  - latest_*.pth (最新一次的): {len(latest_ckpts)}")

    if not best_ckpts:
        print("\n❌ 没找到 best_*.pth!")
        print("   说明 CheckpointHook 的 save_best 没生效,请检查 config:")
        print("   default_hooks.checkpoint = dict(type='CheckpointHook',")
        print("                                  save_best='acc/top1', rule='greater')")
        return 1

    # ============ 2. 从日志中提取 val_acc 历史 ============
    log_file = find_log_file(work_dir)
    if log_file is None:
        print("\n⚠ 未找到训练日志(vis_data/scalars.json),跳过日志一致性检查")
        return 0

    print(f"\n[2/4] 训练日志: {log_file}")
    if log_file.name == "scalars.json":
        epoch_acc = parse_val_acc_from_scalars(log_file)
    else:
        epoch_acc = parse_val_acc_from_legacy_log(log_file)

    if not epoch_acc:
        print("⚠ 日志里没找到 val_acc 记录,可能训练还没跑完一个 epoch")
        return 0

    print(f"    跑了 {len(epoch_acc)} 次验证:")
    sorted_epochs = sorted(epoch_acc.items())
    for ep, acc in sorted_epochs[-5:]:  # 最后 5 个
        print(f"      epoch {ep}: val acc = {acc:.4f}")

    # ============ 3. 验证 best 对应历史最高 ============
    print(f"\n[3/4] best checkpoint 一致性检查")
    history_best_ep, history_best_acc = max(epoch_acc.items(), key=lambda x: x[1])
    print(f"    日志中最高 val acc: {history_best_acc:.4f} (epoch {history_best_ep})")

    for ckpt in best_ckpts:
        # 文件名通常是 best_acc_top1_epoch_18.pth
        m = re.search(r"epoch_(\d+)", ckpt.name)
        if m:
            ckpt_ep = int(m.group(1))
            ckpt_acc = epoch_acc.get(ckpt_ep, None)
            if ckpt_acc is not None:
                if abs(ckpt_acc - history_best_acc) < 1e-6:
                    print(f"    ✓ {ckpt.name} 来自 epoch {ckpt_ep},匹配历史最高 ({ckpt_acc:.4f})")
                else:
                    print(f"    ⚠ {ckpt.name} 来自 epoch {ckpt_ep},acc={ckpt_acc:.4f},"
                          f"但历史最高在 epoch {history_best_ep} ({history_best_acc:.4f})")
            else:
                print(f"    ? {ckpt.name} 来自 epoch {ckpt_ep},日志里没这个 epoch 记录")

    # ============ 4. 汇总 ============
    print(f"\n[4/4] 推荐用于评估的 checkpoint:")
    if best_ckpts:
        best_ckpt = best_ckpts[-1]  # 取最新一个 best
        size_mb = best_ckpt.stat().st_size / (1024 * 1024)
        print(f"    ★ {best_ckpt}")
        print(f"      ({size_mb:.1f} MB)")
        print()
        print(f"    评估命令:")
        config_files = list(work_dir.glob("*.py"))
        if config_files:
            print(f"      bash tools/test.sh {config_files[0]} {best_ckpt}")
        else:
            print(f"      bash tools/test.sh <your_config> {best_ckpt}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

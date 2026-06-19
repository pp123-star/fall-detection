"""
data_prep/download_pkl.py — 下载预提取好的 NTU 2D 骨骼 pickle

OpenMMLab 已经用 HRNet 在 NTU 视频上提取了 2D 骨骼数据并发布为 pickle,
我们直接下载,跳过自己跑姿态估计的步骤(原本要花数小时跑 56880 个视频)。

下载文件:
- ntu60_2d.pkl  (NTU RGB+D 60,~900 MB)
- ntu120_2d.pkl (NTU RGB+D 120,~1.8 GB)   [可选]

文件结构:
{
    'split': {
        'xsub_train': ['SxxxCxxxPxxxRxxxAxxx', ...],
        'xsub_val':   [...],
        'xview_train': [...],
        'xview_val':   [...],
    },
    'annotations': [
        {
            'frame_dir':       样本名,
            'label':           动作类别 ID (0-59),
            'img_shape':       (H, W),
            'original_shape':  (H, W),
            'total_frames':    帧数,
            'keypoint':        np.ndarray (M, T, 17, 2),
            'keypoint_score':  np.ndarray (M, T, 17),
        },
        ...
    ]
}

用法:
    python data_prep/download_pkl.py
    python data_prep/download_pkl.py --no-ntu120   # 不下载 NTU120
    python data_prep/download_pkl.py --data-dir ./mydata
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path


URLS = {
    "ntu60_2d.pkl":  "https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl",
    "ntu120_2d.pkl": "https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_2d.pkl",
}


def download_with_aria2(url, output_path):
    """用 aria2c 多线程下载,失败回退到 wget。"""
    output_path = Path(output_path)
    output_dir = output_path.parent
    filename = output_path.name

    # 优先 aria2
    if subprocess.call(["which", "aria2c"], stdout=subprocess.DEVNULL) == 0:
        print(f"[INFO] 使用 aria2c 多线程下载 {filename}")
        cmd = ["aria2c", "-x", "8", "-s", "8", "-c",
               "-d", str(output_dir), "-o", filename, url]
    elif subprocess.call(["which", "wget"], stdout=subprocess.DEVNULL) == 0:
        print(f"[INFO] 使用 wget 下载 {filename}")
        cmd = ["wget", "-c", "-O", str(output_path), url]
    else:
        print(f"[INFO] 使用 Python urllib 下载 {filename}(可能较慢)")
        import urllib.request
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 / total
                        print(f"\r  {downloaded/(1024**2):.1f} / {total/(1024**2):.1f} MB ({pct:.1f}%)",
                              end="", flush=True)
        print()
        return True

    result = subprocess.call(cmd)
    return result == 0


def verify_pickle(path):
    """简单 sanity check:能否反序列化、结构是否符合预期。"""
    import pickle
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert "split" in data, "pickle 缺少 split 字段"
        assert "annotations" in data, "pickle 缺少 annotations 字段"
        assert len(data["annotations"]) > 0, "annotations 为空"

        sample = data["annotations"][0]
        for key in ["frame_dir", "label", "keypoint", "total_frames"]:
            assert key in sample, f"样本缺少字段 {key}"

        kpt = sample["keypoint"]
        assert kpt.ndim == 4, f"keypoint 应为 4 维 (M,T,V,C),实际 {kpt.ndim} 维"
        assert kpt.shape[2] == 17, f"应为 17 个关键点,实际 {kpt.shape[2]} 个"
        assert kpt.shape[3] == 2, f"应为 2 维坐标 (x,y),实际 {kpt.shape[3]} 维"

        n_total = len(data["annotations"])
        labels = [a["label"] for a in data["annotations"]]
        n_classes = len(set(labels))
        print(f"  ✓ {path.name}: {n_total} 个样本,{n_classes} 个类别")
        print(f"    split: {list(data['split'].keys())}")
        print(f"    第一个样本: frame_dir={sample['frame_dir']}, "
              f"label={sample['label']}, keypoint shape={kpt.shape}")
        return True
    except Exception as e:
        print(f"  ✗ {path.name}: 校验失败 - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 NTU RGB+D 2D 骨骼 pickle")
    parser.add_argument("--data-dir", default="data",
                        help="保存目录(默认 ./data)")
    parser.add_argument("--no-ntu120", action="store_true",
                        help="只下载 NTU60,不下载 NTU120")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="已存在则跳过(默认 True)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = ["ntu60_2d.pkl"]
    if not args.no_ntu120:
        targets.append("ntu120_2d.pkl")

    print("=" * 60)
    print(f"目标目录: {data_dir.resolve()}")
    print(f"目标文件: {targets}")
    print("=" * 60)

    for fname in targets:
        out = data_dir / fname
        if out.exists() and args.skip_existing:
            print(f"[SKIP] {out} 已存在(若需重下,删除该文件)")
        else:
            ok = download_with_aria2(URLS[fname], out)
            if not ok:
                print(f"[ERROR] 下载 {fname} 失败")
                sys.exit(1)

        if not verify_pickle(out):
            print(f"[ERROR] {out} 校验失败,文件可能损坏,请重下")
            sys.exit(1)

    print("=" * 60)
    print("✓ 全部下载并校验完成!下一步:")
    print(f"    python data_prep/build_binary_pkl.py --src {data_dir}/ntu60_2d.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()

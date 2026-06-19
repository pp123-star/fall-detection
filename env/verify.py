"""
env/verify.py — 整体环境验证脚本

确保所有依赖正确装好。
"""
import sys


def main():
    print("=" * 60)
    print("环境验证")
    print("=" * 60)

    errors = []

    # PyTorch
    try:
        import torch
        print(f"PyTorch:     {torch.__version__}")
        print(f"CUDA OK:     {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU:         {torch.cuda.get_device_name(0)}")
            print(f"CUDA Ver:    {torch.version.cuda}")
        else:
            errors.append("CUDA 不可用,请重新装带 cu118 标记的 torch")
    except ImportError as e:
        errors.append(f"PyTorch 未装: {e}")

    # OpenMMLab
    for pkg_name in ["mmengine", "mmcv", "mmaction", "mmdet", "mmpose"]:
        try:
            pkg = __import__(pkg_name)
            print(f"{pkg_name:12s} {pkg.__version__}")
        except ImportError as e:
            errors.append(f"{pkg_name} 未装: {e}")

    # ultralytics
    try:
        import ultralytics
        print(f"ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        errors.append(f"ultralytics 未装: {e}")

    # 辅助包
    for pkg_name in ["numpy", "sklearn", "seaborn", "matplotlib",
                     "tqdm", "cv2", "decord", "pandas"]:
        try:
            pkg = __import__(pkg_name)
            ver = getattr(pkg, "__version__", "(no version)")
            print(f"{pkg_name:12s} {ver}")
        except ImportError as e:
            errors.append(f"{pkg_name} 未装: {e}")

    # mmaction Registry 简单测试
    try:
        from mmaction.registry import MODELS
        print(f"MMAction MODELS registry size: {len(MODELS.module_dict)}")
        if len(MODELS.module_dict) == 0:
            errors.append("MMAction MODELS registry 为空,可能 import 路径有问题")
    except Exception as e:
        errors.append(f"MMAction registry 读取失败: {e}")

    print("=" * 60)
    if errors:
        print("✗ 有以下问题:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✓ 全部 OK!可以开始下一步:数据准备")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""Split quan_lp GT/ into train/sharp and test/sharp as relative symlinks."""
import os
import random
from pathlib import Path

SRC = Path("/mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/GT")
DST_TRAIN = Path("/mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/train/sharp")
DST_TEST = Path("/mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/test/sharp")
SEED = 42
TRAIN_RATIO = 0.9


def symlink_split(files: list[Path], dst_dir: Path) -> None:
    for src_path in files:
        target = os.path.relpath(src_path, dst_dir)
        link = dst_dir / src_path.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)


def main() -> None:
    DST_TRAIN.mkdir(parents=True, exist_ok=True)
    DST_TEST.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC.glob("*.jpg"))
    random.seed(SEED)
    random.shuffle(files)
    split_idx = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    symlink_split(train_files, DST_TRAIN)
    symlink_split(test_files, DST_TEST)
    print(f"train={len(train_files)}, test={len(test_files)}")


if __name__ == "__main__":
    main()

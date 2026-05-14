"""
Unified plate_info.txt generation. Combines resume and text-based generation
with charset switching and unread plate tracking.
"""

import os
import sys

TMPDIR = os.environ.get("TMPDIR", "/tmp")
os.environ["HOME"] = TMPDIR

from paddleocr import PaddleOCR

CHARSETS = {
    "VN": "#0123456789ABCDEFGHJKLMNPQRSTUVWXYZ",
    "CN": "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品",
}


def filter_chars(text, charset):
    """Filter text to only chars in charset. VN uppercases; CN passes through Chinese."""
    if charset == "VN":
        return "".join(c for c in text.upper() if c in CHARSETS["VN"])
    else:
        result = []
        for c in text:
            if c in CHARSETS["CN"]:
                result.append(c)
        return "".join(result)


def text_to_indices(text, char_set, max_len=21):
    """Convert plate text to fixed-length index sequence."""
    indices = []
    for char in text:
        if char in char_set:
            indices.append(char_set.index(char))
        else:
            indices.append(0)
    while len(indices) < max_len:
        indices.append(0)
    return indices[:max_len]


def process_split(split, dataroot, charset="VN", use_gpu=True, gpu_id=0, resume=True):
    sharp_dir = os.path.join(dataroot, split, "sharp")
    plate_info_path = os.path.join(dataroot, split, "plate_info.txt")
    plate_info_text_path = os.path.join(dataroot, split, "plate_info_text.txt")
    unread_path = os.path.join(dataroot, split, "plate_info_unread.txt")

    char_set = CHARSETS[charset]

    images = sorted(os.listdir(sharp_dir))
    total = len(images)
    print(f"Total images in {split}: {total}")
    print(f"Charset: {charset} ({len(char_set)} chars)")

    processed = set()
    if resume and os.path.exists(plate_info_path):
        with open(plate_info_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    processed.add(parts[0])
        print(f"Already processed: {len(processed)}")

    remaining = [img for img in images if img not in processed]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    ocr = PaddleOCR(lang="ch", use_textline_orientation=True, use_gpu=use_gpu, gpu_id=gpu_id)

    plate_info_f = open(plate_info_path, "a" if resume else "w")
    plate_text_f = open(plate_info_text_path, "a" if resume else "w")
    unread_f = open(unread_path, "a" if resume else "w")

    for i, img_name in enumerate(remaining):
        if i > 0 and i % 500 == 0:
            print(f"  {i}/{len(remaining)}")
        img_path = os.path.join(sharp_dir, img_name)
        full_text = "?"
        try:
            ocr_result = ocr.ocr(img_path)
            if ocr_result and len(ocr_result[0]) > 0:
                texts = [item[1][0] for item in ocr_result[0]]
                raw_text = "".join(texts)
                filtered = filter_chars(raw_text, charset)
                full_text = filtered if filtered else "?"
            else:
                full_text = "?"
        except Exception:
            full_text = "?"

        if full_text == "?":
            unread_f.write(img_path + "\n")
            unread_f.flush()

        indices = text_to_indices(full_text, char_set)
        plate_info_f.write(img_name + " " + " ".join(str(idx) for idx in indices) + "\n")
        plate_text_f.write(f"{img_name}, {full_text}\n")

    plate_info_f.flush()
    plate_text_f.flush()
    plate_info_f.close()
    plate_text_f.close()
    unread_f.close()

    final_count = sum(1 for _ in open(plate_info_path))
    print(f"Done. Total entries: {final_count}/{total}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate plate_info.txt (unified)")
    parser.add_argument("dataroot", help="Path to dataset root (contains train/{sharp,blur})")
    parser.add_argument("--charset", default="VN", choices=["VN", "CN"], help="Charset: VN or CN (default: VN)")
    parser.add_argument("--use-gpu", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="Disable resume mode")
    args = parser.parse_args()

    for split in ["train", "test"]:
        print(f"\n=== Processing {split} ===")
        process_split(
            split, args.dataroot,
            charset=args.charset,
            use_gpu=args.use_gpu,
            gpu_id=args.gpu_id,
            resume=not args.no_resume,
        )

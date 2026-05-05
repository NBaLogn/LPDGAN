"""
Generate plate_info.txt using PaddleOCR for LPBlur dataset (Western plates).

Format per issue #2: <image_name> <21 int indices>
Char set (33 classes):
  Index 0: #  (padding/unknown)
  Index 1-10: 0-9 (digits)
  Index 11-30: A-Z excluding I,O (20 letters)
  Index 31: . (full-stop)
  Index 32: - (hyphen)

Square plates (2 rows) are flattened - row1 chars first, then row2 chars.
"""

import os
import sys

TMPDIR = os.environ.get("TMPDIR", "/tmp")
os.environ["HOME"] = TMPDIR

from paddleocr import PaddleOCR
import cv2
import numpy as np

# Western plate character set (33 classes)
# # (padding), 0-9 (10 digits), 20 letters (A-Z excl I,J,O,Q,R,W), . (full-stop), - (hyphen)
CHAR_SET = "#0123456789ABCDEFGHKLMNPSTUVXYZ.-"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_SET)}


def text_to_indices(text, max_len=21):
    """Convert plate text to list of 21 indices."""
    indices = []
    for char in text.upper():
        if char in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[char])
        else:
            indices.append(CHAR_TO_IDX["#"])  # unknown char -> padding
    # Pad to max_len
    while len(indices) < max_len:
        indices.append(CHAR_TO_IDX["#"])
    return indices[:max_len]


def process_split(split, dataroot):
    sharp_dir = os.path.join(dataroot, split, "sharp")
    plate_info_path = os.path.join(dataroot, split, "plate_info.txt")
    backup_path = plate_info_path + ".bak"

    if os.path.exists(plate_info_path):
        os.rename(plate_info_path, backup_path)

    images = sorted(os.listdir(sharp_dir))
    print(f"Processing {len(images)} images in {split}...")

    ocr = PaddleOCR(lang="ch", use_textline_orientation=True)
    results = []
    failed = 0

    for i, img_name in enumerate(images):
        img_path = os.path.join(sharp_dir, img_name)

        try:
            ocr_result = ocr.ocr(img_path)
            if ocr_result and ocr_result[0].get("rec_texts"):
                texts = ocr_result[0].get("rec_texts", [])
                # Concatenate all detected text (handles multi-line)
                full_text = "".join(texts)
                indices = text_to_indices(full_text)
                print(f"  {img_name}: {full_text} -> {indices[:8]}...")
            else:
                print(f"  Warning: no text detected for {img_name}")
                indices = [CHAR_TO_IDX["#"]] * 21
                failed += 1
        except Exception as e:
            print(f"  Error processing {img_name}: {e}")
            indices = [CHAR_TO_IDX["#"]] * 21
            failed += 1

        line = img_name + " " + " ".join(str(idx) for idx in indices) + "\n"
        results.append(line)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(images)}")

    with open(plate_info_path, "w") as f:
        f.writelines(results)

    print(f"Done. Wrote {len(results)} entries to {plate_info_path}")
    print(f"Failed/no detection: {failed}")


if __name__ == "__main__":
    dataroot = "dataset/quan_lp_dataset"
    for split in ["train", "val", "test"]:
        print(f"\n=== Processing {split} ===")
        process_split(split, dataroot)

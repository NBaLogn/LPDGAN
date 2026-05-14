"""
Generate plate_info.txt + plate_info_text.txt using PaddleOCR for LPBlur dataset (Western plates).

Format plate_info.txt: <image_name> <21 int indices>
Format plate_info_text.txt: <image_name> <raw_text>
Char set (33 classes):
  Index 0: #  (padding/unknown)
  Index 1-10: 0-9 (digits)
  Index 11-30: A-Z excluding I,O (20 letters)
  Index 31: . (full-stop)
  Index 32: - (hyphen)

Output files at dataset root: plate_info.txt, plate_info_text.txt
Each file has entries from train/val/test combined (sorted).

Square plates (2 rows) are flattened - row1 chars first, then row2 chars.
"""

import os

TMPDIR = os.environ.get("TMPDIR", "/tmp")
os.environ["HOME"] = TMPDIR

from paddleocr import PaddleOCR

# Western plate character set (33 classes)
# # (padding), 0-9 (10 digits), 21 letters (A-Z excl I,J,O,Q,W), . (full-stop), - (hyphen)
CHAR_SET = "#0123456789ABCDEFGHKLMNPRSTUVXYZ"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_SET)}


def text_to_indices(text, max_len=21):
    """Convert plate text to list of 21 indices."""
    indices = []
    for char in text.upper():
        if char in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[char])
        else:
            indices.append(CHAR_TO_IDX["#"])  # unknown char -> padding
    while len(indices) < max_len:
        indices.append(CHAR_TO_IDX["#"])
    return indices[:max_len]


def collect_all_images(dataroot):
    """Collect all images from train/val/test splits, grouped by split."""
    splits = {}
    for split in ["train", "val", "test"]:
        sharp_dir = os.path.join(dataroot, split, "sharp")
        if os.path.isdir(sharp_dir):
            splits[split] = sorted(os.listdir(sharp_dir))
    return splits


def process_all(dataroot):
    """Process all images, produce two output files at dataset root."""
    plate_info_path = os.path.join(dataroot, "plate_info.txt")
    plate_text_path = os.path.join(dataroot, "plate_info_text.txt")

    # Backup existing files
    for path in [plate_info_path, plate_text_path]:
        if os.path.exists(path):
            os.rename(path, path + ".bak")

    splits = collect_all_images(dataroot)
    total_images = sum(len(imgs) for imgs in splits.values())
    print(f"Total images: {total_images} (train={len(splits.get('train',[]))}, val={len(splits.get('val',[]))}, test={len(splits.get('test',[]))})")

    ocr = PaddleOCR(lang="ch", use_textline_orientation=True)

    idx_lines = []
    txt_lines = []
    failed = 0
    processed = 0

    for split_name, images in splits.items():
        sharp_dir = os.path.join(dataroot, split_name, "sharp")
        for img_name in images:
            img_path = os.path.join(sharp_dir, img_name)
            try:
                result = ocr.predict(img_path)
                if result and len(result) > 0:
                    rec_res = result[0]
                    texts = []
                    if hasattr(rec_res, 'rec_texts'):
                        texts = rec_res.rec_texts
                    elif isinstance(rec_res, dict):
                        texts = rec_res.get('rec_texts', [])
                    full_text = "".join(texts) if texts else ""
                    if not full_text:
                        failed += 1
                        full_text = "#" * 21
                else:
                    failed += 1
                    full_text = "#" * 21
            except Exception as e:
                print(f"  Error {img_name}: {e}")
                failed += 1
                full_text = "#" * 21

            indices = text_to_indices(full_text)
            idx_line = img_name + " " + " ".join(str(idx) for idx in indices)
            txt_line = img_name + " " + full_text
            idx_lines.append(idx_line)
            txt_lines.append(txt_line)
            processed += 1

            if processed % 500 == 0:
                print(f"  Processed {processed}/{total_images}")

    with open(plate_info_path, "w") as f:
        f.write("\n".join(idx_lines) + "\n")

    with open(plate_text_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")

    print(f"Done. Wrote {len(idx_lines)} entries to {plate_info_path}")
    print(f"  and {len(txt_lines)} entries to {plate_text_path}")
    print(f"Failed/no detection: {failed}")


if __name__ == "__main__":
    dataroot = "dataset/quan_lp_dataset"
    process_all(dataroot)

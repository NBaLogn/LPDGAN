"""
Generate plate_info_text.txt using PaddleOCR for LPBlur dataset (Western plates).
Writes human-readable text instead of encoded indices.
Output format: <image_name> <plate_text>
"""

import os
import sys

TMPDIR = os.environ.get("TMPDIR", "/tmp")
os.environ["HOME"] = TMPDIR

from paddleocr import PaddleOCR

CHAR_SET = "#0123456789ABCDEFGHKLMNPSTUVXYZ"
VALID_CHARS = set(CHAR_SET)


def filter_chars(text):
    """Drop any char not in VALID_CHARS. E.g. '35H-058.34' -> '35H05834'."""
    return "".join(c for c in text.upper() if c in VALID_CHARS)


def process_split(split, dataroot, chunk_size=500, use_gpu=True, gpu_id=0):
    sharp_dir = os.path.join(dataroot, split, "sharp")
    plate_info_path = os.path.join(dataroot, split, "plate_info_text.txt")
    suspicious_path = os.path.join(dataroot, split, "plate_info_suspicious.txt")
    backup_path = plate_info_path + ".bak"

    if os.path.exists(plate_info_path):
        os.rename(plate_info_path, backup_path)

    images = sorted(os.listdir(sharp_dir))
    total = len(images)
    print(f"Processing {total} images in {split} (chunk_size={chunk_size})...")

    ocr = PaddleOCR(lang="ch", use_textline_orientation=True, use_gpu=use_gpu, gpu_id=gpu_id)

    with open(plate_info_path, "w") as f, open(suspicious_path, "w") as sf:
        sf.write("image_name, raw_ocr_text\n")
        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_images = images[chunk_start:chunk_end]
            print(f"  Chunk {chunk_start}-{chunk_end}/{total}")
            for img_name in chunk_images:
                img_path = os.path.join(sharp_dir, img_name)
                try:
                    ocr_result = ocr.ocr(img_path)
                    if ocr_result and len(ocr_result[0]) > 0:
                        texts = [item[1][0] for item in ocr_result[0]]
                        raw_text = "".join(texts)
                        filtered = filter_chars(raw_text)
                        if filtered != raw_text.upper():
                            sf.write(f"{img_name}, {raw_text}\n")
                        full_text = filtered if filtered else "?"
                    else:
                        full_text = "?"
                except Exception:
                    full_text = "?"

                line = f"{img_name}, {full_text}\n"
                f.write(line)

            f.flush()

    print(f"Done. Wrote {total} entries to {plate_info_path}")


if __name__ == "__main__":
    dataroot = "dataset/quan_lp_dataset"
    use_gpu = sys.argv[1].lower() == "true" if len(sys.argv) > 1 else True
    gpu_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for split in ["train", "val", "test"]:
        print(f"\n=== Processing {split} ===")
        process_split(split, dataroot, use_gpu=use_gpu, gpu_id=gpu_id)

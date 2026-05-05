"""
Memory-efficient plate_info.txt generator using PaddleOCR.
Processes in chunks to avoid OOM. Writes incrementally.
"""

import os, sys

TMPDIR = os.environ.get("TMPDIR", "/tmp")
os.environ["HOME"] = TMPDIR

from paddleocr import PaddleOCR

CHAR_SET = "#0123456789ABCDEFGHKLMNPSTUVXYZ.-"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_SET)}


def text_to_indices(text, max_len=21):
    indices = []
    for char in text.upper():
        if char in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[char])
        else:
            indices.append(CHAR_TO_IDX["#"])
    while len(indices) < max_len:
        indices.append(CHAR_TO_IDX["#"])
    return indices[:max_len]


def process_split_chunked(split, dataroot, chunk_size=500):
    sharp_dir = os.path.join(dataroot, split, "sharp")
    plate_info_path = os.path.join(dataroot, split, "plate_info.txt")
    backup_path = plate_info_path + ".bak"

    if os.path.exists(plate_info_path):
        os.rename(plate_info_path, backup_path)

    images = sorted(os.listdir(sharp_dir))
    total = len(images)
    print(f"Processing {total} images in {split} (chunk_size={chunk_size})...")

    ocr = PaddleOCR(lang="ch", use_textline_orientation=True)

    with open(plate_info_path, "w") as f:
        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_images = images[chunk_start:chunk_end]
            print(f"  Chunk {chunk_start}-{chunk_end}/{total}")
            for i, img_name in enumerate(chunk_images):
                img_path = os.path.join(sharp_dir, img_name)
                try:
                    ocr_result = ocr.ocr(img_path)
                    if ocr_result and ocr_result[0].get("rec_texts"):
                        texts = ocr_result[0].get("rec_texts", [])
                        full_text = "".join(texts)
                        indices = text_to_indices(full_text)
                    else:
                        indices = [CHAR_TO_IDX["#"]] * 21
                except Exception as e:
                    indices = [CHAR_TO_IDX["#"]] * 21

                line = img_name + " " + " ".join(str(idx) for idx in indices) + "\n"
                f.write(line)

            f.flush()

    print(f"Done. Wrote {total} entries to {plate_info_path}")


if __name__ == "__main__":
    dataroot = "dataset/quan_lp_dataset"
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    process_split_chunked(split, dataroot, chunk_size)
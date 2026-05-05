"""
Resume plate_info.txt generation. Appends remaining images to existing file.
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


def process_split_resume(split, dataroot, chunk_size=500):
    sharp_dir = os.path.join(dataroot, split, "sharp")
    plate_info_path = os.path.join(dataroot, split, "plate_info.txt")

    images = sorted(os.listdir(sharp_dir))
    total = len(images)
    print(f"Total images in {split}: {total}")

    # Find already-processed images
    processed = set()
    if os.path.exists(plate_info_path):
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

    ocr = PaddleOCR(lang="ch", use_textline_orientation=True)

    # Append mode
    with open(plate_info_path, "a") as f:
        for chunk_start in range(0, len(remaining), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(remaining))
            chunk_images = remaining[chunk_start:chunk_end]
            print(f"  Chunk {chunk_start}-{chunk_end}/{len(remaining)}")
            for img_name in chunk_images:
                img_path = os.path.join(sharp_dir, img_name)
                try:
                    ocr_result = ocr.ocr(img_path)
                    if ocr_result and ocr_result[0].get("rec_texts"):
                        texts = ocr_result[0].get("rec_texts", [])
                        full_text = "".join(texts)
                        indices = text_to_indices(full_text)
                    else:
                        indices = [CHAR_TO_IDX["#"]] * 21
                except Exception:
                    indices = [CHAR_TO_IDX["#"]] * 21

                line = img_name + " " + " ".join(str(idx) for idx in indices) + "\n"
                f.write(line)
            f.flush()
            print(f"  Wrote {chunk_end} total entries so far")

    final_count = sum(1 for _ in open(plate_info_path))
    print(f"Done. Total entries: {final_count}/{total}")


if __name__ == "__main__":
    dataroot = "dataset/quan_lp_dataset"
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    process_split_resume(split, dataroot, chunk_size)
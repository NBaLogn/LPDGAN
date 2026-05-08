"""
Fix anomalous OCR plates from generate_plate_info_text.py output.
Passes through plate text and corrects common OCR errors.
"""

import os
import sys


def fix_plate(plate: str) -> tuple[str, list[tuple[int, str, str]]]:
    """
    Apply 5 correction passes to plate text.
    Returns (fixed_plate, list of (pass_num, old_char, new_char) corrections).
    """
    plate = plate.strip()
    corrections = []

    # Pass 1: Delete non-alphanumeric chars at index 0 and -1
    # Also explicitly strip · since isalnum() returns True for Unicode middle-dot
    STRIP_CHARS = set('·:°"\'`附^：$%*!@#$&()+=[]{}|<>?')

    while plate and (not plate[0].isalnum() or plate[0] in STRIP_CHARS):
        corrections.append((1, plate[0], "(deleted)"))
        plate = plate[1:]
    while plate and (not plate[-1].isalnum() or plate[-1] in STRIP_CHARS):
        corrections.append((1, plate[-1], "(deleted)"))
        plate = plate[:-1]

    if not plate:
        return plate, corrections

    # Pass 2: Index 0 cannot be '0' or 'O' -> delete if so
    if plate[0] in ('0', 'O'):
        corrections.append((2, plate[0], "(deleted)"))
        plate = plate[1:]

    if not plate:
        return plate, corrections

    # Pass 3: Dash can only be in index 2 or 3. Set index 2 to '-' if invalid.
    if len(plate) > 2 and plate[2] not in ('-', '.'):
        corrections.append((3, plate[2], "-"))
        plate = plate[:2] + '-' + plate[3:]

    # Pass 4: Index -3 non-alphanumeric -> '.', leave existing '.' or '-' untouched
    if len(plate) >= 3 and not plate[-3].isalnum() and plate[-3] not in ('.', '-'):
        corrections.append((4, plate[-3], "."))
        plate = plate[:-3] + '.' + plate[-2:]

    # Pass 5: Index 2 '0' or 'O' -> 'C'
    if len(plate) > 2 and plate[2] in ('0', 'O'):
        corrections.append((5, plate[2], "C"))
        plate = plate[:2] + 'C' + plate[3:]

    return plate, corrections


def process_split(split: str, dataroot: str):
    input_path = os.path.join(dataroot, split, "plate_info_text.txt")
    output_path = os.path.join(dataroot, split, "plate_info_text_fixed.txt")

    if not os.path.exists(input_path):
        print(f"  {split}: {input_path} not found, skipping")
        return

    total = 0
    corrected = 0
    log_entries = []

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line or ", " not in line:
                continue

            parts = line.split(", ", 1)
            if len(parts) != 2:
                continue

            img_name, plate_text = parts
            fixed, corrections = fix_plate(plate_text)

            total += 1
            if corrections:
                corrected += 1
                log_entries.append((img_name, plate_text, fixed, corrections))

            fout.write(f"{img_name}, {fixed}\n")

    log_path = os.path.join(dataroot, split, "plate_info_fix_log.txt")
    with open(log_path, "w") as flog:
        for img, original, fixed, corrections in log_entries:
            flog.write(f"{img}: {original!r} -> {fixed!r}\n")
            for p, old, new in corrections:
                flog.write(f"  Pass {p}: {old!r} -> {new!r}\n")

    print(f"  {split}: {corrected}/{total} corrected -> {output_path}")
    print(f"         {len(log_entries)} corrections logged -> {log_path}")


if __name__ == "__main__":
    dataroot = "/mnt/data/nblong-t04/LPDGAN/dataset/quan_lp"
    for split in ["train", "val", "test"]:
        print(f"\n=== Processing {split} ===")
        process_split(split, dataroot)

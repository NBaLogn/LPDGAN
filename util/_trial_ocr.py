"""Trial-and-error PaddleOCR config tester for 2-row VN plates.

Run:
    uvr util/_trial_ocr.py dataset/quan_lp/train/sharp/lp161.jpg
"""

import sys
import cv2

IMG = sys.argv[1] if len(sys.argv) > 1 else "dataset/quan_lp/train/sharp/lp161.jpg"

TRIALS = [
    ("v2-baseline",          dict(lang="en", use_angle_cls=True)),
    ("v2-low-thresh",        dict(lang="en", use_angle_cls=True, det_db_thresh=0.15, det_db_box_thresh=0.3)),
    ("v2-unclip-large",      dict(lang="en", use_angle_cls=True, det_db_thresh=0.2, det_db_box_thresh=0.3, det_db_unclip_ratio=3.0)),
    ("v2-limit-min-32",      dict(lang="en", use_angle_cls=True, det_limit_type="min", det_limit_side_len=32)),
    ("v2-limit-min-16",      dict(lang="en", use_angle_cls=True, det_db_thresh=0.15, det_db_box_thresh=0.3, det_limit_type="min", det_limit_side_len=16)),
    ("v2-all",               dict(lang="en", use_angle_cls=True, det_db_thresh=0.15, det_db_box_thresh=0.3, det_db_unclip_ratio=3.0, det_limit_type="min", det_limit_side_len=16)),
    ("v2-ppv4",              dict(lang="en", use_angle_cls=True, ocr_version="PP-OCRv4", det_db_thresh=0.2, det_db_box_thresh=0.3)),
    ("v2-split-overlap",     None),  # handled separately below
]


def run_ocr_plain(ocr, img):
    result = ocr.ocr(img, cls=True)
    return result[0] if (result and result[0]) else []


def run_split_with_overlap(ocr, img):
    """Split at mid with 20% overlap each side."""
    h = img.shape[0]
    mid = h // 2
    pad = max(4, h // 5)
    top = img[:mid + pad]
    bot = img[mid - pad:]
    offset = mid - pad

    dets = []
    for half, off in [(top, 0), (bot, offset)]:
        r = ocr.ocr(half, cls=True)
        if r and r[0]:
            for box, tc in r[0]:
                adjusted = [[p[0], p[1] + off] for p in box]
                dets.append((adjusted, tc))
    return dets


from paddleocr import PaddleOCR
import os, logging
logging.disable(logging.CRITICAL)
os.environ["GLOG_v"] = "0"
os.environ["GLOG_logtostderr"] = "0"

img_orig = cv2.imread(IMG)
h, w = img_orig.shape[:2]
img_up2 = cv2.resize(img_orig, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
img_up3 = cv2.resize(img_orig, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

print(f"Image: {IMG}  shape={img_orig.shape}")
print("=" * 60)

img_up4 = cv2.resize(img_orig, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
img_up3_lz = cv2.resize(img_orig, (w * 3, h * 3), interpolation=cv2.INTER_LANCZOS4)

UPSCALE_TRIALS = [
    ("up2x-baseline",       img_up2, dict(lang="en", use_angle_cls=True)),
    ("up2x-low-thresh",     img_up2, dict(lang="en", use_angle_cls=True, det_db_thresh=0.2, det_db_box_thresh=0.3)),
    ("up3x-baseline",       img_up3, dict(lang="en", use_angle_cls=True)),
    ("up3x-low-thresh",     img_up3, dict(lang="en", use_angle_cls=True, det_db_thresh=0.2, det_db_box_thresh=0.3)),
    ("up3x-unclip3",        img_up3, dict(lang="en", use_angle_cls=True, det_db_unclip_ratio=3.0)),
    ("up3x-unclip3-thresh", img_up3, dict(lang="en", use_angle_cls=True, det_db_thresh=0.2, det_db_box_thresh=0.3, det_db_unclip_ratio=3.0)),
    ("up3x-lanczos",        img_up3_lz, dict(lang="en", use_angle_cls=True)),
    ("up4x-baseline",       img_up4, dict(lang="en", use_angle_cls=True)),
    ("up4x-low-thresh",     img_up4, dict(lang="en", use_angle_cls=True, det_db_thresh=0.2, det_db_box_thresh=0.3)),
    ("up2x-split",          img_up2, None),  # split fallback
    ("up3x-split",          img_up3, None),
]

for name, img, kwargs in UPSCALE_TRIALS:
    try:
        if kwargs is None:
            ocr = PaddleOCR(lang="en", use_angle_cls=True, det_db_thresh=0.2, det_db_box_thresh=0.3)
            dets = run_split_with_overlap(ocr, img)
        else:
            ocr = PaddleOCR(**kwargs)
            dets = run_ocr_plain(ocr, img)
        texts = [(text, round(conf, 3)) for _, (text, conf) in dets]
        print(f"[{name}]  rows={len(texts)}  {texts}")
    except Exception as e:
        print(f"[{name}]  ERROR: {e}")

print("-" * 60)
img = img_orig
for name, kwargs in TRIALS:
    try:
        if name == "v2-split-overlap":
            ocr = PaddleOCR(lang="en", use_angle_cls=True, det_db_thresh=0.2, det_db_box_thresh=0.3)
            dets = run_split_with_overlap(ocr, img)
        else:
            ocr = PaddleOCR(**kwargs)
            dets = run_ocr_plain(ocr, img)

        texts = [(text, round(conf, 3)) for _, (text, conf) in dets]
        print(f"[{name}]  rows={len(texts)}  {texts}")
    except Exception as e:
        print(f"[{name}]  ERROR: {e}")

print("=" * 60)

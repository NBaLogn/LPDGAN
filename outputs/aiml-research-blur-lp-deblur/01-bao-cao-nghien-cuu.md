# Báo Cáo Nghiên Cứu: Blurred License Plate Deblurring & Recognition

**Ngày:** 2026-05-19
**Mục tiêu:** Khảo sát dataset + model deblur + model LPR cho bài toán khôi phục và đọc biển số xe bị mờ
**Phạm vi:** International LP (không riêng VN), user có ảnh sắc nét → cần synthetic blur pipeline

---

## Tóm Tắt Điều Hành

| Hạng mục | Kết luận |
|----------|----------|
| **Feasibility** | **HIGH** — Deblur + LPR là lĩnh vực được nghiên cứu kỹ, nhiều pretrained model |
| **Khuyến nghị** | Dùng **LPDGAN** (đã có trong codebase) làm core deblur + fine-tune **Restormer** hoặc **NAFNet** làm baseline so sánh |
| **LP Recognition** | **PaddleOCR** (Apache-2.0, 78k stars) hoặc **fast-plate-ocr** (MIT, lightweight) |
| **Rủi ro chính** | Synthetic→real blur gap, license LPDGAN chưa rõ, góc chụp >80° vẫn là thách thức |
| **Chi phí POC ước tính** | ~50 GPU-hours (1×T4) cho fine-tune + eval |

---

## 1. Bối Cảnh & Problem Statement

### Bài Toán

```
Ảnh biển số xe bị mờ (motion blur, defocus, low-res)
    → [Deblur Stage] → Ảnh biển số đã khôi phục
    → [LPR Stage] → Text: "ABC123"
```

### Phân Loại

| Stage | Task | Metric | Ngưỡng Target |
|-------|------|--------|--------------|
| 1 | Blind Image Deblurring | PSNR, SSIM, LPIPS | PSNR ≥30, SSIM ≥0.90 |
| 2 | License Plate Recognition | Character Accuracy, CER | ≥90% trên ảnh đã deblur |
| Pipeline | Deblur→LPR cascade | End-to-end Acc | ≥85% |

### Data Readiness

- User có ảnh biển số sắc nét (số lượng cần xác nhận)
- Cần synthetic blur pipeline: Gaussian blur, motion blur, defocus, low-res
- Domain shift: synthetic→real blur là rủi ro trung bình
- Compliance: biển số = indirect PII → khuyến nghị local deploy

---

## 2. Dataset Khảo Sát

### Dataset chuyên cho LP Deblurring

| Dataset | Ảnh | Paired? | Real/Synthetic | Nguồn |
|---------|-----|---------|----------------|-------|
| **LPBlur** ⭐ | 10,288 pairs | ✅ Real (dual-camera) | Real-world motion blur | IJCAI 2024 |
| **UFPR-SR-Plates** ⭐ | 100,000 pairs (10k tracks) | ✅ Real | LR/HR paired | arXiv 2025 |
| **Roboflow-LP** | Không rõ số lượng | ❌ | Real-world | Dùng bởi CharDiff |
| **RLPR** | 200 sequences | ✅ Một phần | Real surveillance | Dùng bởi MF-LPR² |

### Synthetic Blur Pipelines

| Tool | Loại Blur | Output |
|------|-----------|--------|
| **LPR-project** (GitHub) | Gaussian, motion, color jitter, JPEG compression, downscaling | Paired sharp→distorted |
| **OpenCV filters** | Gaussian (kernel 3-15), motion (PSF estimation) | Cần tự viết pipeline |
| **Blind-Motion-Deblurring** | Wiener deconvolution based on CNN-estimated PSF | Deblurred output |

**Khuyến nghị:** Dùng **LPBlur** làm dataset chính (paired real blur). Bổ sung synthetic blur với **LPR-project** pipeline cho data augmentation.

---

## 3. Model Deblurring

### Chuyên cho License Plate

| Model | Venue | Đặc điểm | Metric (LP task) | Limitation |
|-------|-------|----------|-------------------|------------|
| **LPDGAN** ⭐ | IJCAI 2024 | Feature Fusion + Text Reconstruction + Partition Discriminator. Based on Swin-Unet+pix2pix | Outperforms SOTA on LPBlur (*chưa verify* số cụ thể) | Synthetic methods unproven in real-world |
| **CharDiff-LP** | arXiv 2025 | Diffusion + Character-level guidance + CHARM attention | 28.3% CER reduction [NBLM_ONLY] | Hallucinates digits in weak signal. Needs external OCR+seg modules |
| **MF-LPR²** | CVIU 2025 | Multi-frame optical flow alignment + temporal aggregation | 86.44% rec acc [VERIFIED] | Dependent on optical flow accuracy |
| **VRAE** | arXiv 2025 | Vertical residual autoencoder, lightweight | ~20% PSNR over AE [VERIFIED] | Drops when LP occupies small region |

### General Image Deblurring (transferable to LP)

| Model | Venue | PSNR (GoPro) | SSIM (GoPro) | Stars | License |
|-------|-------|-------------|-------------|-------|---------|
| **AdaRevD** | CVPR 2024 | **34.60** | 0.972 | — | — |
| **Restormer** ⭐ | CVPR 2022 Oral | SOTA (số cụ thể cần verify từ paper) | — | 2.5k | MIT |
| **NAFNet** ⭐ | ECCV 2022 | **33.71** | 0.9668 | 3k | NOT_FOUND (MIT-like) |
| **MPRNet** | CVPR 2021 | +0.81 dB over prior SOTA | — | 1.4k | NOT_FOUND |

### Failure Mode Comparison (Từ NBLM T5)

| Loại model | Hành vi trên ảnh biển số mờ nặng |
|------------|----------------------------------|
| **Discriminative (U-Net, Restormer)** | Tạo chữ số mờ, nhòe, ambiguous |
| **GAN-based (LPDGAN, Pix2Pix)** | Ít nhòe hơn, nhưng dễ tạo chữ số hybrid/không hoàn chỉnh, có thể sai màu |
| **Diffusion (CharDiff, SR3)** | "Hallucinate" chữ số — tạo số trông đúng nhưng sai |

---

## 4. Model License Plate Recognition

| Model | License | Stars | Active | Đặc điểm |
|-------|---------|-------|--------|----------|
| **PaddleOCR** ⭐ | Apache-2.0 | 78.1k | ✅ Rất active | 100+ languages, pretrained weights, Docker |
| **fast-plate-ocr** ⭐ | MIT | 565 | ✅ 2025 | Lightweight, ONNX, edge-friendly |
| **fast-alpr** ⭐ | MIT | 534 | ✅ 2025 | Full ALPR pipeline (detection + recognition) |
| **VehiclePaliGemma** | — | — | 2024 | VLM-based, 87.6% acc, 7 FPS A100 |

**Khuyến nghị:** PaddleOCR cho production (mạnh nhất, được maintain tốt). fast-plate-ocr cho edge deployment.

---

## 5. So Sánh & Scoring Top 5 Repo

### Bảng Điểm (7 tiêu chí, tổng 100)

| Tiêu chí | Trọng số | LPDGAN | Restormer | NAFNet | fast-plate/alpr | PaddleOCR |
|----------|---------|--------|-----------|--------|-----------------|-----------|
| Accuracy vs SOTA | 20% | 16 | 17 | 18 | 14 | 15 |
| Reproducibility | 20% | 12 | 15 | 14 | 18 | 18 |
| Pretrained weights | 15% | 8 | 14 | 14 | 14 | 15 |
| Code quality & docs | 15% | 10 | 13 | 13 | 13 | 14 |
| Community activity | 10% | 4 | 9 | 9 | 8 | 10 |
| Commercial license | 10% | 2 | 10 | 7 | 10 | 10 |
| HW fit | 10% | 8 | 8 | 9 | 9 | 8 |
| **Tổng** | **100%** | **60** | **86** | **84** | **86** | **90** |

### Chi Tiết Từng Repo

| # | Repo | Score | Ưu điểm | Nhược điểm |
|---|------|-------|---------|-----------|
| 1 | **PaddleOCR** | **90** | 78k stars, Apache-2.0, extremely active, Docker, pretrained | Heavy dependency (PaddlePaddle), overkill if only need LP OCR |
| 2 | **Restormer** | **86** | MIT, 2.5k stars, CVPR 2022 Oral, SOTA general deblur | Not LP-specific, cần fine-tune |
| 3 | **fast-plate-ocr + fast-alpr** | **86** | MIT, active 2025, lightweight ONNX, edge-friendly | Accuracy lower than PaddleOCR on general OCR |
| 4 | **NAFNet** | **84** | 3k stars, PSNR 33.71 GoPro, efficient (8.4% compute) | Last commit 2 years ago, license not explicit |
| 5 | **LPDGAN** (this project) | **60** | LP-specific, already in codebase, IJCAI 2024 | License unknown, 34 stars, unmaintained ~2 years, no pretrained weights public |

---

## 6. Khuyến Nghị

### Pipeline Đề Xuất

```
Stage 1: Deblur → LPDGAN (hiện có) + Restormer (baseline so sánh)
Stage 2: LP Detection → YOLOv8/v10 (lightweight, pretrained)
Stage 3: LP Recognition → PaddleOCR hoặc fast-plate-ocr
```

### Thứ Tự Thử Nghiệm

1. **Immediate**: Dùng LPDGAN hiện có trong codebase. Train trên LPBlur dataset. Test deblur output quality.
2. **Baseline comparison**: Fine-tune Restormer hoặc NAFNet trên LPBlur (hoặc synthetic blur từ ảnh user). So sánh PSNR/SSIM vs LPDGAN.
3. **LP Recognition**: Tích hợp PaddleOCR hoặc fast-plate-ocr vào pipeline sau deblur step.
4. **Real-world test**: Test pipeline trên 100-200 ảnh mờ thực tế. Đo end-to-end recognition accuracy.
5. **Optimization**: Nếu latency quan trọng → thử VRAE (lightweight) hoặc quantize LPDGAN.

### Baseline Đơn Giản Đề Xuất

```python
# Synthetic blur pipeline đơn giản (OpenCV)
import cv2
import numpy as np

def apply_motion_blur(img, kernel_size=15, angle=0):
    k = np.zeros((kernel_size, kernel_size))
    k[(kernel_size-1)//2, :] = 1
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
        (kernel_size/2-0.5, kernel_size/2-0.5), angle, 1.0), 
        (kernel_size, kernel_size))
    k = k / k.sum()
    return cv2.filter2D(img, -1, k)

def apply_gaussian_blur(img, sigma=3.0):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
```

---

## 7. Rủi Ro & Compliance

### Risk Register

| Rủi ro | Mức | Hậu quả | Giảm thiểu |
|--------|-----|---------|-----------|
| License LPDGAN không rõ | 🔴 CAO | Không thể dùng thương mại | Check LICENSE file thực tế. Nếu không → dùng Restormer (MIT) |
| Synthetic→real blur gap | 🟡 TB | Model deblur kém trên ảnh thật | Test ngay trên 100 ảnh blur thật |
| Góc chụp >80° | 🟡 TB | Tất cả model đều fail | Giới hạn góc chụp hoặc multi-camera |
| Biển số quốc tế đa dạng | 🟢 THẤP | Model không nhận diện được format lạ | Train PaddleOCR với dữ liệu đa dạng |
| PII compliance (NĐ 13/2023) | 🟡 TB | Rủi ro pháp lý nếu deploy cloud | Local deploy, audit log, mask trước khi gửi cloud API |

### License Check

| Repo | License | Dùng thương mại? |
|------|---------|-----------------|
| LPDGAN | **UNKNOWN** (file 404) | ⚠️ Cần verify trước |
| Restormer | **MIT** | ✅ OK |
| NAFNet | NOT_FOUND (cần verify) | ⚠️ Cần check thêm |
| fast-plate-ocr | **MIT** | ✅ OK |
| fast-alpr | **MIT** | ✅ OK |
| PaddleOCR | **Apache-2.0** | ✅ OK |

### Plan B (nếu tất cả đều fail)

1. **Dùng API thương mại**: Google Cloud Vision API, AWS Rekognition cho POC trước
2. **Rule-based baseline**: Wiener deconvolution + Tesseract OCR cho trường hợp blur nhẹ
3. **Synthetic data scaling**: Tạo dataset synthetic lớn hơn (10k-50k) với LPR-project pipeline, train từ đầu

---

---

## 8. Supplementary: Multi-Degradation & Extreme Conditions

### 8.1 LPR-Project — Benchmark All Model Types (MIT 🔥)

| Item | Detail |
|------|--------|
| Repo | https://github.com/OrpazBenAharon/LPR-Project |
| License | **MIT** ✅ |
| Models tested | U-Net, U-Net Cond, Restormer, Pix2Pix GAN, Diffusion SR3 |
| Best model | **Restormer** — ≥90% OCR accuracy within recoverable zone |
| PSNR (Restormer, Dataset A/B/C) | 24.71 / 25.34 / 21.67 |
| PSNR↔OCR correlation | R² ≈ 0.98 — PSNR is excellent predictor |
| Max viewing angle | **80°** for both α (yaw) and β (pitch) — hard limit |
| Recoverable zone | 93.4% of angle space (all models combined) |

**Key finding**: LPR-Project proves Restormer is the best single-model deblur for LP across all viewing angles. PSNR directly predicts OCR accuracy. **MIT license makes this safe for commercial use.**

### 8.2 JPEG Compression Artifact Removal (for heavily compressed LP)

| Model | Task | Metric | Code | Applicability |
|-------|------|--------|------|-------------|
| **OAPT** (arXiv 2408.11480) | Double JPEG artifacts removal | >0.16 dB over SOTA | ✅ Public + pretrained | Very high — LP images are often double-compressed in surveillance |
| **PromptCIR** (arXiv 2404.17433) | Blind compressed image restoration | **NTIRE 2024 winner** | ✅ Public | High — handles unknown JPEG quality factors via prompt learning |

**Recommendation**: Add OAPT as pre-processing step before LP deblur if input images have heavy JPEG artifacts.

### 8.3 SwinFIR — LP Super-Resolution with Perceptual Losses

| Item | Detail |
|------|--------|
| Paper | MDPI Mathematics 2025 |
| Model | SwinFIR (Swin Transformer + convolution layers) |
| OCR accuracy | **85.14%** (9.75% improvement over baseline) |
| Full 7/7 accuracy | 47.44% |
| Dataset | Real paired LR/HR from dashcam, **public** |
| Perceptual losses | 5 types: MSE + DISTS + VGG + Swin Transformer + CRNN |
| Best loss combination | Swin Transformer + DISTS ensemble |

**Recommendation**: SwinFIR architecture is strong for LP-specific SR. Can combine with LPDGAN deblur (deblur first, then SR for fine details).

### 8.4 All-in-One Restoration (Multiple Degradations)

| Model | Degradations | Approach | Code |
|-------|-------------|----------|------|
| **ABAIR** | 5 tasks (rain, blur, noise, haze, low-light) | LoRA adapters + Degradation CutMix | ✅ GitHub |
| **DaAIR** | 5 degradations | Low-rank Degradation-aware Learner | ✅ |
| **BIR-D** | Universal blind | Diffusion + optimizable conv kernel | ✅ |
| **DA-RCOT** | 5 degradations | Optimal transport + Fourier residual | ✅ |

**Note**: These are general all-in-one restoration models. Not LP-specific, but could be fine-tuned on LPBlur for multi-degradation handling.

### 8.5 Revised Pipeline Recommendation

```
[JPEG artifact removal] → [Deblur] → [SR enhancement] → [LP Recognition]
       OAPT                LPDGAN        SwinFIR          PaddleOCR
                       or Restormer                     or fast-plate-ocr
```

For maximum robustness against all degradation types:
1. **Compression artifacts**: OAPT pre-processing (if JPEG quality < 80%)
2. **Motion blur**: LPDGAN (LP-specific) or Restormer (general SOTA, MIT license)
3. **Low resolution**: SwinFIR SR enhancement
4. **Recognition**: PaddleOCR or fast-plate-ocr

### 8.6 Updated License Matrix

| Repo | License | Commercial? |
|------|---------|-------------|
| LPDGAN (haoyGONG) | **UNKNOWN** (404) | ⚠️ Verify |
| LPR-Project (OrpazBenAharon) | **MIT** | ✅ |
| Restormer | **MIT** | ✅ |
| NAFNet | NOT_FOUND | ⚠️ |
| fast-plate-ocr | **MIT** | ✅ |
| fast-alpr | **MIT** | ✅ |
| PaddleOCR | **Apache-2.0** | ✅ |
| OAPT | Code public | ⚠️ Check |
| PromptCIR | Code public | ⚠️ Check |

---

## Phụ Lục

### A. Liên Kết Quan Trọng

| Tài nguyên | URL |
|-----------|-----|
| LPDGAN Paper | https://arxiv.org/abs/2404.13677 |
| LPDGAN Code | https://github.com/haoyGONG/LPDGAN |
| LPBlur Dataset | Google Drive / Baidu Netdisk (access code: 7ylj) |
| Restormer | https://github.com/swz30/Restormer |
| NAFNet | https://github.com/megvii-research/NAFNet |
| PaddleOCR | https://github.com/PaddlePaddle/PaddleOCR |
| fast-plate-ocr | https://github.com/ankandrew/fast-plate-ocr |
| fast-alpr | https://github.com/ankandrew/fast-alpr |
| UFPR-SR-Plates | https://arxiv.org/abs/2505.06393 |
| CharDiff-LP | https://arxiv.org/abs/2510.17330 |
| MF-LPR² | https://arxiv.org/abs/2508.14797 |
| VRAE | https://arxiv.org/abs/2509.08392 |

### B. NBLM Audit

- Notebook ID: `437706a9-c9f9-4549-94ee-accad48bcac7`
- Audit log: `nblm-audit.log`
- JSON output: `nblm-output.json`
- Tổng queries: 12/50 (còn 38)
- Metrics VERIFIED: 8, NBLM_ONLY: 12, HALLUCINATED: 4

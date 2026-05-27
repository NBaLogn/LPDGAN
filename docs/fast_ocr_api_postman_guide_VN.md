# OCR Biển Số Xe — Hướng Dẫn Sử Dụng Postman

Cách gọi API OCR từ Postman. Không cần viết code.

## Trước khi bắt đầu

Hỏi người dựng server để lấy:

1. **Server URL** — ví dụ `http://10.0.0.5:8001` (thay `100.111.0.111:8001` ở các bước dưới nếu cần).
2. **Ảnh biển số mẫu** để thử. Mỗi ảnh phải là biển số đã được cắt sát (JPEG hoặc PNG, dưới 8 MB).

> Nếu ảnh chứa cả xe hoặc nhiều phần nền, kết quả sẽ kém. Cắt sát biển số trước khi gửi.

---

## Bước 1 — Kiểm tra server còn sống

**Method**: `GET`
**URL**: `http://100.111.0.111:8001/health`

Bấm **Send**.

Kết quả mong đợi:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "cct-s-v2-global-model"
}
```

Nếu báo connection error, nghĩa là server tắt hoặc URL sai. Dừng lại và hỏi admin.

---

## Bước 2 — Đọc một biển số (`/ocr`)

**Method**: `POST`
**URL**: `http://100.111.0.111:8001/ocr`

### Tab Body

1. Chọn **Body**.
2. Chọn **form-data**.
3. Thêm một dòng:
   - **Key**: `file`
   - Di chuột vào ô key, đổi dropdown bên phải từ **Text** sang **File**.
   - **Value**: bấm **Select Files**, chọn ảnh biển số.

Bấm **Send**.

Kết quả mong đợi:
```json
{
  "filename": "plate.jpg",
  "text": "ABC1234",
  "indices": [11, 12, 13, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "confidences": [0.999, 0.998, 0.997, 0.999, 0.999, 1.0, 0.999],
  "avg_confidence": 0.9987
}
```

### Ý nghĩa các trường

| Trường           | Ý nghĩa                                                                                |
|------------------|----------------------------------------------------------------------------------------|
| `filename`       | Tên ảnh đã upload (để đối chiếu kết quả khi chạy hàng loạt).                          |
| `text`           | Biển số nhận diện được. **Đây là kết quả chính.**                                      |
| `avg_confidence` | Mức độ tự tin tổng thể của model (0.0–1.0). Dưới 0.8 thì nên kiểm tra lại bằng mắt.   |
| `confidences`    | Độ tự tin từng ký tự — hữu ích khi nghi một chữ bị sai.                                |
| `indices`        | Mã số dùng cho code training. Người dùng cuối có thể bỏ qua.                          |

---

## Bước 3 — Đọc nhiều biển số cùng lúc (`/ocr/batch`)

Nhanh hơn gọi `/ocr` nhiều lần — server xử lý tất cả trong một lần chạy GPU.

**Method**: `POST`
**URL**: `http://100.111.0.111:8001/ocr/batch`

### Tab Body

1. Chọn **Body** → **form-data**.
2. Thêm **nhiều dòng, tất cả dùng chung tên key** `files`:
   - **Key**: `files` (kiểu **File**)
   - **Value**: chọn ảnh đầu tiên.
   - Bấm **+** để thêm dòng mới, key vẫn là `files`, chọn ảnh tiếp theo.
   - Lặp lại cho mỗi ảnh, **tối đa 1000 file mỗi request**.

> Mẹo Postman: ở phiên bản Postman mới, sau khi đặt key là `files`, bạn có thể chọn nhiều file cùng lúc. Giữ Ctrl/Cmd khi chọn.

Bấm **Send**.

Kết quả mong đợi:
```json
{
  "results": [
    {
      "filename": "plate1.jpg",
      "text": "ABC1234",
      "indices": [...],
      "confidences": [...],
      "avg_confidence": 0.998
    },
    {
      "filename": "plate2.jpg",
      "text": "XYZ9876",
      "indices": [...],
      "confidences": [...],
      "avg_confidence": 0.995
    }
  ],
  "count": 2
}
```

- `results[i]` có **thứ tự giống** với thứ tự file bạn upload.
- Dùng `filename` để đối chiếu kết quả với file gốc trên máy.
- `count` là số lượng kết quả (= `len(results)`).

### Giới hạn

| Giới hạn                  | Giá trị | Khi vượt quá                                                  |
|---------------------------|---------|---------------------------------------------------------------|
| Số file mỗi request       | 1000    | Postman báo "Too many files" / server từ chối upload          |
| Kích thước mỗi file       | 8 MB    | Server trả về `413 upload exceeds 8388608 bytes`              |

Nếu cần xử lý hơn 1000 ảnh, chia thành nhiều request `/ocr/batch`.

---

## Lỗi thường gặp

| Status | Ví dụ body                                  | Cách xử lý                                                       |
|--------|---------------------------------------------|------------------------------------------------------------------|
| 400    | `{"detail": "empty upload"}`                | Trường file trống. Gắn lại ảnh.                                  |
| 400    | `{"detail": "cannot decode image"}`         | File không phải JPEG/PNG hợp lệ. Export lại hoặc đổi file khác. |
| 413    | `{"detail": "upload exceeds 8388608 bytes"}`| File > 8 MB. Lưu lại ở chất lượng / kích thước nhỏ hơn.         |
| 503    | `{"detail": "OCR not initialised"}`         | Server đang load model. Chờ ~10 giây rồi thử lại.                |
| —      | Connection refused / timeout                | Server tắt hoặc URL sai. Hỏi admin.                              |

---

## Mẹo để có kết quả tốt

- **Cắt sát** biển số. Càng nhiều nền thừa, độ chính xác càng giảm.
- **Sáng / mờ**: ảnh quá tối, mờ do chuyển động, hoặc độ phân giải thấp dễ sai. Kiểm tra `avg_confidence` — dưới 0.8 nghĩa là kết quả có thể sai dù `text` nhìn có vẻ hợp lý.
- **Độ phân giải**: model tự resize, không cần kích thước cố định. Khoảng ~200 px chiều ngang là đủ.
- **Tên file quan trọng**: API trả lại tên file. Đặt tên có nghĩa để dễ đối chiếu kết quả batch.

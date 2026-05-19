# Install Guide: LP Recognition — PaddleOCR + fast-plate-ocr

## Option 1: PaddleOCR (Production-grade)

**GitHub:** https://github.com/PaddlePaddle/PaddleOCR
**License:** Apache-2.0
**Stars:** ~78.1k

### Install
```bash
pip install paddlepaddle paddleocr
```

### Usage: LP Text Recognition
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr('deblurred_plate.jpg', cls=True)
for line in result:
    print(line[1][0])  # Recognized text
```

### Docker
```bash
docker pull paddlepaddle/paddleocr:latest
docker run -it --rm paddlepaddle/paddleocr /bin/bash
```

## Option 2: fast-plate-ocr (Lightweight, Edge)

**GitHub:** https://github.com/ankandrew/fast-plate-ocr
**License:** MIT
**Stars:** ~565

### Install
```bash
pip install fast-plate-ocr[onnx]
```

### Usage
```bash
python -m fast_plate_ocr.cli.predict \
  --image-path /path/to/plate.png \
  --config-path /path/to/config.yaml \
  --weights-path /path/to/model.onnx
```

Or as library:
```python
from fast_plate_ocr import ONNXPlateRecognizer

model = ONNXPlateRecognizer('european-plates-mobilenet-v3')
text = model('deblurred_plate.jpg')
print(text)
```

## Option 3: fast-alpr (Full Pipeline)

**GitHub:** https://github.com/ankandrew/fast-alpr
**License:** MIT
**Stars:** ~534

### Install
```bash
pip install fast-alpr[onnx]
```

### Usage
```python
from fast_alpr import ALPR

alpr = ALPR()
result = alpr('vehicle_image.jpg')
for plate in result:
    print(plate.text, plate.confidence)
```

## Recommendation
- **PaddleOCR** for best accuracy on deblurred plates (Apache-2.0, active dev)
- **fast-plate-ocr** for edge deployment (MIT, lightweight ONNX)

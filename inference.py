#!/usr/bin/env python3
"""Standalone inference script for LPDGAN.
Runs the generator on input images to deblur/sharpen them.

Usage:
    python inference.py --input path/to/blurry_image.jpg --output ./output/
    python inference.py --input ./dataset/test/blur/ --output ./output/
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
from glob import glob

import albumentations as albu
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models.LPDGAN import create_model


def get_transforms_fortest(size):
    resize = albu.Resize(height=size[0], width=size[1])
    pipeline = albu.Compose([resize], additional_targets={"target": "image"})
    def process(a, b):
        r = pipeline(image=a, target=b)
        return r["image"], r["target"]
    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={"target": "image"})
    def process(a, b):
        r = normalize(image=a, target=b)
        return r["image"], r["target"]
    return process


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    transform_fn = get_transforms_fortest(size=(112, 224))
    transform_fn1 = get_transforms_fortest(size=(56, 112))
    transform_fn2 = get_transforms_fortest(size=(28, 56))
    transform_fn3 = get_transforms_fortest(size=(14, 28))
    normalize_fn = get_normalize()

    img, _ = transform_fn(image, image)
    img1, _ = transform_fn1(image, image)
    img2, _ = transform_fn2(image, image)
    img3, _ = transform_fn3(image, image)

    img, _ = normalize_fn(img, img)
    img1, _ = normalize_fn(img1, img1)
    img2, _ = normalize_fn(img2, img2)
    img3, _ = normalize_fn(img3, img3)

    to_tensor = T.ToTensor()
    return {
        "A": to_tensor(img).unsqueeze(0),
        "A1": to_tensor(img1).unsqueeze(0),
        "A2": to_tensor(img2).unsqueeze(0),
        "A3": to_tensor(img3).unsqueeze(0),
        "path": image_path,
    }


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data[0].cpu().float().numpy()
        image_numpy = np.transpose(image_tensor, (1, 2, 0))
        image_numpy = (image_numpy + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_output(output_tensor, output_path):
    image_numpy = tensor2im(output_tensor)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(output_path)


class MockOpt:
    def __init__(self, checkpoint_dir, epoch):
        self.name = "LPDGAN"
        self.mode = "test"
        self.gpu_ids = "0"
        self.input_nc = 3
        self.output_nc = 3
        self.ndf = 64
        self.gan_mode = "wgangp"
        self.lr = 0.0002
        self.checkpoints_dir = checkpoint_dir
        self.continue_train = False
        self.load_iter = 0
        self.epoch = epoch


def load_model(checkpoint_dir, epoch="latest"):
    # save_dir = checkpoint_dir + name, so pass checkpoint_dir as the parent dir
    opt = MockOpt(os.path.dirname(checkpoint_dir.rstrip("/")), epoch)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model


def infer_image(model, image_path, output_dir):
    data = preprocess_image(image_path)
    # Use same device as model (CPU/CUDA/MPS)
    device = next(model.netG.parameters()).device
    data["A"] = data["A"].to(device)
    data["A1"] = data["A1"].to(device)
    data["A2"] = data["A2"].to(device)
    data["A3"] = data["A3"].to(device)

    with torch.no_grad():
        fake_B, fake_B1, fake_B2, fake_B3, plate1, plate2 = model.netG(
            data["A"], data["A1"], data["A2"], data["A3"],
        )

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_sharp.png")
    save_output(fake_B, output_path)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="LPDGAN Inference")
    parser.add_argument("--input", "-i", required=True, help="Input image or folder")
    parser.add_argument("--output", "-o", default="./inference_output/", help="Output directory")
    parser.add_argument("--checkpoint_dir", "-c", default="./checkpoints/LPDGAN", help="Checkpoint directory")
    parser.add_argument("--epoch", "-e", default="latest", help="Checkpoint epoch (e.g., 20, latest)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        image_files = glob(os.path.join(args.input, "*.jpg")) + \
                      glob(os.path.join(args.input, "*.png")) + \
                      glob(os.path.join(args.input, "*.jpeg"))
        print(f"Found {len(image_files)} images in {args.input}")
    elif os.path.isfile(args.input):
        image_files = [args.input]
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"Loading generator from {args.checkpoint_dir} (epoch: {args.epoch})...")
    model = load_model(args.checkpoint_dir, args.epoch)

    for img_path in image_files:
        try:
            infer_image(model, img_path, args.output)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\nDone! Outputs saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Augment a YOLO-format dataset to a target number of images using Albumentations.

Input:
- A directory of images
- A directory of YOLO labels (.txt) with the same basenames

Output:
- An output directory with augmented images and labels

Usage example:
  python augment_yolo_dataset.py \
    --images "../Frames" \
    --labels "data/yolo_labels" \
    --out "data/aug" \
    --target 5000
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A


def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            labels.append((cls_id, cx, cy, w, h))
    return labels


def save_yolo_labels(label_path: Path, labels: List[Tuple[int, float, float, float, float]]) -> None:
    if not labels:
        # write empty file (valid for images with no objs)
        label_path.write_text('')
        return
    with open(label_path, 'w', encoding='utf-8') as f:
        for cls_id, cx, cy, w, h in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def build_augmentations(image_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=image_size, p=1.0),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.2),
        A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-10, 10), shear=(-5, 5), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        A.Blur(blur_limit=(3, 5), p=0.1),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))


def find_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images: List[Path] = []
    for ext in exts:
        images.extend(images_dir.glob(f"*{ext}"))
    pairs: List[Tuple[Path, Path]] = []
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def augment_dataset(images_dir: str, labels_dir: str, out_dir: str, target: int, image_size: int = 640, seed: int = 42) -> None:
    rng = random.Random(seed)
    images_dir_p = Path(images_dir)
    labels_dir_p = Path(labels_dir)
    out_images = Path(out_dir) / 'images'
    out_labels = Path(out_dir) / 'labels'
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(images_dir_p, labels_dir_p)
    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {images_dir} and {labels_dir}")

    # First, copy originals
    for img_path, lbl_path in tqdm(pairs, desc='Copy originals'):
        dst_img = out_images / img_path.name
        dst_lbl = out_labels / lbl_path.name
        if not dst_img.exists():
            _ = cv2.imwrite(str(dst_img), cv2.imread(str(img_path)))
        if not dst_lbl.exists():
            dst_lbl.write_text(lbl_path.read_text(encoding='utf-8'))

    current = len(list(out_images.glob('*.*')))
    if current >= target:
        print(f"Already have {current} images >= target {target}")
        return

    aug = build_augmentations(image_size=image_size)

    # Augment until reaching target
    idx = 0
    with tqdm(total=target - current, desc='Augmenting') as pbar:
        while current < target:
            img_path, lbl_path = rng.choice(pairs)
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            labels = load_yolo_labels(lbl_path)
            class_labels = [int(l[0]) for l in labels]
            bboxes = [list(l[1:]) for l in labels]

            try:
                transformed = aug(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_classes = transformed['class_labels']
            except Exception:
                # Occasionally augmentations may fail due to invalid boxes
                continue

            # Skip empty bbox results to ensure useful training samples
            if len(aug_bboxes) == 0:
                continue

            # Save
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            out_img_name = f"{img_path.stem}_aug_{idx:06d}.jpg"
            out_lbl_name = f"{img_path.stem}_aug_{idx:06d}.txt"
            cv2.imwrite(str(out_images / out_img_name), aug_img_bgr)

            yolo_labels = []
            for cls_id, (cx, cy, w, h) in zip(aug_classes, aug_bboxes):
                # Clamp to [0,1]
                cx = float(max(0.0, min(1.0, cx)))
                cy = float(max(0.0, min(1.0, cy)))
                w = float(max(0.0, min(1.0, w)))
                h = float(max(0.0, min(1.0, h)))
                # Skip extremely small boxes after transforms
                if w <= 0.001 or h <= 0.001:
                    continue
                yolo_labels.append((int(cls_id), cx, cy, w, h))

            if not yolo_labels:
                # If after filtering there's no box, skip saving label to keep consistency
                # Remove image saved
                try:
                    (out_images / out_img_name).unlink(missing_ok=True)
                except TypeError:
                    # Python<3.8 compatibility
                    if (out_images / out_img_name).exists():
                        (out_images / out_img_name).unlink()
                continue

            save_yolo_labels(out_labels / out_lbl_name, yolo_labels)

            current += 1
            idx += 1
            pbar.update(1)

    print(f"Done. Augmented dataset at: {out_dir}")
    print(f"Total images: {len(list(out_images.glob('*.*')))}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Augment YOLO dataset to a target size')
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to YOLO labels directory')
    parser.add_argument('--out', type=str, required=True, help='Output base directory')
    parser.add_argument('--target', type=int, default=5000, help='Target number of images (including originals)')
    parser.add_argument('--img_size', type=int, default=640, help='Augmented image size (default 640)')
    args = parser.parse_args()

    augment_dataset(args.images, args.labels, args.out, args.target, image_size=args.img_size)


if __name__ == '__main__':
    main()


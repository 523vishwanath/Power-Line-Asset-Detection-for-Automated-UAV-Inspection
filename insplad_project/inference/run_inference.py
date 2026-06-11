"""
run_inference.py

Runs a trained RF-DETR model over a folder of images (or a single image) and saves
annotated copies showing detected power-line assets — useful for generating before/after
demo images for documentation, or for batch-processing drone inspection footage.

Usage:
    python run_inference.py \
        --weights runs/insplad_rfdetr/checkpoint_best_total.pth \
        --source /path/to/images_or_single_image.jpg \
        --output_dir outputs/predictions \
        --threshold 0.4 \
        --model base
"""

import argparse
import os
import glob

import supervision as sv
from PIL import Image

from rfdetr import RFDETRBase, RFDETRLarge

MODEL_REGISTRY = {
    "base": RFDETRBase,
    "large": RFDETRLarge,
}

CLASSES = [
    "spiral_damper",
    "stockbridge_damper",
    "glass_insulator",
    "glass_insulator_big_shackle",
    "glass_insulator_small_shackle",
    "glass_insulator_tower_shackle",
    "lightning_rod_shackle",
    "lightning_rod_suspension",
    "tower_id_plate",
    "polymer_insulator",
    "polymer_insulator_lower_shackle",
    "polymer_insulator_upper_shackle",
    "polymer_insulator_tower_shackle",
    "spacer",
    "vari_grip",
    "yoke",
    "yoke_suspension",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def gather_images(source):
    if os.path.isdir(source):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(source, f"*{ext}")))
        return sorted(files)
    return [source]


def main():
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on InsPLAD images")
    parser.add_argument("--weights", required=True, help="Path to checkpoint_best_total.pth")
    parser.add_argument("--source", required=True, help="Image file or directory of images")
    parser.add_argument("--output_dir", default="outputs/predictions")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--model", choices=["base", "large"], default="base")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(pretrain_weights=args.weights)
    print(f"Loaded checkpoint: {args.weights}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    images = gather_images(args.source)
    print(f"Running inference on {len(images)} image(s)...")

    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        detections = model.predict(image, threshold=args.threshold)

        labels = [
            f"{CLASSES[c]} {conf:.2f}"
            for c, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        out_path = os.path.join(args.output_dir, os.path.basename(img_path))
        annotated.save(out_path)
        print(f"  {os.path.basename(img_path)}: {len(detections)} detections -> {out_path}")

    print(f"\nDone. Annotated images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

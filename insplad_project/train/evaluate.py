"""
evaluate.py

Loads a trained RF-DETR checkpoint and computes COCO-style detection metrics
(mAP50, mAP50-95) on a held-out split using `supervision`.

Usage:
    python evaluate.py \
        --weights runs/insplad_rfdetr/checkpoint_best_total.pth \
        --dataset_dir /path/to/InsPLAD-det-coco \
        --split test \
        --threshold 0.4 \
        --model base
"""

import argparse

import supervision as sv
from supervision.metrics import MeanAveragePrecision

from rfdetr import RFDETRBase, RFDETRLarge

MODEL_REGISTRY = {
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RF-DETR checkpoint")
    parser.add_argument("--weights", required=True, help="Path to checkpoint_best_total.pth")
    parser.add_argument("--dataset_dir", required=True, help="COCO-format dataset root")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--model", choices=["base", "large"], default="base")
    args = parser.parse_args()

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(pretrain_weights=args.weights)
    print(f"Loaded checkpoint: {args.weights}")

    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=f"{args.dataset_dir}/{args.split}",
        annotations_path=f"{args.dataset_dir}/{args.split}/_annotations.coco.json",
    )

    predictions, targets = [], []
    for _, image, annotations in dataset:
        detections = model.predict(image, threshold=args.threshold)
        predictions.append(detections)
        targets.append(annotations)

    result = MeanAveragePrecision().update(predictions, targets).compute()

    print(f"\n{args.split} set results ({len(dataset)} images):")
    print(f"  mAP50:    {result.map50:.4f}")
    print(f"  mAP50-95: {result.map50_95:.4f}")


if __name__ == "__main__":
    main()

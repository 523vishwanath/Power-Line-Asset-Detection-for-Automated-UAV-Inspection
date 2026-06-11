"""
train_rfdetr.py

Fine-tunes RF-DETR (Roboflow's real-time DETR variant) on the InsPLAD-det power-line
asset dataset, in COCO format (see data_pipeline/convert_to_coco.py).

Usage:
    python train_rfdetr.py \
        --dataset_dir /path/to/InsPLAD-det-coco \
        --output_dir runs/insplad_rfdetr \
        --epochs 25 \
        --batch_size 8 \
        --grad_accum_steps 2 \
        --model base
"""

import argparse

from rfdetr import RFDETRBase, RFDETRLarge


MODEL_REGISTRY = {
    "base": RFDETRBase,
    "large": RFDETRLarge,
}


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR on InsPLAD-det")
    parser.add_argument("--dataset_dir", required=True, help="COCO-format dataset root (train/valid/test)")
    parser.add_argument("--output_dir", default="runs/insplad_rfdetr")
    parser.add_argument("--model", choices=["base", "large"], default="base")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=560, help="Must be divisible by 56")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--resume", default=None, help="Path to checkpoint.pth to resume from")
    args = parser.parse_args()

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls()

    train_kwargs = dict(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resolution=args.resolution,
        output_dir=args.output_dir,
        early_stopping=True,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

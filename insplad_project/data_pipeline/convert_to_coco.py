"""
convert_to_coco.py

Converts the raw InsPLAD-det dataset (Pascal VOC XML / COCO JSON / YOLO txt annotations,
depending on the export you downloaded) into the COCO-format directory layout that
RF-DETR expects:

    <output_root>/
        train/
            _annotations.coco.json
            *.jpg
        valid/
            _annotations.coco.json
            *.jpg
        test/
            _annotations.coco.json
            *.jpg

Usage:
    python convert_to_coco.py \
        --raw_dir /path/to/InsPLAD-det-raw \
        --output_root /path/to/InsPLAD-det-coco \
        --train_split 0.70 --val_split 0.15
"""

import argparse
import glob
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

from PIL import Image


# ---------------------------------------------------------------------------
# 17 InsPLAD-det asset classes (Vieira e Silva et al., 2023)
# ---------------------------------------------------------------------------
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
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}  # 0-indexed; COCO category id = id + 1

# Normalises raw label strings found in the dataset to the canonical names above.
CLASS_ALIASES = {
    "damper - spiral": "spiral_damper",
    "spiral damper": "spiral_damper",
    "damper-spiral": "spiral_damper",
    "damper - stockbridge": "stockbridge_damper",
    "stockbridge damper": "stockbridge_damper",
    "damper-stockbridge": "stockbridge_damper",
    "glass insulator": "glass_insulator",
    "glass insulator big shackle": "glass_insulator_big_shackle",
    "glass insulator's big shackle": "glass_insulator_big_shackle",
    "glass insulator small shackle": "glass_insulator_small_shackle",
    "glass insulator's small shackle": "glass_insulator_small_shackle",
    "glass insulator tower shackle": "glass_insulator_tower_shackle",
    "glass insulator's tower shackle": "glass_insulator_tower_shackle",
    "lightning rod shackle": "lightning_rod_shackle",
    "lightning rod's shackle": "lightning_rod_shackle",
    "lightning rod suspension": "lightning_rod_suspension",
    "lightning rod's suspension": "lightning_rod_suspension",
    "tower id plate": "tower_id_plate",
    "tower ID plate": "tower_id_plate",
    "polymer insulator": "polymer_insulator",
    "polymer insulator lower shackle": "polymer_insulator_lower_shackle",
    "polymer insulator's lower shackle": "polymer_insulator_lower_shackle",
    "polymer insulator upper shackle": "polymer_insulator_upper_shackle",
    "polymer insulator's upper shackle": "polymer_insulator_upper_shackle",
    "polymer insulator tower shackle": "polymer_insulator_tower_shackle",
    "polymer insulator's tower shackle": "polymer_insulator_tower_shackle",
    "spacer": "spacer",
    "sphere": "spacer",
    "vari-grip": "vari_grip",
    "vari grip": "vari_grip",
    "varigrip": "vari_grip",
    "yoke": "yoke",
    "yoke suspension": "yoke_suspension",
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def normalize_class_name(name):
    key = name.strip().lower()
    if key in CLASS_TO_ID:
        return key
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    key2 = key.replace("-", " ").replace("_", " ").strip()
    if key2 in CLASS_ALIASES:
        return CLASS_ALIASES[key2]
    key3 = key2.replace(" ", "_")
    if key3 in CLASS_TO_ID:
        return key3
    return None


def find_annotation(img_path):
    """Look for a same-stem .xml/.txt/.json annotation, or one in a sibling labels dir."""
    stem = os.path.splitext(img_path)[0]
    for ext in (".xml", ".txt", ".json"):
        cand = stem + ext
        if os.path.exists(cand):
            return cand
    for alt_dir in ("labels", "Labels", "annotations", "Annotations"):
        alt = img_path.replace("/images/", f"/{alt_dir}/").replace("/Images/", f"/{alt_dir}/")
        for ext in (".xml", ".txt", ".json"):
            cand = os.path.splitext(alt)[0] + ext
            if os.path.exists(cand):
                return cand
    return None


def parse_voc_xml(xml_path):
    boxes = []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes


def parse_yolo_txt(txt_path, img_w, img_h):
    """YOLO file -> pixel boxes; class is already an int id."""
    boxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id, xc, yc, w, h = parts[:5]
            cls_id = int(float(cls_id))
            xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
            xmin = (xc - w / 2) * img_w
            ymin = (yc - h / 2) * img_h
            xmax = (xc + w / 2) * img_w
            ymax = (yc + h / 2) * img_h
            boxes.append((cls_id, xmin, ymin, xmax, ymax))
    return boxes


def load_coco_lookup(raw_dir, all_images):
    """Build filename -> [(class_name, x1, y1, x2, y2), ...] from any COCO json in raw_dir."""
    coco_lookup = None
    coco_json_candidates = glob.glob(f"{raw_dir}/**/*.json", recursive=True)
    has_xml = any(os.path.exists(os.path.splitext(p)[0] + ".xml") for p in all_images[:50])
    if not coco_json_candidates or has_xml:
        return None

    for jp in coco_json_candidates:
        try:
            with open(jp) as f:
                data = json.load(f)
            if not all(k in data for k in ("images", "annotations", "categories")):
                continue
            cat_map = {c["id"]: c["name"] for c in data["categories"]}
            img_map = {im["id"]: im["file_name"] for im in data["images"]}
            lookup = {}
            for ann in data["annotations"]:
                fn = os.path.basename(img_map[ann["image_id"]])
                x, y, w, h = ann["bbox"]
                lookup.setdefault(fn, []).append((cat_map[ann["category_id"]], x, y, x + w, y + h))
            coco_lookup = coco_lookup or {}
            coco_lookup.update(lookup)
            print(f"Loaded COCO annotations from {jp} ({len(lookup)} images)")
        except Exception:
            continue
    return coco_lookup


def convert_dataset(raw_dir, output_root, train_split=0.70, val_split=0.15, seed=42):
    random.seed(seed)

    for split in ("train", "valid", "test"):
        os.makedirs(f"{output_root}/{split}", exist_ok=True)

    all_images = []
    for ext in IMG_EXTS:
        all_images.extend(glob.glob(f"{raw_dir}/**/*{ext}", recursive=True))
    print(f"Found {len(all_images)} images in raw dataset.")

    coco_lookup = load_coco_lookup(raw_dir, all_images)

    unmapped = set()
    converted = []
    no_annotation = 0

    for img_path in all_images:
        fname = os.path.basename(img_path)
        try:
            with Image.open(img_path) as im:
                w_img, h_img = im.size
        except Exception:
            continue

        boxes = []
        ann_path = find_annotation(img_path)

        if ann_path and ann_path.endswith(".xml"):
            for name, xmin, ymin, xmax, ymax in parse_voc_xml(ann_path):
                norm = normalize_class_name(name)
                if norm is None:
                    unmapped.add(name)
                    continue
                cid = CLASS_TO_ID[norm]
                boxes.append((cid, xmin, ymin, xmax - xmin, ymax - ymin))

        elif ann_path and ann_path.endswith(".txt"):
            for cid, xmin, ymin, xmax, ymax in parse_yolo_txt(ann_path, w_img, h_img):
                if not (0 <= cid < len(CLASSES)):
                    continue
                boxes.append((cid, xmin, ymin, xmax - xmin, ymax - ymin))

        elif coco_lookup and fname in coco_lookup:
            for name, xmin, ymin, xmax, ymax in coco_lookup[fname]:
                norm = normalize_class_name(name)
                if norm is None:
                    unmapped.add(name)
                    continue
                cid = CLASS_TO_ID[norm]
                boxes.append((cid, xmin, ymin, xmax - xmin, ymax - ymin))

        else:
            no_annotation += 1
            continue

        converted.append((img_path, w_img, h_img, boxes))

    print(f"\nConverted {len(converted)} / {len(all_images)} images.")
    print(f"Images with no annotation file found: {no_annotation}")
    if unmapped:
        print(f"\nUnmapped class names found ({len(unmapped)}):")
        for u in sorted(unmapped):
            print(f"   - '{u}'")
        print("Add these to CLASS_ALIASES and re-run if counts look high.")

    # ------------------------------------------------------------------
    # Split and write COCO json per split
    # ------------------------------------------------------------------
    random.shuffle(converted)
    n = len(converted)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    splits = {
        "train": converted[:n_train],
        "valid": converted[n_train:n_train + n_val],
        "test": converted[n_train + n_val:],
    }

    categories = [{"id": i + 1, "name": c, "supercategory": "asset"} for i, c in enumerate(CLASSES)]
    class_counts = Counter()

    for split, items in splits.items():
        images_json, annotations_json, ann_id = [], [], 1

        for img_id, (img_path, w_img, h_img, boxes) in enumerate(items, start=1):
            fname = os.path.basename(img_path)
            shutil.copy(img_path, f"{output_root}/{split}/{fname}")

            images_json.append({"id": img_id, "file_name": fname, "width": w_img, "height": h_img})

            for cid, x, y, bw, bh in boxes:
                annotations_json.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid + 1,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1
                class_counts[cid] += 1

        with open(f"{output_root}/{split}/_annotations.coco.json", "w") as f:
            json.dump({"images": images_json, "annotations": annotations_json, "categories": categories}, f)

        print(f"{split}: {len(images_json)} images, {len(annotations_json)} annotations")

    print(f"\nDone. COCO dataset written to: {output_root}\n")
    print(f"{'Class':35s} {'Instances':>10s}")
    print("-" * 47)
    for cid, name in enumerate(CLASSES):
        print(f"{name:35s} {class_counts.get(cid, 0):10d}")
    print("-" * 47)
    print(f"{'TOTAL':35s} {sum(class_counts.values()):10d}")


def main():
    parser = argparse.ArgumentParser(description="Convert InsPLAD-det raw annotations to COCO format")
    parser.add_argument("--raw_dir", required=True, help="Path to extracted InsPLAD-det raw dataset")
    parser.add_argument("--output_root", required=True, help="Output directory for COCO-format dataset")
    parser.add_argument("--train_split", type=float, default=0.70)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_dataset(args.raw_dir, args.output_root, args.train_split, args.val_split, args.seed)


if __name__ == "__main__":
    main()

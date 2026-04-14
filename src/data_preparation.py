
"""
PHASE 1: DATA PREPARATION
Task: Download, organize, and preprocess few-shot dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json


CONFIG = {
    'RAW_DATA_DIR': Path('data/raw'),
    'FEW_SHOT_DIR': Path('data/few_shot'),
    'PROCESSED_DIR': Path('data/processed'),

    # 5 Traffic Sign Classes
    'CLASSES': {
        1: 'Speed limit 30 km/h',
        2: 'Speed limit 50 km/h',
        5: 'Speed limit 80 km/h',
        14: 'Stop',
        17: 'No entry'
    },

    'IMAGE_SIZE': 32,
    'TRAIN_IMAGES_PER_CLASS': 6,
    'REFERENCE_IMAGES_PER_CLASS': 2,
}


def load_image(image_path, resize=True, normalize=False):
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        if resize:
            img_array = cv2.resize(
                img_array,
                (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']),
                interpolation=cv2.INTER_CUBIC
            )

        if normalize:
            img_array = img_array.astype(np.float32) / 255.0

        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def save_image(image_array, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = (image_array * 255).astype(np.uint8) if image_array.max() <= 1 else image_array.astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img.save(output_path, quality=95)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


class DataPreparation:

    def __init__(self):
        self.config = CONFIG
        self.raw_dir = CONFIG['RAW_DATA_DIR']
        self.few_shot_dir = CONFIG['FEW_SHOT_DIR']
        self.processed_dir = CONFIG['PROCESSED_DIR']
        self.classes = CONFIG['CLASSES']

        # Create directories
        for d in [self.few_shot_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def verify_manual_structure(self):

        print_section("STEP 1: VERIFYING MANUAL FEW-SHOT STRUCTURE")

        train_dir = self.few_shot_dir / "train"
        reference_dir = self.few_shot_dir / "reference"

        if not train_dir.exists() or not reference_dir.exists():
            print(" ERROR: Few-shot directories not found!")
            print("\n📋 MANUAL SETUP REQUIRED:")
            print(f"\n1. Create folder structure:")
            print(f"   data/few_shot/train/")
            print(f"   data/few_shot/reference/")
            print(f"\n2. For each class, create:")
            for class_id, class_name in self.classes.items():
                print(f"   - data/few_shot/train/class_{class_id}/ ({class_name})")
                print(f"   - data/few_shot/reference/class_{class_id}/")
            return False

        # Verify each class has images
        stats = {}
        for class_id in self.classes.keys():
            train_class_dir = train_dir / f"class_{class_id}"
            ref_class_dir = reference_dir / f"class_{class_id}"

            if not train_class_dir.exists():
                print(f" Missing: {train_class_dir}")
                return False

            if not ref_class_dir.exists():
                print(f" Missing: {ref_class_dir}")
                return False

            train_count = len(list(train_class_dir.glob("*.*")))
            ref_count = len(list(ref_class_dir.glob("*.*")))

            stats[class_id] = {'train': train_count, 'reference': ref_count}
            print(f"✓ Class {class_id}: {train_count} train + {ref_count} reference")

        return True

    def convert_ppm_to_jpg(self):
        print_section("STEP 2: CONVERTING PPM → JPG")

        for base_name in ["train", "reference"]:
            base_dir = self.few_shot_dir / base_name

            for class_id in self.classes.keys():
                class_dir = base_dir / f"class_{class_id}"

                if not class_dir.exists():
                    continue

                image_files = list(class_dir.glob("*.*"))

                for img_file in image_files:
                    if img_file.suffix.lower() == ".jpg":
                        continue

                    try:
                        img = Image.open(img_file).convert('RGB')
                        new_file = img_file.with_suffix('.jpg')
                        img.save(new_file, quality=95)

                        if img_file != new_file and img_file.exists():
                            img_file.unlink()

                        print(f"  ✓ Converted: {img_file.name} → {new_file.name}")
                    except Exception as e:
                        print(f"   Error: {img_file}: {e}")

    def verify_image_quality(self):
        print_section("STEP 3: VERIFYING IMAGE QUALITY")

        all_valid = True

        for base_name in ["train", "reference"]:
            base_dir = self.few_shot_dir / base_name

            for class_id in self.classes.keys():
                class_dir = base_dir / f"class_{class_id}"

                if not class_dir.exists():
                    continue

                image_files = list(class_dir.glob("*.jpg"))

                for img_file in image_files:
                    try:
                        img = Image.open(img_file)
                        width, height = img.size

                        if width != CONFIG['IMAGE_SIZE'] or height != CONFIG['IMAGE_SIZE']:
                            print(f"  {img_file.name}: Size {(width, height)} (expected 32×32)")
                            all_valid = False
                        else:
                            print(f"✓ {img_file.name}: Valid ({width}×{height})")

                    except Exception as e:
                        print(f" {img_file.name}: {e}")
                        all_valid = False

        return all_valid

    def create_processed_dataset(self):
        print_section("STEP 4: CREATING PROCESSED DATASET")

        total = 0

        for base_name in ["train", "reference"]:
            source_base = self.few_shot_dir / base_name
            output_base = self.processed_dir / base_name

            for class_id in self.classes.keys():
                source_class_dir = source_base / f"class_{class_id}"
                output_class_dir = output_base / f"class_{class_id}"
                output_class_dir.mkdir(parents=True, exist_ok=True)

                if not source_class_dir.exists():
                    continue

                image_files = list(source_class_dir.glob("*.jpg"))

                for img_file in tqdm(image_files, desc=f"{base_name} - Class {class_id}"):
                    img = load_image(img_file, resize=True, normalize=False)
                    if img is not None:
                        output_file = output_class_dir / img_file.name
                        save_image(img, output_file)
                        total += 1

        print(f"\n Created processed dataset: {total} images")

    def generate_metadata(self):
        print_section("STEP 5: GENERATING METADATA")

        metadata = {
            'classes': self.classes,
            'image_size': CONFIG['IMAGE_SIZE'],
            'train_images_per_class': CONFIG['TRAIN_IMAGES_PER_CLASS'],
            'reference_images_per_class': CONFIG['REFERENCE_IMAGES_PER_CLASS'],
            'total_classes': len(self.classes),
            'total_images_per_class': CONFIG['TRAIN_IMAGES_PER_CLASS'] + CONFIG['REFERENCE_IMAGES_PER_CLASS'],
        }

        # Calculate totals
        metadata['total_images'] = metadata['total_classes'] * metadata['total_images_per_class']

        metadata_file = self.processed_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f" Metadata saved: {metadata_file}")
        print(f"\n Dataset Statistics:")
        print(f"   - Classes: {metadata['total_classes']}")
        print(f"   - Images per class: {metadata['total_images_per_class']}")
        print(f"   - Total images: {metadata['total_images']}")

        return metadata

    def run_all(self):
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  PHASE 1: DATA PREPARATION".center(68) + "█")
        print("█" + "  Few-Shot Traffic Sign Recognition".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        # Execute steps
        if not self.verify_manual_structure():
            print("\nCannot proceed! Complete manual setup first.")
            return False

        self.convert_ppm_to_jpg()

        if not self.verify_image_quality():
            print("\n⚠  Some image quality issues detected.")

        self.create_processed_dataset()

        metadata = self.generate_metadata()

        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  ✅ PHASE 1 COMPLETE!".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    prep = DataPreparation()
    success = prep.run_all()

    if success:
        print("\n Ready for PHASE 2: Pair Generation")
    else:
        print("\n Please complete manual setup and try again")

"""
PHASE 2: PAIR GENERATION
Task: Create similar and dissimilar image pairs for training
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import random
from PIL import Image


CONFIG = {
    'PROCESSED_DIR': Path('data/processed'),
    'PAIRS_DIR': Path('data/pairs'),
    'IMAGE_SIZE': 32,
    'TRAIN_SPLIT': 0.7,
    'VAL_SPLIT': 0.15,
    'TEST_SPLIT': 0.15,
    'RANDOM_SEED': 42,
}



class PairGenerator:
    """Generate similar and dissimilar image pairs for Siamese training"""

    def __init__(self):
        self.processed_dir = CONFIG['PROCESSED_DIR']
        self.pairs_dir = CONFIG['PAIRS_DIR']
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

        random.seed(CONFIG['RANDOM_SEED'])
        np.random.seed(CONFIG['RANDOM_SEED'])

        # Load metadata
        metadata_file = self.processed_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.classes = self.metadata['classes']
        self.class_ids = sorted([int(k) for k in self.classes.keys()])

    def load_class_images(self, class_id, image_type='train'):
        """Load all images for a specific class"""
        class_dir = self.processed_dir / image_type / f"class_{class_id}"

        images = []
        image_paths = []

        if not class_dir.exists():
            return images, image_paths

        for img_file in sorted(class_dir.glob("*.jpg")):
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    image_paths.append(str(img_file))
            except Exception as e:
                print(f"Error loading {img_file}: {e}")

        return images, image_paths

    def create_similar_pairs(self, max_pairs_per_class=20):

        print("\n" + "=" * 70)
        print("CREATING SIMILAR PAIRS (Same class)")
        print("=" * 70)

        similar_pairs = []

        for class_id in self.class_ids:
            images, image_paths = self.load_class_images(class_id, 'train')

            if len(images) < 2:
                print(f"  Class {class_id}: Not enough images ({len(images)})")
                continue

            # Generate all combinations
            n = len(images)
            pair_count = 0

            for i in range(n):
                for j in range(i + 1, n):
                    similar_pairs.append({
                        'image1_path': image_paths[i],
                        'image2_path': image_paths[j],
                        'image1_array': images[i],
                        'image2_array': images[j],
                        'label': 1,
                        'class_id': class_id,
                        'pair_type': 'similar'
                    })
                    pair_count += 1

                    if pair_count >= max_pairs_per_class:
                        break

                if pair_count >= max_pairs_per_class:
                    break

            print(f"✓ Class {class_id} ({self.classes[str(class_id)]}): {pair_count} similar pairs")

        print(f"\n Total similar pairs: {len(similar_pairs)}")
        return similar_pairs

    def create_dissimilar_pairs(self, num_dissimilar_pairs):

        print("\n" + "=" * 70)
        print(f"CREATING DISSIMILAR PAIRS (Different classes)")
        print("=" * 70)

        # Load all images from all classes
        all_class_images = {}
        for class_id in self.class_ids:
            images, image_paths = self.load_class_images(class_id, 'train')
            if images:
                all_class_images[class_id] = {
                    'images': images,
                    'paths': image_paths
                }

        dissimilar_pairs = []

        for _ in tqdm(range(num_dissimilar_pairs), desc="Generating dissimilar pairs"):
            # Random sample 2 different classes
            class_ids_pair = random.sample(self.class_ids, 2)
            class_A, class_B = class_ids_pair

            if class_A not in all_class_images or class_B not in all_class_images:
                continue

            # Random images from each class
            img_idx_A = random.randint(0, len(all_class_images[class_A]['images']) - 1)
            img_idx_B = random.randint(0, len(all_class_images[class_B]['images']) - 1)

            image_A = all_class_images[class_A]['images'][img_idx_A]
            image_B = all_class_images[class_B]['images'][img_idx_B]
            path_A = all_class_images[class_A]['paths'][img_idx_A]
            path_B = all_class_images[class_B]['paths'][img_idx_B]

            dissimilar_pairs.append({
                'image1_path': path_A,
                'image2_path': path_B,
                'image1_array': image_A,
                'image2_array': image_B,
                'label': 0,
                'class_id_1': class_A,
                'class_id_2': class_B,
                'pair_type': 'dissimilar'
            })

        print(f" Total dissimilar pairs: {len(dissimilar_pairs)}")
        return dissimilar_pairs

    def balance_pairs(self, similar_pairs, dissimilar_pairs):
        print("\n" + "=" * 70)
        print("BALANCING PAIRS")
        print("=" * 70)

        num_similar = len(similar_pairs)
        num_dissimilar = len(dissimilar_pairs)

        print(f"Similar pairs: {num_similar}")
        print(f"Dissimilar pairs: {num_dissimilar}")

        # Balance to smaller set
        min_count = min(num_similar, num_dissimilar)

        similar_pairs = similar_pairs[:min_count]
        dissimilar_pairs = dissimilar_pairs[:min_count]

        print(f"\n Balanced to {min_count} pairs each")
        print(f"   Total pairs: {len(similar_pairs) + len(dissimilar_pairs)}")

        return similar_pairs, dissimilar_pairs

    def split_pairs(self, all_pairs):
        print("\n" + "=" * 70)
        print("SPLITTING PAIRS INTO TRAIN/VAL/TEST")
        print("=" * 70)

        random.shuffle(all_pairs)

        n = len(all_pairs)
        train_idx = int(n * CONFIG['TRAIN_SPLIT'])
        val_idx = train_idx + int(n * CONFIG['VAL_SPLIT'])

        train_pairs = all_pairs[:train_idx]
        val_pairs = all_pairs[train_idx:val_idx]
        test_pairs = all_pairs[val_idx:]

        print(f"Train: {len(train_pairs)} pairs ({CONFIG['TRAIN_SPLIT'] * 100:.0f}%)")
        print(f"Val:   {len(val_pairs)} pairs ({CONFIG['VAL_SPLIT'] * 100:.0f}%)")
        print(f"Test:  {len(test_pairs)} pairs ({CONFIG['TEST_SPLIT'] * 100:.0f}%)")

        return train_pairs, val_pairs, test_pairs

    def save_pairs_to_csv(self, pairs, output_file):
        data = []

        for pair in pairs:
            data.append({
                'image1_path': pair['image1_path'],
                'image2_path': pair['image2_path'],
                'label': pair['label'],
                'pair_type': pair.get('pair_type', 'unknown'),
                'class_id': pair.get('class_id', pair.get('class_id_1', -1))
            })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file} ({len(df)} pairs)")

    def save_pairs_to_npz(self, pairs, output_file):
        images1 = []
        images2 = []
        labels = []

        for pair in tqdm(pairs, desc=f"Saving pairs to {output_file.name}"):
            images1.append(pair['image1_array'])
            images2.append(pair['image2_array'])
            labels.append(pair['label'])

        np.savez_compressed(
            output_file,
            images1=np.array(images1),
            images2=np.array(images2),
            labels=np.array(labels)
        )
        print(f"✓ Saved: {output_file} ({len(labels)} pairs)")

    def run_all(self):
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  PHASE 2: PAIR GENERATION".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        # Step 1: Create similar pairs
        similar_pairs = self.create_similar_pairs(max_pairs_per_class=50)

        # Step 2: Create dissimilar pairs
        dissimilar_pairs = self.create_dissimilar_pairs(num_dissimilar_pairs=len(similar_pairs))

        # Step 3: Balance pairs
        similar_pairs, dissimilar_pairs = self.balance_pairs(similar_pairs, dissimilar_pairs)

        # Step 4: Combine and shuffle
        all_pairs = similar_pairs + dissimilar_pairs
        random.shuffle(all_pairs)

        # Step 5: Split
        train_pairs, val_pairs, test_pairs = self.split_pairs(all_pairs)

        # Step 6: Save to CSV
        print("\n" + "=" * 70)
        print("SAVING PAIRS")
        print("=" * 70)

        self.save_pairs_to_csv(train_pairs, self.pairs_dir / "train_pairs.csv")
        self.save_pairs_to_csv(val_pairs, self.pairs_dir / "val_pairs.csv")
        self.save_pairs_to_csv(test_pairs, self.pairs_dir / "test_pairs.csv")

        # Step 7: Save to NPZ (for faster training)
        self.save_pairs_to_npz(train_pairs, self.pairs_dir / "train_pairs.npz")
        self.save_pairs_to_npz(val_pairs, self.pairs_dir / "val_pairs.npz")
        self.save_pairs_to_npz(test_pairs, self.pairs_dir / "test_pairs.npz")

        # Summary
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  ✅ PHASE 2 COMPLETE!".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        summary = {
            'total_pairs': len(all_pairs),
            'similar_pairs': len(similar_pairs),
            'dissimilar_pairs': len(dissimilar_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'test_pairs': len(test_pairs),
            'train_split': CONFIG['TRAIN_SPLIT'],
            'val_split': CONFIG['VAL_SPLIT'],
            'test_split': CONFIG['TEST_SPLIT']
        }

        with open(self.pairs_dir / "pairs_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n Summary saved: {self.pairs_dir / 'pairs_summary.json'}")

        return True



if __name__ == "__main__":
    generator = PairGenerator()
    generator.run_all()
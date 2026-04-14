
"""
PHASE 4: MODEL TRAINING
Task: Train Siamese network on pairs
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SiameseNetwork


class PairDataLoader:
    """Load image pairs from NPZ files"""

    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.images1 = data['images1'].astype('float32') / 255.0
        self.images2 = data['images2'].astype('float32') / 255.0
        self.labels = data['labels'].astype('float32')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images1[idx], self.images2[idx], self.labels[idx]


class SiameseTrainer:
    """Train Siamese network"""

    def __init__(self):
        self.pairs_dir = Path('data/pairs')
        self.models_dir = Path('models')
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def load_data(self):
        """Load training and validation pairs"""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        train_data = PairDataLoader(self.pairs_dir / "train_pairs.npz")
        val_data = PairDataLoader(self.pairs_dir / "val_pairs.npz")

        print(f"✓ Train pairs: {len(train_data)}")
        print(f"✓ Val pairs: {len(val_data)}")

        return train_data, val_data

    def train(self, epochs=50, batch_size=32):
        """Train Siamese model"""
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  PHASE 4: TRAINING".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        # Load data
        train_data, val_data = self.load_data()

        # Build model
        print("\n" + "=" * 70)
        print("BUILDING MODEL")
        print("=" * 70)

        siamese = SiameseNetwork(input_shape=(32, 32, 3), feature_dim=128)
        model = siamese.build_siamese_model()
        siamese.compile_model(learning_rate=0.001)

        # Display architecture
        siamese.summary()

        # Create TF datasets
        print("\n" + "=" * 70)
        print("CREATING TENSORFLOW DATASETS")
        print("=" * 70)

        def create_pair_dataset(images1, images2, labels, batch_size):
            dataset = tf.data.Dataset.from_tensor_slices(((images1, images2), labels))
            dataset = dataset.shuffle(len(labels))
            dataset = dataset.batch(batch_size)
            return dataset

        # Create datasets
        train_dataset = create_pair_dataset(
            train_data.images1,
            train_data.images2,
            train_data.labels,
            batch_size
        )

        val_dataset = create_pair_dataset(
            val_data.images1,
            val_data.images2,
            val_data.labels,
            batch_size
        )

        print(f"✓ Created datasets")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)

        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )

        # Save model
        print("\n" + "=" * 70)
        print("SAVING MODEL")
        print("=" * 70)

        model_path = self.models_dir / "siamese_model.h5"
        siamese.save_model(model_path)

        # Save history
        training_log = {
            'epochs': epochs,
            'batch_size': batch_size,
            'train_loss': [float(l) for l in history.history['loss']],
            'train_accuracy': [float(a) for a in history.history['accuracy']],
            'val_loss': [float(l) for l in history.history['val_loss']],
            'val_accuracy': [float(a) for a in history.history['val_accuracy']],
        }

        with open(self.models_dir / "training_log.json", 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"✓ Training log saved")

        # Plot results
        self.plot_training_history(history)

        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  ✅ PHASE 4 COMPLETE!".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        return model

    def plot_training_history(self, history):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid()

        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Train Acc')
        axes[1].plot(history.history['val_accuracy'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid()

        plt.tight_layout()
        plt.savefig(self.models_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {self.models_dir / 'training_curves.png'}")
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    trainer = SiameseTrainer()
    trainer.train(epochs=50, batch_size=32)
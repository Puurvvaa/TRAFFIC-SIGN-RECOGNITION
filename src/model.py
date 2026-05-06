
"""
PHASE 3: SIAMESE NETWORK MODEL
Task: Build Siamese architecture for similarity learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import json
from pathlib import Path


class SiameseNetwork:


    def __init__(self, input_shape=(32, 32, 3), feature_dim=128):

        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.model = None
        self.feature_extractor = None

    def build_feature_extractor(self):

        inputs = layers.Input(shape=self.input_shape)

        # Block 1
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Block 2
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Block 3
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.feature_dim, activation='relu')(x)

        feature_extractor = Model(inputs, x, name='feature_extractor')
        return feature_extractor

    def build_siamese_model(self):

        # Build feature extractor
        self.feature_extractor = self.build_feature_extractor()

        # Input layers for two images
        input_img1 = layers.Input(shape=self.input_shape, name='image1')
        input_img2 = layers.Input(shape=self.input_shape, name='image2')

        # Extract features (shared CNN)
        features1 = self.feature_extractor(input_img1)
        features2 = self.feature_extractor(input_img2)

        # Compute absolute difference
        diff = layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]))([features1, features2])

        # Dense layers
        x = layers.Dense(128, activation='relu')(diff)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)

        # Output: similarity score [0, 1]
        similarity_score = layers.Dense(1, activation='sigmoid', name='similarity')(x)

        # Build model
        self.model = Model(
            inputs=[input_img1, input_img2],
            outputs=similarity_score,
            name='SiameseNetwork'
        )

        return self.model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )

    def summary(self):
        """Print model summary"""
        print("\n" + "=" * 70)
        print("SIAMESE NETWORK ARCHITECTURE")
        print("=" * 70)
        self.model.summary()

    def save_model(self, path):
        """Save model to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"✓ Model saved: {path}")

    def load_model(self, path):
        """Load model from file"""
        self.model = keras.models.load_model(path)
        print(f"✓ Model loaded: {path}")



def contrastive_loss(y_true, y_pred, margin=1.0):

    y_true = tf.cast(y_true, tf.float32)

    # Compute Euclidean distance
    distance = y_pred

    # Contrastive loss
    similar_loss = (1 - y_true) * 0.5 * tf.square(distance)
    dissimilar_loss = y_true * 0.5 * tf.square(tf.maximum(margin - distance, 0.0))

    return tf.reduce_mean(similar_loss + dissimilar_loss)



if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SIAMESE NETWORK MODEL BUILDER")
    print("=" * 70)

    # Create model
    siamese = SiameseNetwork(input_shape=(32, 32, 3), feature_dim=128)
    model = siamese.build_siamese_model()
    siamese.compile_model(learning_rate=0.001)

    # Display architecture
    siamese.summary()

    print("\n Model ready for training!")
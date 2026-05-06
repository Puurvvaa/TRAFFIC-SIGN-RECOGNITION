# src/6_prediction.py
"""
PHASE 6 & 7: PREDICTION & STREAMLIT INTEGRATION
Task: Inference and Streamlit UI
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from PIL import Image
import json
import cv2


class ImagePreprocessor:
    """Preprocess images for model input"""

    def __init__(self, image_size=32):
        self.image_size = image_size

    def load_image(self, image_path_or_array):
        """Load image from file or array"""
        if isinstance(image_path_or_array, str) or isinstance(image_path_or_array, Path):
            img = Image.open(image_path_or_array).convert('RGB')
            img_array = np.array(img)
        else:
            img_array = image_path_or_array

        return img_array

    def preprocess(self, image_input):
        """Preprocess image for model"""
        # Handle Streamlit UploadedFile objects
        if hasattr(image_input, 'read'):
            from PIL import Image
            img = Image.open(image_input).convert('RGB')
            img_array = np.array(img)
        else:
            img_array = self.load_image(image_input)

        # Ensure it's numpy array
        if not isinstance(img_array, np.ndarray):
            img_array = np.array(img_array)

        # Resize
        img_resized = cv2.resize(img_array, (self.image_size, self.image_size),
                                 interpolation=cv2.INTER_CUBIC)

        # Normalize
        img_normalized = img_resized.astype('float32') / 255.0

        return img_normalized


class SiamesePredictioner:
    """Make predictions with trained Siamese model"""

    def __init__(self, model_path='models/siamese_model.h5',
                 reference_data_path='data/processed/reference'):
        self.model = keras.models.load_model(model_path)
        self.reference_data_path = Path(reference_data_path)
        self.preprocessor = ImagePreprocessor(image_size=32)

        # Load class mapping
        with open(Path('data/processed/metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.classes = {int(k): v for k, v in metadata['classes'].items()}
        self.class_ids = sorted(self.classes.keys())

        # Load reference images
        self.reference_images = self._load_reference_images()

    def _load_reference_images(self):
        """Load one reference image per class"""
        reference_images = {}

        for class_id in self.class_ids:
            class_dir = self.reference_data_path / f"class_{class_id}"

            if not class_dir.exists():
                continue

            # Get first image
            image_files = list(class_dir.glob("*.jpg"))
            if image_files:
                img_path = image_files[0]
                img_array = self.preprocessor.preprocess(img_path)
                reference_images[class_id] = img_array

        return reference_images

    def predict_single_image(self, image_input):
        """
        Predict class for a single image

        Returns:
            predicted_class: Class ID
            similarity_scores: Dict of all similarity scores
            confidence: Confidence percentage
        """
        # Preprocess input image
        input_image = self.preprocessor.preprocess(image_input)
        input_image_batch = np.expand_dims(input_image, axis=0)

        # Compare with all reference images
        similarity_scores = {}

        for class_id, reference_image in self.reference_images.items():
            reference_batch = np.expand_dims(reference_image, axis=0)

            # Predict similarity
            similarity = self.model.predict(
                [input_image_batch, reference_batch],
                verbose=0
            )[0][0]

            similarity_scores[class_id] = float(similarity)

        # Find class with highest similarity
        predicted_class = max(similarity_scores, key=similarity_scores.get)
        confidence = similarity_scores[predicted_class]

        return predicted_class, similarity_scores, confidence

    def predict_batch(self, image_inputs):
        """Predict for multiple images"""
        results = []

        for image_input in image_inputs:
            pred_class, scores, conf = self.predict_single_image(image_input)
            results.append({
                'predicted_class': pred_class,
                'class_name': self.classes[pred_class],
                'confidence': conf,
                'similarity_scores': scores
            })

        return results

    def get_class_name(self, class_id):
        """Get class name from ID"""
        return self.classes.get(class_id, f"Unknown (Class {class_id})")

    def get_top_predictions(self, similarity_scores, top_k=3):
        """Get top K predictions"""
        sorted_scores = sorted(
            similarity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_predictions = []
        for class_id, score in sorted_scores[:top_k]:
            top_predictions.append({
                'class_id': class_id,
                'class_name': self.get_class_name(class_id),
                'similarity': score,
                'confidence_percent': score * 100
            })

        return top_predictions


# ============================================================================
# UTILITY FUNCTIONS FOR STREAMLIT
# ============================================================================

def load_predictor():
    """Load predictor (cached)"""
    return SiamesePredictioner()


def get_color_for_score(score):
    """Get color based on confidence score"""
    if score > 0.8:
        return "🟢 Very High"
    elif score > 0.6:
        return "🟡 High"
    elif score > 0.4:
        return "🟠 Medium"
    else:
        return "🔴 Low"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Testing Predictioner...")

    predictor = SiamesePredictioner()
    print(f"✓ Loaded model")
    print(f"✓ Classes: {list(predictor.classes.values())}")
    print(f"✓ Reference images: {len(predictor.reference_images)}")
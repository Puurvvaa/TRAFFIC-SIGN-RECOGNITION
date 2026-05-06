# src/5_evaluation.py
"""
PHASE 5: BEFORE & AFTER EVALUATION
Task: Compare model predictions before and after training
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report



class ModelEvaluator:
    """Evaluate Siamese model performance"""

    def __init__(self):
        self.pairs_dir = Path('data/pairs')
        self.models_dir = Path('models')
        self.results_dir = Path('results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load test pairs
        test_data = np.load(self.pairs_dir / "test_pairs.npz")
        self.test_images1 = test_data['images1'].astype('float32') / 255.0
        self.test_images2 = test_data['images2'].astype('float32') / 255.0
        self.test_labels = test_data['labels']

    def get_untrained_predictions(self):
        """Get predictions from random model (baseline)"""
        print("\n" + "=" * 70)
        print("GETTING UNTRAINED MODEL PREDICTIONS (BASELINE)")
        print("=" * 70)

        # Random predictions
        random_scores = np.random.rand(len(self.test_labels))

        print(f"✓ Generated {len(random_scores)} random predictions")
        print(f"  Mean score: {random_scores.mean():.4f}")
        print(f"  Std score:  {random_scores.std():.4f}")

        return random_scores

    def get_trained_predictions(self):
        """Get predictions from trained model"""
        print("\n" + "=" * 70)
        print("GETTING TRAINED MODEL PREDICTIONS")
        print("=" * 70)

        model_path = self.models_dir / "siamese_model.h5"
        model = keras.models.load_model(model_path)

        predictions = model.predict([self.test_images1, self.test_images2], verbose=0)
        predictions = predictions.flatten()

        print(f"✓ Generated {len(predictions)} predictions")
        print(f"  Mean score: {predictions.mean():.4f}")
        print(f"  Std score:  {predictions.std():.4f}")

        return predictions

    def compute_metrics(self, predictions, labels, prefix=""):
        """Compute evaluation metrics"""
        # Threshold at 0.5
        predicted_labels = (predictions > 0.5).astype(int)

        # Accuracy
        accuracy = np.mean(predicted_labels == labels)

        # Precision, Recall, F1
        tn = np.sum((predicted_labels == 0) & (labels == 0))
        fp = np.sum((predicted_labels == 1) & (labels == 0))
        fn = np.sum((predicted_labels == 0) & (labels == 1))
        tp = np.sum((predicted_labels == 1) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # AUC-ROC
        try:
            auc_roc = roc_auc_score(labels, predictions)
        except:
            auc_roc = 0.0

        metrics = {
            f'{prefix}_accuracy': float(accuracy),
            f'{prefix}_precision': float(precision),
            f'{prefix}_recall': float(recall),
            f'{prefix}_f1': float(f1),
            f'{prefix}_auc_roc': float(auc_roc),
            f'{prefix}_true_negatives': int(tn),
            f'{prefix}_false_positives': int(fp),
            f'{prefix}_false_negatives': int(fn),
            f'{prefix}_true_positives': int(tp),
        }

        return metrics

    def create_comparison_table(self, untrained_preds, trained_preds):
        """Create before/after comparison table"""
        print("\n" + "=" * 70)
        print("CREATING COMPARISON TABLE")
        print("=" * 70)

        comparison_data = {
            'pair_id': range(len(self.test_labels)),
            'untrained_score': untrained_preds,
            'trained_score': trained_preds,
            'true_label': self.test_labels,
            'label_text': ['Similar' if l == 1 else 'Dissimilar' for l in self.test_labels]
        }

        df = pd.DataFrame(comparison_data)

        # Save to CSV
        csv_file = self.results_dir / "before_after_comparison.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved: {csv_file}")

        return df

    def print_metrics_table(self, untrained_metrics, trained_metrics):
        """Print metrics comparison table"""
        print("\n" + "=" * 70)
        print("METRICS COMPARISON: BEFORE vs AFTER TRAINING")
        print("=" * 70)

        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']

        print(f"\n{'Metric':<20} {'BEFORE TRAINING':<20} {'AFTER TRAINING':<20} {'IMPROVEMENT':<15}")
        print("-" * 75)

        for metric in metrics_names:
            before = untrained_metrics[f'untrained_{metric}']
            after = trained_metrics[f'trained_{metric}']
            improvement = after - before

            print(f"{metric:<20} {before:<20.4f} {after:<20.4f} {improvement:>+.4f}")

    def plot_score_distributions(self, untrained_preds, trained_preds):
        """Plot similarity score distributions"""
        print("\n" + "=" * 70)
        print("PLOTTING SIMILARITY DISTRIBUTIONS")
        print("=" * 70)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Before training - Similar pairs
        similar_mask = self.test_labels == 1
        axes[0, 0].hist(untrained_preds[similar_mask], bins=20, alpha=0.7, label='Similar', color='blue')
        axes[0, 0].set_title('BEFORE: Similar Pairs')
        axes[0, 0].set_xlabel('Similarity Score')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid()

        # Before training - Dissimilar pairs
        dissimilar_mask = self.test_labels == 0
        axes[0, 1].hist(untrained_preds[dissimilar_mask], bins=20, alpha=0.7, label='Dissimilar', color='orange')
        axes[0, 1].set_title('BEFORE: Dissimilar Pairs')
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid()

        # After training - Similar pairs
        axes[1, 0].hist(trained_preds[similar_mask], bins=20, alpha=0.7, label='Similar', color='blue')
        axes[1, 0].set_title('AFTER: Similar Pairs')
        axes[1, 0].set_xlabel('Similarity Score')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid()

        # After training - Dissimilar pairs
        axes[1, 1].hist(trained_preds[dissimilar_mask], bins=20, alpha=0.7, label='Dissimilar', color='orange')
        axes[1, 1].set_title('AFTER: Dissimilar Pairs')
        axes[1, 1].set_xlabel('Similarity Score')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid()

        plt.tight_layout()
        plt.savefig(self.results_dir / "similarity_distributions.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {self.results_dir / 'similarity_distributions.png'}")
        plt.close()

    def plot_roc_curves(self, untrained_preds, trained_preds):
        """Plot ROC curves"""
        print("\n" + "=" * 70)
        print("PLOTTING ROC CURVES")
        print("=" * 70)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Before training
        fpr_before, tpr_before, _ = roc_curve(self.test_labels, untrained_preds)
        auc_before = roc_auc_score(self.test_labels, untrained_preds) if len(np.unique(self.test_labels)) > 1 else 0.5
        axes[0].plot(fpr_before, tpr_before, label=f'AUC = {auc_before:.3f}', linewidth=2)
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve - BEFORE Training')
        axes[0].legend()
        axes[0].grid()

        # After training
        fpr_after, tpr_after, _ = roc_curve(self.test_labels, trained_preds)
        auc_after = roc_auc_score(self.test_labels, trained_preds)
        axes[1].plot(fpr_after, tpr_after, label=f'AUC = {auc_after:.3f}', linewidth=2, color='green')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve - AFTER Training')
        axes[1].legend()
        axes[1].grid()

        plt.tight_layout()
        plt.savefig(self.results_dir / "roc_curves.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {self.results_dir / 'roc_curves.png'}")
        plt.close()

    def run_evaluation(self):
        """Run complete evaluation"""
        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  PHASE 5: EVALUATION".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        # Get predictions
        untrained_preds = self.get_untrained_predictions()
        trained_preds = self.get_trained_predictions()

        # Compute metrics
        untrained_metrics = self.compute_metrics(untrained_preds, self.test_labels, prefix="untrained")
        trained_metrics = self.compute_metrics(trained_preds, self.test_labels, prefix="trained")

        # Print metrics
        self.print_metrics_table(untrained_metrics, trained_metrics)

        # Save metrics
        all_metrics = {**untrained_metrics, **trained_metrics}
        with open(self.results_dir / "metrics.json", 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"✓ Saved metrics: {self.results_dir / 'metrics.json'}")

        # Create comparison table
        comparison_df = self.create_comparison_table(untrained_preds, trained_preds)

        # Plot distributions
        self.plot_score_distributions(untrained_preds, trained_preds)

        # Plot ROC curves
        self.plot_roc_curves(untrained_preds, trained_preds)

        print("\n" + "█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "  ✅ PHASE 5 COMPLETE!".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
"""
Unit tests for model architectures, loss formulations, and diagnostic metrics.
"""

import unittest
from src.training.metrics import evaluate_diagnostic_metrics, calculate_confusion_matrix
from src.dataset.multimodal_loader import MultiModalCottonDataset
from src.models.multimodal_fusion import build_model


class TestModelsAndMetrics(unittest.TestCase):
    def test_metrics_computation(self):
        # 100 samples with 94 correct predictions
        y_true = [0] * 50 + [1] * 22 + [2] * 18 + [3] * 10
        y_pred = list(y_true)
        # Introduce 6 errors
        y_pred[0] = 1
        y_pred[1] = 1
        y_pred[51] = 0
        y_pred[52] = 2
        y_pred[73] = 1
        y_pred[91] = 2

        metrics = evaluate_diagnostic_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["overall_accuracy_pct"], 94.0, places=1)
        self.assertIn("Early-Stage Wilt (Pre-Visual)", metrics["class_breakdown"])

    def test_confusion_matrix(self):
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        cm = calculate_confusion_matrix(y_true, y_pred, num_classes=4)
        for i in range(4):
            self.assertEqual(cm[i][i], 1)

    def test_model_builder_interface(self):
        for m_type in ["cnn_lstm", "transformer", "gated_fusion"]:
            model = build_model(m_type)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()

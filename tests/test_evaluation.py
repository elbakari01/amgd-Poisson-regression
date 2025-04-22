import unittest
import numpy as np
from src.evaluation import evaluate_model

class TestEvaluation(unittest.TestCase):
    def test_perfect_prediction(self):
        # Perfect prediction: y_pred == y
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        beta = np.array([0, 1])  # Linear model: y = x
        metrics = evaluate_model(beta, X, y)
        self.assertEqual(metrics['MAE'], 0)  # MAE should be zero
        self.assertEqual(metrics['RMSE'], 0)  # RMSE should be zero

if __name__ == "__main__":
    unittest.main() 

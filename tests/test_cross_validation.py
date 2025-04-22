import unittest
import numpy as np
from src.cross_validation import k_fold_cross_validation

class TestCrossValidation(unittest.TestCase):
    def test_cv_results(self):
        # Simple dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])
        best_params, cv_results = k_fold_cross_validation(X, y, k=3, lambda_values=[0.1, 1.0])
        self.assertIn('AMGD', best_params)  # Ensure AMGD results are present
        self.assertTrue(len(cv_results) > 0)  # Ensure results are generated

if __name__ == "__main__":
    unittest.main() 
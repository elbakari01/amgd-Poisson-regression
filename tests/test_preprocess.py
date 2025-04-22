import unittest
import numpy as np
import pandas as pd
from src.preprocess import preprocess_ecological_dataset

class TestPreprocessing(unittest.TestCase):
    def test_preprocess(self):
        X, y, feature_names = preprocess_ecological_dataset("data/ecological_health_dataset.csv")
        self.assertEqual(X.shape[0], y.shape[0])  # Rows in X match rows in y
        self.assertTrue(isinstance(feature_names, list))  # Feature names are returned
        self.assertTrue(np.all(np.isfinite(X)))  # No NaN or infinite values in X
        self.assertTrue(np.all(np.isfinite(y)))  # No NaN or infinite values in y

if __name__ == "__main__":
    unittest.main() 
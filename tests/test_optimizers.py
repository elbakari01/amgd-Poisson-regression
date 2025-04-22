import unittest
import numpy as np
from src.optimizers import amgd

class TestAMGD(unittest.TestCase):
    def test_amgd_convergence(self):
        # Simple linear regression problem: y = 2x + 1
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([3, 5, 7, 9, 11])
        beta, _, _, _ = amgd(X, y, alpha=0.01, max_iter=1000)
        self.assertAlmostEqual(beta[0], 2.0, places=2)  # Slope should be close to 2
        self.assertAlmostEqual(beta[1], 1.0, places=2)  # Intercept should be close to 1

if __name__ == "__main__":
    unittest.main() 
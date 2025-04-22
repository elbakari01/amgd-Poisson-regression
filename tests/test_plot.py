import unittest
import matplotlib.pyplot as plt
from src.plots import compare_convergence_rates

class TestPlots(unittest.TestCase):
    def test_convergence_plot(self):
        # Simple dataset
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])
        try:
            fig = compare_convergence_rates(X, y, {})
            self.assertIsInstance(fig, plt.Figure)  # Ensure a valid figure is returned
        except Exception as e:
            self.fail(f"Plot generation failed: {e}")

if __name__ == "__main__":
    unittest.main() 
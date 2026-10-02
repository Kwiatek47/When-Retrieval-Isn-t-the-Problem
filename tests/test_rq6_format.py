import unittest

import numpy as np

from scripts.agents.analyze_rq6_format import cv_auroc, fit_logistic


class RQ6Test(unittest.TestCase):
    def test_logistic_recovers_signal_and_cv_auroc_is_chance_on_noise(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 3))
        y = (rng.random(600) < 1 / (1 + np.exp(-2 * x[:, 0]))).astype(float)
        w = fit_logistic(x, y)
        self.assertGreater(w[1], 1.2)
        self.assertLess(abs(w[2]), 0.5)
        self.assertGreater(cv_auroc(x, y, rng, repeats=2), 0.75)
        self.assertLess(abs(cv_auroc(x, rng.integers(0, 2, 600).astype(float), rng, repeats=2) - 0.5), 0.08)


if __name__ == "__main__":
    unittest.main()

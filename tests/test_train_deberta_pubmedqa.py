from __future__ import annotations

import ast
import inspect
import unittest

from scripts.classifier import train_deberta_pubmedqa


class TrainDebertaPubMedQATests(unittest.TestCase):
    def test_temperature_calibration_drops_aux_bow_before_model_forward(self) -> None:
        source = inspect.getsource(train_deberta_pubmedqa._fit_temperature)
        tree = ast.parse(source)

        drops_aux_bow = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "pop":
                continue
            if not node.args:
                continue
            key = node.args[0]
            if isinstance(key, ast.Constant) and key.value == "aux_bow":
                drops_aux_bow = True
                break

        self.assertTrue(drops_aux_bow)


if __name__ == "__main__":
    unittest.main()

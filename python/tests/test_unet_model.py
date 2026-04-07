from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from models.unet import NUM_SEMANTIC_CLASSES, SmallUNet


class SmallUNetTest(unittest.TestCase):
    def test_default_model_outputs_semantic_logits(self) -> None:
        model = SmallUNet()
        x = torch.randn(2, 1, 128, 128)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2, NUM_SEMANTIC_CLASSES, 128, 128))

    def test_model_supports_backward_pass(self) -> None:
        model = SmallUNet()
        x = torch.randn(2, 1, 128, 128)
        target = torch.randint(0, NUM_SEMANTIC_CLASSES, (2, 128, 128))

        logits = model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        has_grad = any(parameter.grad is not None for parameter in model.parameters())
        self.assertTrue(has_grad)

    def test_model_can_be_reconfigured_for_channel_counts(self) -> None:
        model = SmallUNet(in_channels=3, num_classes=7, base_channels=16)
        x = torch.randn(1, 3, 64, 64)

        y = model(x)

        self.assertEqual(tuple(y.shape), (1, 7, 64, 64))


if __name__ == "__main__":
    unittest.main()

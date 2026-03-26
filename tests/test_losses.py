"""
손실 함수 단위 테스트.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestLossFunctions:
    def test_classification_loss_function(self):
        """BCEWithLogitsLoss 기본 동작."""
        from mga.training.losses import get_loss_function
        loss_fn = get_loss_function("classification")
        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4, 2)).float()
        loss = loss_fn(logits, labels)
        assert loss.shape == (4, 2)
        assert not torch.isnan(loss).any()

    def test_regression_loss_function(self):
        """L1Loss 기본 동작."""
        from mga.training.losses import get_loss_function
        loss_fn = get_loss_function("regression")
        logits = torch.randn(4, 2)
        labels = torch.randn(4, 2)
        loss = loss_fn(logits, labels)
        assert loss.shape == (4, 2)
        assert not torch.isnan(loss).any()

    def test_classification_with_pos_weight(self):
        """pos_weight가 전달되면 BCEWithLogitsLoss에 반영."""
        from mga.training.losses import get_loss_function
        pos_weight = torch.tensor([2.0, 3.0])
        loss_fn = get_loss_function("classification", pos_weight=pos_weight)
        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4, 2)).float()
        loss = loss_fn(logits, labels)
        assert loss.shape == (4, 2)


class TestMaskedLoss:
    def test_full_mask(self):
        """mask=1 (모든 레이블 유효): 일반 loss와 동일."""
        from mga.training.losses import compute_masked_loss, get_loss_function
        loss_fn = get_loss_function("regression")
        logits = torch.ones(4, 2)
        labels = torch.zeros(4, 2)
        mask = torch.ones(4, 2)
        loss = compute_masked_loss(loss_fn, logits, labels, mask)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_masked_loss_ignores_missing(self):
        """mask=0 위치는 손실에 기여하지 않아야 함."""
        from mga.training.losses import compute_masked_loss, get_loss_function
        loss_fn = get_loss_function("regression")
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        labels = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        # 첫 번째 컬럼만 유효
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        loss = compute_masked_loss(loss_fn, logits, labels, mask)
        # 두 번째 컬럼(10.0 오차)은 무시되어야 함
        assert loss.item() > 0  # 첫 번째 컬럼 오차는 있음
        # 전체 마스크인 경우 손실이 더 커야 함
        mask_full = torch.ones(2, 2)
        loss_full = compute_masked_loss(loss_fn, logits, labels, mask_full)
        assert loss_full.item() > loss.item()

    def test_zero_mask_returns_zero_loss(self):
        """모든 mask=0이면 loss=0."""
        from mga.training.losses import compute_masked_loss, get_loss_function
        loss_fn = get_loss_function("regression")
        logits = torch.tensor([[999.0, 999.0]])
        labels = torch.tensor([[0.0, 0.0]])
        mask = torch.zeros(1, 2)
        loss = compute_masked_loss(loss_fn, logits, labels, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestPosWeight:
    def test_balanced_data_weight_near_one(self):
        """균형 잡힌 데이터에서 pos_weight ≈ 1."""
        import pandas as pd
        from mga.data.dataset import build_dataset
        from mga.training.losses import compute_pos_weight

        # 50:50 균형 데이터
        df = pd.DataFrame({
            "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "C1CCCCC1"],
            "task": [0, 1, 0, 1],
            "group": ["training"] * 4,
        })
        dataset = build_dataset(df, task_list=["task"])
        four_elem = [(m[0], m[1], m[2], m[3]) for m in dataset]
        weights = compute_pos_weight(four_elem, n_classification_tasks=1)
        assert weights.shape == (1,)
        assert weights[0].item() == pytest.approx(1.0, abs=0.1)

    def test_imbalanced_data_weight_greater_than_one(self):
        """불균형 데이터(neg >> pos)에서 pos_weight > 1."""
        import pandas as pd
        from mga.data.dataset import build_dataset
        from mga.training.losses import compute_pos_weight

        df = pd.DataFrame({
            "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "C1CCCCC1",
                       "CC(N)C(=O)O", "O=C(O)c1ccccc1"],
            "task": [0, 0, 0, 0, 0, 1],  # 5 neg, 1 pos
            "group": ["training"] * 6,
        })
        dataset = build_dataset(df, task_list=["task"])
        four_elem = [(m[0], m[1], m[2], m[3]) for m in dataset]
        weights = compute_pos_weight(four_elem, n_classification_tasks=1)
        assert weights[0].item() > 1.0

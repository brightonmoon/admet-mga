"""
TransferLearningManager 단위 테스트.

CPU + 소규모 모델로 실행.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def pretrained_checkpoint(tmp_path, mini_model):
    """미니 모델 가중치를 저장한 임시 체크포인트."""
    ckpt_path = tmp_path / "pretrained.pth"
    torch.save({"model_state_dict": mini_model.state_dict()}, ckpt_path)
    return ckpt_path


class TestTransferSetup:
    def test_setup_selective_layer_freezes_layer0(self, mini_model, pretrained_checkpoint):
        """selective_layer + freeze_layers=[0] → 첫 번째 RGCN 레이어 프리징."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="selective_layer",
            freeze_layers=[0],
        )
        manager.setup(mini_model)

        frozen_names = [n for n, p in mini_model.named_parameters() if not p.requires_grad]
        assert any("gnn_layers.0" in n or "rgcn_layer1" in n for n in frozen_names), \
            f"Layer 0 should be frozen, but frozen params: {frozen_names}"

    def test_setup_feature_extraction_freezes_all_encoder(self, mini_model, pretrained_checkpoint):
        """feature_extraction → 전체 인코더 프리징."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="feature_extraction",
        )
        manager.setup(mini_model)

        for name, param in mini_model.named_parameters():
            if manager._is_encoder_param(name):
                assert not param.requires_grad, f"Encoder param {name} should be frozen"

    def test_setup_full_finetune_no_freeze(self, mini_model, pretrained_checkpoint):
        """full_finetune → 모든 파라미터 학습 가능."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="full_finetune",
        )
        manager.setup(mini_model)

        for name, param in mini_model.named_parameters():
            assert param.requires_grad, f"Param {name} should not be frozen in full_finetune"

    def test_setup_returns_loaded_count(self, mini_model, pretrained_checkpoint):
        """setup() → 로드된 파라미터 수 반환."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="full_finetune",
        )
        count = manager.setup(mini_model)
        assert count > 0


class TestParameterGroups:
    def test_get_parameter_groups_two_groups(self, mini_model, pretrained_checkpoint):
        """차등 LR: 인코더/헤드 2개 그룹 반환."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="full_finetune",
            encoder_lr_multiplier=0.1,
        )
        manager.setup(mini_model)
        groups = manager.get_parameter_groups(mini_model, base_lr=0.001)

        assert len(groups) == 2
        encoder_group = next(g for g in groups if g["name"] == "encoder")
        head_group = next(g for g in groups if g["name"] == "head")
        assert encoder_group["lr"] == pytest.approx(0.0001, rel=1e-5)
        assert head_group["lr"] == pytest.approx(0.001, rel=1e-5)

    def test_frozen_params_excluded_from_groups(self, mini_model, pretrained_checkpoint):
        """프리징된 파라미터는 optimizer 그룹에서 제외."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="feature_extraction",
        )
        manager.setup(mini_model)
        groups = manager.get_parameter_groups(mini_model, base_lr=0.001)

        # feature_extraction: encoder frozen, so only head group
        assert len(groups) == 1
        assert groups[0]["name"] == "head"


class TestUnfreeze:
    def test_unfreeze_all_restores_requires_grad(self, mini_model, pretrained_checkpoint):
        """unfreeze_all() → 모든 파라미터 requires_grad=True."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="feature_extraction",
        )
        manager.setup(mini_model)

        # 일부 파라미터가 프리징되어 있어야 함
        frozen_before = [n for n, p in mini_model.named_parameters() if not p.requires_grad]
        assert len(frozen_before) > 0

        manager.unfreeze_all(mini_model)

        for name, param in mini_model.named_parameters():
            assert param.requires_grad, f"{name} should be unfrozen"

    def test_maybe_unfreeze_triggers_at_epoch(self, mini_model, pretrained_checkpoint):
        """unfreeze_epoch=5에서 epoch=5일 때 언프리징."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="selective_layer",
            freeze_layers=[0],
            unfreeze_epoch=5,
        )
        manager.setup(mini_model)

        assert not manager.maybe_unfreeze(mini_model, epoch=4)  # epoch 4: 아직 X
        assert manager.maybe_unfreeze(mini_model, epoch=5)      # epoch 5: 언프리징

    def test_maybe_unfreeze_no_epoch_set(self, mini_model, pretrained_checkpoint):
        """unfreeze_epoch=None이면 절대 언프리징하지 않음."""
        from mga.training.transfer import TransferLearningManager

        manager = TransferLearningManager(
            pretrained_path=pretrained_checkpoint,
            strategy="selective_layer",
            freeze_layers=[0],
            unfreeze_epoch=None,
        )
        manager.setup(mini_model)

        for epoch in range(100):
            assert not manager.maybe_unfreeze(mini_model, epoch)

"""
학습 파이프라인 단위 테스트.

CPU + 소규모 모델 (hidden=[16,16], n_tasks=2, 2 epochs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def trainer_with_loaders(mini_config, mini_model, mini_loaders, tmp_path):
    """MGATrainer + mini DataLoaders (CPU)."""
    from mga.training.trainer import MGATrainer

    mini_config.paths.model_dir = tmp_path / "models"
    train_loader, val_loader, test_loader = mini_loaders

    trainer = MGATrainer(
        model=mini_model,
        config=mini_config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    return trainer, train_loader, val_loader, test_loader


class TestMGATrainerInit:
    def test_trainer_creates_optimizer(self, trainer_with_loaders):
        """MGATrainer 초기화 후 optimizer가 생성됨."""
        trainer, *_ = trainer_with_loaders
        assert trainer.optimizer is not None

    def test_trainer_creates_stopper(self, trainer_with_loaders):
        """MGATrainer 초기화 후 EarlyStopping이 설정됨."""
        trainer, *_ = trainer_with_loaders
        assert trainer.stopper is not None
        assert trainer.stopper.patience == 10

    def test_trainer_device_is_cpu(self, trainer_with_loaders):
        """device='cpu'로 모델이 이동됨."""
        trainer, *_ = trainer_with_loaders
        assert trainer.device == "cpu"
        for p in trainer.model.parameters():
            assert p.device.type == "cpu"

    def test_no_transfer_manager_by_default(self, trainer_with_loaders):
        """기본 설정에서 transfer_manager=None."""
        trainer, *_ = trainer_with_loaders
        assert trainer.transfer_manager is None


class TestTrainEpoch:
    def test_train_epoch_returns_scores_and_loss(self, trainer_with_loaders):
        """train_epoch() → (scores_list, avg_loss) 반환."""
        trainer, *_ = trainer_with_loaders
        trainer.config.training.wandb.enabled = False
        scores, loss = trainer.train_epoch(epoch=0)
        assert isinstance(scores, list)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_epoch_scores_length(self, trainer_with_loaders):
        """scores 길이 = n_tasks."""
        trainer, *_ = trainer_with_loaders
        trainer.config.training.wandb.enabled = False
        scores, _ = trainer.train_epoch(epoch=0)
        assert len(scores) == 2  # n_tasks = 2

    def test_train_epoch_updates_parameters(self, trainer_with_loaders):
        """한 epoch 후 파라미터가 변경됨."""
        trainer, *_ = trainer_with_loaders
        trainer.config.training.wandb.enabled = False

        params_before = [p.clone() for p in trainer.model.parameters()]
        trainer.train_epoch(epoch=0)
        params_after = list(trainer.model.parameters())

        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        assert changed, "Parameters should change after training epoch"


class TestEvaluate:
    def test_evaluate_returns_scores(self, trainer_with_loaders):
        """evaluate() → scores_list 반환."""
        trainer, _, val_loader, _ = trainer_with_loaders
        scores = trainer.evaluate(val_loader)
        assert isinstance(scores, list)
        assert len(scores) == 2  # n_tasks

    def test_evaluate_all_metrics(self, trainer_with_loaders):
        """evaluate(return_all_metrics=True) → dict 반환."""
        trainer, _, val_loader, _ = trainer_with_loaders
        metrics = trainer.evaluate(val_loader, return_all_metrics=True)
        assert isinstance(metrics, dict)
        assert "primary_scores" in metrics
        assert "classification" in metrics

    def test_evaluate_does_not_update_params(self, trainer_with_loaders):
        """evaluate() → 파라미터 변경 없음 (no_grad)."""
        trainer, _, val_loader, _ = trainer_with_loaders
        params_before = [p.clone() for p in trainer.model.parameters()]
        trainer.evaluate(val_loader)
        params_after = list(trainer.model.parameters())
        for before, after in zip(params_before, params_after):
            assert torch.equal(before, after)


class TestFullTrainLoop:
    def test_train_full_loop_returns_history(self, trainer_with_loaders):
        """2-epoch 학습 완료 → history dict 반환."""
        trainer, *_ = trainer_with_loaders
        trainer.config.training.wandb.enabled = False
        history = trainer.train()
        assert isinstance(history, dict)
        assert "train_scores" in history
        assert "val_scores" in history
        assert "train_losses" in history

    def test_train_history_length(self, trainer_with_loaders):
        """history 길이 = 실행된 epoch 수."""
        trainer, *_ = trainer_with_loaders
        trainer.config.training.wandb.enabled = False
        history = trainer.train()
        # 2 epochs max; early stopping may trigger earlier
        assert len(history["train_losses"]) <= 2

    def test_train_best_score_set(self, trainer_with_loaders):
        """학습 후 stopper.best_score가 설정됨."""
        trainer, *_ = trainer_with_loaders
        trainer.config.training.wandb.enabled = False
        trainer.train()
        assert trainer.stopper.best_score is not None


class TestEarlyStopping:
    def test_early_stopping_triggers(self, tmp_path, mini_config, mini_model, mini_loaders):
        """patience=1이면 두 번째 epoch에서 조기 종료."""
        from mga.training.trainer import MGATrainer

        mini_config.training.patience = 1
        mini_config.training.num_epochs = 10
        mini_config.training.wandb.enabled = False
        mini_config.paths.model_dir = tmp_path / "models"

        train_loader, val_loader, test_loader = mini_loaders
        trainer = MGATrainer(mini_model, mini_config, train_loader, val_loader, test_loader)

        history = trainer.train()
        assert len(history["train_losses"]) < 10  # 조기 종료로 10 epoch 미만


class TestPosWeight:
    def test_use_pos_weight_creates_weighted_loss(
        self, tmp_path, mini_config, mini_model, mini_loaders
    ):
        """use_pos_weight=True → pos_weight가 loss_fn에 설정됨."""
        from mga.training.trainer import MGATrainer

        mini_config.training.use_pos_weight = True
        mini_config.paths.model_dir = tmp_path / "models"
        train_loader, val_loader, _ = mini_loaders

        trainer = MGATrainer(mini_model, mini_config, train_loader, val_loader)
        # pos_weight가 있으면 BCEWithLogitsLoss의 pos_weight가 None이 아님
        assert trainer.loss_fn_classification.pos_weight is not None

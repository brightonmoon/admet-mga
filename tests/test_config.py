"""
MGAConfig Pydantic 설정 단위 테스트.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


class TestMGAConfigFromDict:
    def test_legacy_flat_format(self):
        """flat dict (config_admet.yaml 형식) → MGAConfig 정상 변환."""
        from mga.config import MGAConfig

        data = {
            "task_name": "admet_52",
            "task_class": "classification",
            "n_tasks": 52,
            "batch_size": 512,
            "lr": 0.001,
            "num_epochs": 100,
            "device": "cpu",
            "rgcn_hidden_feats": [64, 64],
            "classification_list": ["Caco2_Wang", "HIA_Hou"],
            "regression_list": [],
        }
        config = MGAConfig.from_dict(data)
        assert config.task.task_name == "admet_52"
        assert config.model.n_tasks == 52
        assert config.training.batch_size == 512
        assert config.training.lr == 0.001

    def test_nested_format(self):
        """중첩 dict (model:, training:, task:) 형식 지원."""
        from mga.config import MGAConfig

        data = {
            "model": {"n_tasks": 10, "rgcn_hidden_feats": [32, 32]},
            "training": {"batch_size": 128, "lr": 0.0001, "device": "cpu"},
            "task": {"task_name": "test", "task_class": "regression"},
        }
        config = MGAConfig.from_dict(data)
        assert config.model.n_tasks == 10
        assert config.training.batch_size == 128
        assert config.task.task_class == "regression"

    def test_default_task_name(self):
        """task_name 없으면 'mga-default' 기본값."""
        from mga.config import MGAConfig
        config = MGAConfig.from_dict({})
        assert config.task.task_name == "mga-default"

    def test_transfer_config_from_dict(self, tmp_path):
        """transfer 키가 있으면 TransferConfig로 변환."""
        from mga.config import MGAConfig

        ckpt = tmp_path / "model.pth"
        ckpt.touch()

        data = {
            "task_name": "test",
            "transfer": {
                "strategy": "full_finetune",
                "pretrained_model_path": str(ckpt),
                "encoder_lr_multiplier": 0.05,
            },
        }
        config = MGAConfig.from_dict(data)
        assert config.transfer is not None
        assert config.transfer.strategy == "full_finetune"
        assert config.transfer.encoder_lr_multiplier == 0.05

    def test_bin_path_group_path_mapped(self):
        """레거시 bin_path, group_path가 PathConfig에 매핑."""
        from mga.config import MGAConfig

        data = {
            "task_name": "test",
            "bin_path": "/tmp/data.bin",
            "group_path": "/tmp/data_group.csv",
        }
        config = MGAConfig.from_dict(data)
        assert config.paths.bin_path == Path("/tmp/data.bin")
        assert config.paths.group_path == Path("/tmp/data_group.csv")

    def test_use_pos_weight_default_false(self):
        """use_pos_weight 기본값은 False."""
        from mga.config import MGAConfig
        config = MGAConfig.from_dict({})
        assert config.training.use_pos_weight is False

    def test_use_pos_weight_from_dict(self):
        """use_pos_weight=True가 올바르게 설정."""
        from mga.config import MGAConfig
        config = MGAConfig.from_dict({"use_pos_weight": True})
        assert config.training.use_pos_weight is True


class TestMGAConfigYAML:
    def test_from_yaml_roundtrip(self, tmp_path, mini_config):
        """from_yaml → to_yaml → from_yaml: 값 보존."""
        from mga.config import MGAConfig

        yaml_path = tmp_path / "config.yaml"
        mini_config.to_yaml(yaml_path)
        loaded = MGAConfig.from_yaml(yaml_path)

        assert loaded.task.task_name == mini_config.task.task_name
        assert loaded.training.batch_size == mini_config.training.batch_size
        assert loaded.model.n_tasks == mini_config.model.n_tasks

    def test_from_yaml_missing_file(self, tmp_path):
        """존재하지 않는 파일 → FileNotFoundError."""
        from mga.config import MGAConfig
        with pytest.raises(FileNotFoundError):
            MGAConfig.from_yaml(tmp_path / "nonexistent.yaml")


class TestMGAConfigValidation:
    def test_invalid_device(self):
        """잘못된 device 값 → ValidationError."""
        from pydantic import ValidationError
        from mga.config.config import TrainingConfig
        with pytest.raises(ValidationError):
            TrainingConfig(device="tpu")

    def test_invalid_dropout_range(self):
        """dropout > 1.0 → ValidationError."""
        from pydantic import ValidationError
        from mga.config.config import ModelConfig
        with pytest.raises(ValidationError):
            ModelConfig(rgcn_drop_out=1.5)


class TestWandbConfig:
    def test_to_wandb_config_has_expected_keys(self, mini_config):
        """to_wandb_config()가 model/training/task 키를 포함."""
        wc = mini_config.to_wandb_config()
        assert "n_tasks" in wc
        assert "batch_size" in wc
        assert "task_name" in wc
        assert "wandb" not in wc  # wandb 설정 자체는 제외

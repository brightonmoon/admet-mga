"""
체크포인트 호환성 유틸리티.

mga_inference에서 저장된 MGAChemiverse 체크포인트에는 미사용 'gates.*' 키가 포함되어 있음.
MGATest로 로드할 때 해당 키를 제거하여 strict=True 로드가 가능하도록 처리.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_checkpoint_compat(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
) -> None:
    """
    체크포인트를 모델에 로드. MGAChemiverse에서 저장된 경우 불필요한 키를 제거 후 적용.

    Args:
        model: 대상 모델 (MGATest 등)
        checkpoint_path: .pth 파일 경로
        device: 로드할 디바이스 (None이면 자동 감지)
    """
    map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: 체크포인트에 커스텀 객체(optimizer state 등)가 포함될 수 있음
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # 체크포인트 포맷 처리
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # MGAChemiverse의 미사용 'gates.*' 키 제거
    cleaned = {k: v for k, v in state_dict.items() if not k.startswith("gates.")}

    # 모델 키와 체크포인트 키 불일치 확인 후 로드
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(cleaned.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        raise RuntimeError(
            f"Missing keys in checkpoint '{checkpoint_path}':\n{sorted(missing)}"
        )
    if unexpected:
        logger.warning(
            "Ignoring unexpected keys in '%s': %s",
            Path(checkpoint_path).name,
            sorted(unexpected),
        )
        model.load_state_dict(cleaned, strict=False)
    else:
        model.load_state_dict(cleaned, strict=True)

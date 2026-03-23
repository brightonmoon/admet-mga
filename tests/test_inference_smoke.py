"""
ADMETPredictor 스모크 테스트.

실제 모델 체크포인트가 있어야 실행 가능. 없으면 스킵.

실행:
    pytest tests/test_inference_smoke.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"

pytestmark = pytest.mark.skipif(
    not CHECKPOINTS_DIR.exists(),
    reason="checkpoints/ 디렉토리 없음 - 모델 파일 필요"
)

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"


@pytest.fixture(scope="module")
def predictor():
    from mga.inference import ADMETPredictor
    return ADMETPredictor(checkpoints_dir=CHECKPOINTS_DIR)


def test_predict_single_returns_list(predictor):
    result = predictor.predict_single(ASPIRIN_SMILES)
    assert isinstance(result, list)
    assert len(result) == 1


def test_predict_single_has_smiles_key(predictor):
    result = predictor.predict_single(ASPIRIN_SMILES)
    assert "SMILES" in result[0]
    assert result[0]["SMILES"] == ASPIRIN_SMILES


def test_predict_single_has_predictions(predictor):
    result = predictor.predict_single(ASPIRIN_SMILES)
    preds = result[0]["Predict"]
    assert isinstance(preds, list)
    assert len(preds) > 0


def test_predict_single_prediction_fields(predictor):
    result = predictor.predict_single(ASPIRIN_SMILES)
    pred = result[0]["Predict"][0]
    assert "task" in pred
    assert "category" in pred
    assert "value" in pred
    assert "threshold" in pred
    assert isinstance(pred["value"], float)


def test_predict_single_value_range(predictor):
    """분류 태스크 예측값은 0~100 범위여야 함."""
    result = predictor.predict_single(ASPIRIN_SMILES)
    for pred in result[0]["Predict"]:
        if pred["threshold"] != "regression":
            assert 0.0 <= pred["value"] <= 100.0, (
                f"{pred['task']} value out of range: {pred['value']}"
            )


def test_predict_batch_multiple_smiles(predictor):
    smiles_list = [ASPIRIN_SMILES, "CCO", "c1ccccc1"]
    import pandas as pd
    df = predictor.predict_batch(smiles_list)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_predict_invalid_smiles(predictor):
    """유효하지 않은 SMILES는 결과에서 제외되거나 에러 없이 처리되어야 함."""
    result = predictor.predict_single("INVALID_SMILES_XYZ")
    # 빈 리스트이거나, 에러 없이 반환되어야 함
    assert isinstance(result, list)


def test_predict_single_with_base64_image(predictor):
    result = predictor.predict_single(ASPIRIN_SMILES, image_mode="base64")
    preds = result[0]["Predict"]
    # sub_structure 필드가 있어야 함 (base64 문자열 또는 빈 문자열)
    for pred in preds:
        assert "sub_structure" in pred

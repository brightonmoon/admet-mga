"""
PredictionFormatter 단위 테스트.

의존성 없이 순수 Python으로 실행 가능.
"""

import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mga.inference.formatter import PredictionFormatter


def make_formatter():
    f = PredictionFormatter()
    f.add_prediction("CC", "caco2_reg", "absorption", "regression", -5.2, "", "cm/s")
    f.add_prediction("CC", "pgp_substrate", "absorption", "classification", 70.0, "svg_b64", "%")
    f.add_prediction("CCO", "caco2_reg", "absorption", "regression", -4.1, "", "cm/s")
    return f


def test_single_smiles_entry():
    f = make_formatter()
    result = f.get_result()
    assert len(result) == 2, "CC and CCO should be two entries"
    cc_entry = next(e for e in result if e["SMILES"] == "CC")
    assert len(cc_entry["Predict"]) == 2


def test_duplicate_smiles_not_added():
    f = PredictionFormatter()
    f.add_prediction("CC", "task1", "cat", "classification", 60.0, "", "%")
    f.add_prediction("CC", "task2", "cat", "classification", 40.0, "", "%")
    result = f.get_result()
    assert len(result) == 1
    assert len(result[0]["Predict"]) == 2


def test_threshold_classification():
    f = PredictionFormatter()
    f.add_prediction("CC", "herg", "toxicity", "classification", 80.0, "", "%")
    f.add_prediction("CC", "ames", "toxicity", "classification", 30.0, "", "%")
    preds = {p["task"]: p for p in f.get_result()[0]["Predict"]}
    assert preds["herg"]["threshold"] == "herg"
    assert preds["ames"]["threshold"] == "non-ames"


def test_threshold_regression():
    f = PredictionFormatter()
    f.add_prediction("CC", "caco2_reg", "absorption", "regression", -5.0, "", "cm/s")
    pred = f.get_result()[0]["Predict"][0]
    assert pred["threshold"] == "regression"


def test_save_to_json_roundtrip():
    f = make_formatter()
    buf = io.StringIO()
    data = json.dumps(f.get_result(), indent=4)
    reloaded = json.loads(data)
    assert len(reloaded) == 2
    assert reloaded[0]["SMILES"] == "CC"


def test_save_to_csv_roundtrip(tmp_path):
    f = make_formatter()
    csv_path = tmp_path / "out.csv"
    f.save_to_csv(str(csv_path))
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    smiles_col = {r["SMILES"] for r in rows}
    assert "CC" in smiles_col and "CCO" in smiles_col


def test_empty_formatter():
    f = PredictionFormatter()
    assert f.get_result() == []
    # save_to_csv on empty should not raise
    f.save_to_csv("/dev/null")

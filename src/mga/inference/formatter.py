"""
PredictionFormatter - 예측 결과를 구조화된 형식으로 정리.
"""

import csv
import json


class PredictionFormatter:
    def __init__(self) -> None:
        self._index: dict[str, dict] = {}  # SMILES -> entry (O(1) lookup)
        self.result: list[dict] = []       # 순서 보존용 리스트

    def add_prediction(
        self,
        smiles: str,
        task_name: str,
        category_name: str,
        learning_task: str,
        predict_value: float,
        structure: str,
        unit: str,
    ) -> None:
        if learning_task == "classification":
            threshold = task_name if predict_value >= 50 else f"non-{task_name}"
        else:
            threshold = "regression"

        if smiles not in self._index:
            entry = {"SMILES": smiles, "Predict": []}
            self._index[smiles] = entry
            self.result.append(entry)

        self._index[smiles]["Predict"].append({
            "task": task_name,
            "category": category_name,
            "value": predict_value,
            "threshold": threshold,
            "sub_structure": structure,
            "unit": unit,
        })

    def get_result(self) -> list[dict]:
        return self.result

    def save_to_json(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            json.dump(self.result, f, indent=4)

    def save_to_csv(self, file_path: str) -> None:
        if not self.result:
            return
        # 첫 번째 엔트리의 태스크 순서를 헤더로 사용
        task_names = [pred["task"] for pred in self.result[0]["Predict"]]
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["SMILES"] + task_names)
            for entry in self.result:
                pred_map = {p["task"]: p["value"] for p in entry["Predict"]}
                row = [entry["SMILES"]] + [pred_map.get(t, "") for t in task_names]
                writer.writerow(row)

    def print_json(self) -> None:
        print(json.dumps(self.result, indent=4))

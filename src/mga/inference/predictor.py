"""
ADMETPredictor - 통합 예측 파이프라인.
"""

import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mga.data.collate import collate_molgraphs_inference
from mga.data.dataset import inference_build_dataset
from mga.inference.formatter import PredictionFormatter
from mga.inference.task_registry import (
    TASK_LISTS,
    TASK_PARAMS,
    TOXICITY_REG_TASKS,
    apply_postprocessing_batch,
    apply_postprocessing_single,
    get_model_paths,
    get_task_meta,
)
from mga.inference.visualization import ImageHandler, return_result, return_result_supgraph
from mga.models.mga import MGATest
from mga.utils.compat import load_checkpoint_compat
from mga.utils.seed import set_random_seed

# RDKit 등 외부 라이브러리의 알려진 경고만 억제
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")

_DEFAULT_CHECKPOINTS = Path(__file__).resolve().parents[3] / "checkpoints"


class ADMETPredictor:
    """
    ADMET 예측기.

    사용 예:
        predictor = ADMETPredictor()
        result = predictor.predict_single("CC(=O)OC1=CC=CC=C1C(=O)O")
        df = predictor.predict_batch(["SMILES1", "SMILES2"])
    """

    def __init__(
        self,
        checkpoints_dir: str | Path | None = None,
        device: str | None = None,
        seed: int = 42,
    ):
        """
        Args:
            checkpoints_dir: .pth 파일이 있는 디렉토리. None이면 프로젝트 루트의 checkpoints/ 사용.
            device: 'cuda' | 'cpu' | None (자동 감지)
            seed: 재현성을 위한 랜덤 시드
        """
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else _DEFAULT_CHECKPOINTS
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        set_random_seed(seed)
        self._models: dict | None = None
        self._models_return_weight: bool | None = None

    def _load_models(self, return_weight: bool) -> dict:
        """모델 로드. 동일 return_weight로 이미 로드된 경우 캐시 반환."""
        if self._models is not None and self._models_return_weight == return_weight:
            return self._models

        model_paths = get_model_paths(self.checkpoints_dir)
        models = {}
        for task, path in model_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            model = MGATest(
                in_feats=40,
                rgcn_hidden_feats=64,
                gnn_out_feats=64,
                n_tasks=len(TASK_LISTS[task]),
                rgcn_drop_out=0.2,
                classifier_hidden_feats=TASK_PARAMS[task]["hidden_feats"],
                dropout=TASK_PARAMS[task]["dropout"],
                loop=True,
                return_weight=return_weight,
            )
            load_checkpoint_compat(model, path, device=self.device)
            model.to(self.device)
            model.eval()
            models[task] = model

        self._models = models
        self._models_return_weight = return_weight
        return models

    def _build_dataloader(self, smiles_list: list[str], batch_size: int) -> DataLoader:
        df = pd.DataFrame({"SMILES": smiles_list})
        dataset = inference_build_dataset(df, smiles_col="SMILES")
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_molgraphs_inference)

    def predict_single(
        self,
        smiles: str | list[str],
        batch_size: int = 32,
        image_mode: str = "auto",
        output_dir: str | Path | None = None,
    ) -> list[dict]:
        """
        단일/소량 처리 - 이미지 포함 JSON 결과 반환.

        Args:
            smiles: SMILES 문자열 또는 리스트
            batch_size: DataLoader 배치 크기
            image_mode: 'base64' | 'file' | 'auto'
            output_dir: file 모드 시 이미지 저장 디렉토리

        Returns:
            [{"SMILES": ..., "Predict": [...]}, ...] 형태의 리스트
        """
        smiles_list = [smiles] if isinstance(smiles, str) else smiles
        models = self._load_models(return_weight=True)
        data_loader = self._build_dataloader(smiles_list, batch_size)
        image_handler = ImageHandler(mode=image_mode, output_dir=output_dir or os.getcwd())
        formatter = PredictionFormatter()

        for batch_data in data_loader:
            with torch.inference_mode():
                smiles_batch, bg_batch = batch_data
                atom_feats = bg_batch.ndata.pop("atom").float().to(self.device)
                bond_feats = bg_batch.edata.pop("etype").long().to(self.device)
                bg_batch = bg_batch.to(self.device)

                for task, model in models.items():
                    result, images = return_result_supgraph(
                        model=model,
                        smiles=smiles_batch,
                        bg=bg_batch,
                        atom_feats=atom_feats,
                        bond_feats=bond_feats,
                        task_list=TASK_LISTS[task],
                    )
                    result = apply_postprocessing_single(result, task)

                    for smi, res, img in zip(smiles_batch, result.tolist(), images):
                        for task_name, val, structure_b64 in zip(TASK_LISTS[task], res, img):
                            if task == "toxicity" and task_name in TOXICITY_REG_TASKS:
                                continue
                            meta = get_task_meta(task, task_name)
                            structure = image_handler.process_image(structure_b64, smi, task, task_name)
                            formatter.add_prediction(
                                smiles=smi,
                                task_name=task_name,
                                category_name=meta["category"],
                                learning_task=meta["learning_task"],
                                predict_value=round(val * meta["scale"], 4),
                                structure=structure,
                                unit=meta["unit"],
                            )

        return formatter.get_result()

    def predict_batch(
        self,
        smiles_list: list[str],
        batch_size: int = 32,
        generate_images: bool = False,
        image_mode: str = "auto",
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        대용량 배치 처리 - CSV용 DataFrame 반환.

        Args:
            smiles_list: SMILES 문자열 목록
            batch_size: DataLoader 배치 크기
            generate_images: 이미지 생성 여부 (느려짐)
            image_mode: 'base64' | 'file' | 'auto'
            output_dir: file 모드 시 이미지 저장 디렉토리

        Returns:
            SMILES + 태스크별 예측값 DataFrame
        """
        models = self._load_models(return_weight=generate_images)
        data_loader = self._build_dataloader(smiles_list, batch_size)
        image_handler = (
            ImageHandler(mode=image_mode, output_dir=output_dir or os.getcwd())
            if generate_images else None
        )

        all_batches: list[pd.DataFrame] = []
        images_dict: dict = {}

        for batch_data in data_loader:
            with torch.inference_mode():
                smiles_batch, bg_batch = batch_data
                atom_feats = bg_batch.ndata.pop("atom").float().to(self.device)
                bond_feats = bg_batch.edata.pop("etype").long().to(self.device)
                bg_batch = bg_batch.to(self.device)

                batch_results: dict = {}
                for task, model in models.items():
                    if generate_images:
                        result, images = return_result_supgraph(
                            model=model,
                            smiles=smiles_batch,
                            bg=bg_batch,
                            atom_feats=atom_feats,
                            bond_feats=bond_feats,
                            task_list=TASK_LISTS[task],
                        )
                        for smi, img_list in zip(smiles_batch, images):
                            images_dict.setdefault(smi, {}).setdefault(task, {})
                            for task_name, b64 in zip(TASK_LISTS[task], img_list):
                                images_dict[smi][task][task_name] = b64
                        result = apply_postprocessing_single(result, task)
                    else:
                        result = return_result(model, bg_batch, atom_feats, bond_feats)
                        result = apply_postprocessing_batch(result, task)

                    result_np = result.cpu().numpy()
                    task_df = pd.DataFrame(result_np, columns=TASK_LISTS[task])
                    batch_results.update(task_df.to_dict("list"))

                all_batches.append(pd.DataFrame(batch_results))

        results_df = pd.concat(all_batches, ignore_index=True)

        if images_dict and image_handler and image_handler.mode == "file":
            for col_task, task_dict in next(iter(images_dict.values())).items():
                for task_name in task_dict:
                    col_name = f"{task_name}_image_path"
                    paths = [
                        image_handler.process_image(
                            images_dict.get(smi, {}).get(col_task, {}).get(task_name, ""),
                            smi, col_task, task_name,
                        ) if images_dict.get(smi, {}).get(col_task, {}).get(task_name) else ""
                        for smi in smiles_list[: len(results_df)]
                    ]
                    results_df[col_name] = paths

        results_df.insert(0, "SMILES", smiles_list[: len(results_df)])
        return results_df

    @staticmethod
    def parse_input(input_data: str | list) -> tuple[list[str], bool]:
        """
        다양한 입력 형식을 SMILES 리스트로 변환.

        Returns:
            (smiles_list, is_single)
        """
        if isinstance(input_data, list):
            return input_data, len(input_data) == 1

        if isinstance(input_data, str):
            if os.path.isfile(input_data):
                df = pd.read_csv(input_data)
                if "SMILES" not in df.columns:
                    raise ValueError("CSV must contain a 'SMILES' column")
                smiles_list = df["SMILES"].dropna().tolist()
                return smiles_list, len(smiles_list) == 1
            return [input_data], True

        raise ValueError(f"Unsupported input type: {type(input_data)}")

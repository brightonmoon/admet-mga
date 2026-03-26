"""
MGA 독립 평가 CLI entry point.

저장된 체크포인트를 데이터셋에서 평가하여 전체 메트릭을 출력합니다.

Usage:
    mga-eval --config config/config_admet.yaml --checkpoint checkpoints/train/best.pth \\
             --data-bin data/train/admet_52.bin --data-group data/train/admet_52_group.csv

    # JSON 저장
    mga-eval --config ... --checkpoint ... --data-bin ... --output results/eval.json

    # CSV 저장
    mga-eval --config ... --checkpoint ... --data-bin ... --output results/eval.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mga.config import load_config
from mga.data import load_graph_dataset, collate_molgraphs
from mga.models import MGA
from mga.utils.checkpoint import load_checkpoint
from mga.utils.logging import configure_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="MGA 모델 독립 평가 (체크포인트 → 메트릭)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="학습 설정 YAML 파일 경로",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="평가할 모델 체크포인트 (.pth) 경로",
    )
    parser.add_argument(
        "--data-bin", type=str, default=None,
        help="바이너리 그래프 데이터 경로 (미지정 시 config에서 추론)",
    )
    parser.add_argument(
        "--data-group", type=str, default=None,
        help="그룹 CSV 경로 (미지정 시 config에서 추론)",
    )
    parser.add_argument(
        "--split", type=str, choices=["test", "val", "train", "all"], default="test",
        help="평가할 데이터 분할 (default: test)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="결과 저장 경로 (.json 또는 .csv). 미지정 시 stdout",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="배치 크기 (default: 512)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda / cpu (default: 자동 감지)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="DEBUG 로깅 활성화",
    )
    return parser.parse_args()


def _get_loader(split: str, train_set, val_set, test_set, loader_kwargs):
    """split 이름에 따라 DataLoader 반환."""
    if split == "test":
        return DataLoader(test_set, shuffle=False, **loader_kwargs)
    elif split == "val":
        return DataLoader(val_set, shuffle=False, **loader_kwargs)
    elif split == "train":
        return DataLoader(train_set, shuffle=False, **loader_kwargs)
    else:  # all
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([train_set, val_set, test_set])
        return DataLoader(combined, shuffle=False, **loader_kwargs)


def _flatten_metrics(metrics: dict) -> dict:
    """중첩된 메트릭 dict을 단일 레벨로 펼침."""
    flat = {}
    for group, values in metrics.items():
        if group == "primary_scores":
            flat["primary_score_mean"] = float(np.mean(values))
            continue
        if isinstance(values, dict):
            for metric_name, scores in values.items():
                if isinstance(scores, list):
                    flat[f"{group}/{metric_name}_mean"] = float(np.mean(scores))
                    for i, s in enumerate(scores):
                        flat[f"{group}/{metric_name}_{i}"] = float(s)
                else:
                    flat[f"{group}/{metric_name}"] = float(scores)
    return flat


def main():
    args = parse_args()

    configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    logger = logging.getLogger("mga.cli.evaluate")

    # ── Config 로딩 ──────────────────────────────────────
    config = load_config(args.config)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 불가, CPU로 폴백")
        device = "cpu"
    config.training.device = device

    # ── 데이터 로딩 ──────────────────────────────────────
    data_bin = args.data_bin or str(config.paths.data_dir / "admet_52.bin")
    data_group = args.data_group or str(config.paths.data_dir / "admet_52_group.csv")

    logger.info(f"데이터 로딩: {data_bin}")
    train_set, val_set, test_set, n_tasks = load_graph_dataset(
        bin_path=data_bin,
        group_path=data_group,
        select_task_index=config.task.select_task_index,
    )
    config.model.n_tasks = n_tasks

    num_workers = min(config.training.num_workers, multiprocessing.cpu_count() or 1)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_molgraphs,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
    }

    eval_loader = _get_loader(args.split, train_set, val_set, test_set, loader_kwargs)
    logger.info(f"평가 대상: {args.split} ({len(eval_loader.dataset)} 분자)")

    # ── 모델 로딩 ──────────────────────────────────────
    model = MGA(
        in_feats=config.model.in_feats,
        rgcn_hidden_feats=config.model.rgcn_hidden_feats,
        n_tasks=config.model.n_tasks,
        classifier_hidden_feats=config.model.classifier_hidden_feats,
        loop=config.model.loop,
        rgcn_drop_out=config.model.rgcn_drop_out,
        dropout=config.model.drop_out,
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"체크포인트 없음: {checkpoint_path}")
        sys.exit(1)

    load_checkpoint(model, str(checkpoint_path), device=device)
    model.to(device)
    logger.info(f"체크포인트 로드: {checkpoint_path}")

    # ── 평가 ──────────────────────────────────────────
    from mga.training import MGATrainer

    # MGATrainer.evaluate()를 재사용 (dummy loaders for train/val)
    trainer = MGATrainer(
        model=model,
        config=config,
        train_loader=eval_loader,
        val_loader=eval_loader,
        test_loader=None,
    )

    logger.info("평가 중...")
    all_metrics = trainer.evaluate(eval_loader, return_all_metrics=True)
    flat = _flatten_metrics(all_metrics)

    # ── 출력 ──────────────────────────────────────────
    logger.info("─" * 50)
    logger.info(f"평가 결과 ({args.split} split)")
    logger.info("─" * 50)
    for k, v in flat.items():
        if "_mean" in k or "primary" in k:
            logger.info(f"  {k}: {v:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.output.lower().endswith(".csv"):
            import pandas as pd
            pd.DataFrame([flat]).to_csv(output_path, index=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(flat, f, indent=2, ensure_ascii=False)

        logger.info(f"결과 저장: {output_path}")
    else:
        print(json.dumps(flat, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

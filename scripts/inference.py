"""
MGA ADMET 추론 CLI.

ADMETPredictor를 사용하는 통합 스크립트.
단일 처리(이미지 포함 JSON) 및 배치 처리(CSV) 모두 지원.

Usage:
    # 단일 SMILES
    python scripts/inference.py --input "CC(=O)OC1=CC=CC=C1C(=O)O" --output result.json

    # CSV 배치
    python scripts/inference.py --input data/compounds.csv --output results.csv

    # Docker 배치 (이미지 파일 저장)
    python scripts/inference.py --input data.csv --mode batch --output out.csv --docker
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 패키지 설치 전 직접 실행 시 src/ 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mga.inference.predictor import ADMETPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="MGA ADMET 예측 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=str,
        help="SMILES 문자열 또는 CSV 파일 경로 (SMILES 컬럼 필요)",
    )
    input_group.add_argument(
        "--data", type=str,
        help="[Deprecated] --input 사용 권장",
    )

    parser.add_argument(
        "--mode", type=str, choices=["single", "batch"],
        help="처리 모드: single(이미지 포함 JSON) 또는 batch(CSV). 미지정 시 자동 감지",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="출력 파일 경로 (.json for single, .csv for batch)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="이미지 파일 저장 디렉토리 (file 모드 전용)",
    )
    parser.add_argument(
        "--image-mode", type=str, choices=["base64", "file", "auto"], default="auto",
        help="이미지 출력 방식 (default: auto - Docker 감지 시 file, 아니면 base64)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="DataLoader 배치 크기 (default: 32)",
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default=None,
        help=".pth 체크포인트 파일이 있는 디렉토리 (default: ./checkpoints)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda' 또는 'cpu' (default: 자동 감지)",
    )
    parser.add_argument(
        "--docker", action="store_true",
        help="Docker 모드 강제 활성화 (--image-mode file과 동일)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 하위 호환성
    if args.data and not args.input:
        args.input = args.data
    if args.docker:
        args.image_mode = "file"

    predictor = ADMETPredictor(
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
    )
    smiles_list, is_single = predictor.parse_input(args.input)

    # 모드 결정
    if args.mode == "single":
        is_single = True
    elif args.mode == "batch":
        is_single = False

    if is_single:
        result = predictor.predict_single(
            smiles=smiles_list,
            batch_size=args.batch_size,
            image_mode=args.image_mode,
            output_dir=args.output_dir,
        )
        if args.output:
            if not args.output.lower().endswith(".json"):
                raise ValueError("단일 처리 모드 출력은 .json 파일이어야 합니다.")
            with open(args.output, "w") as f:
                json.dump(result, f, indent=4)
            print(f"결과 저장: {args.output}")
        else:
            print(json.dumps(result, indent=4))
    else:
        df = predictor.predict_batch(
            smiles_list=smiles_list,
            batch_size=args.batch_size,
            generate_images=args.image_mode in ["base64", "file"],
            image_mode=args.image_mode,
            output_dir=args.output_dir,
        )
        if args.output:
            if not args.output.lower().endswith(".csv"):
                raise ValueError("배치 처리 모드 출력은 .csv 파일이어야 합니다.")
            df.to_csv(args.output, index=False)
            print(f"결과 저장: {args.output}")
            if args.output_dir:
                print(f"이미지 저장: {args.output_dir}/images/")
        else:
            print(df.to_string())


if __name__ == "__main__":
    main()

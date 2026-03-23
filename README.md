# MGA — Multi-task Graph Attention for ADMET Prediction

GNN 기반 ADMET(흡수·분포·대사·배설·독성) 특성 예측 프레임워크.
**52개 태스크**를 단일 모델로 동시 예측하며, 분자별 attention 시각화를 제공합니다.

**Stack**: PyTorch 2.4 · DGL · RDKit · Streamlit

---

## Features

- **52개 ADMET 태스크** — Absorption, Distribution, Metabolism, Excretion, Toxicity, Tox21
- **RGCN 기반 Multi-task Graph Attention** 아키텍처 (MGATest)
- **Substructure 시각화** — 분자별 atom attention 가중치 기반 강조 이미지
- **4가지 인터페이스** — Python API / CLI / Streamlit UI / Docker
- **Transfer learning** 지원 (TransferLearningManager)
- **Checkpoint 호환성** — 레거시 MGAChemiverse 체크포인트 자동 변환

---

## Project Structure

```
admet-mga/
├── pyproject.toml              # 패키지 설정 (mga v0.2.0)
├── src/mga/
│   ├── inference/              # 추론 파이프라인
│   │   ├── predictor.py        # ADMETPredictor (핵심 API)
│   │   ├── task_registry.py    # 모델 경로·태스크 목록 단일 소스
│   │   ├── formatter.py        # 결과 포맷팅 (JSON/CSV)
│   │   └── visualization.py   # Attention 시각화
│   ├── models/                 # MGA, MGATest 모델 정의
│   ├── data/                   # 그래프 피처링·데이터셋·콜레이션
│   ├── training/               # 학습기·콜백·전이학습
│   ├── config/                 # Pydantic 기반 설정
│   ├── metrics/                # 평가 지표
│   └── utils/                  # 체크포인트·시드·호환성
├── scripts/
│   ├── inference.py            # CLI 추론
│   ├── train.py                # CLI 학습
│   └── prepare_data.py         # 데이터 전처리
├── serve/
│   ├── app.py                  # Streamlit UI
│   ├── Dockerfile              # CUDA 12.1 + Miniconda 이미지
│   └── docker-compose.yml      # GPU 개발 컨테이너
├── config/                     # 학습 YAML 설정 파일
├── checkpoints/
│   └── inference/              # 추론용 .pth 체크포인트
├── data/                       # 바이너리 그래프 데이터
└── tests/                      # 단위·스모크 테스트
```

---

## Installation

### Prerequisites

| 항목 | 버전 |
|------|------|
| Python | 3.11 |
| CUDA | 12.1 (GPU 사용 시) |
| DGL | 별도 설치 필요 (아래 참조) |

### 1. 패키지 설치

```bash
# 추론 전용 (core)
pip install -e .

# 학습 환경 (wandb, optuna, pytdc 포함)
pip install -e ".[train]"

# Streamlit UI 포함
pip install -e ".[serve]"

# 전체 개발 환경
pip install -e ".[dev]"
```

### 2. DGL 설치 (별도)

DGL은 PyPI 기본 인덱스에서 설치할 수 없습니다. CUDA 버전에 맞는 방법을 사용하세요.

```bash
# conda (권장)
conda install -c dglteam/label/th24_cu121 dgl

# pip (CUDA 12.1)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```

### 3. Conda 환경 (서빙 전용)

Docker 없이 서빙 환경을 그대로 재현하려면 `serve/environment.yaml`을 사용하세요.

```bash
conda env create -f serve/environment.yaml
conda activate ADMET_v2
```

---

## Quick Start

### Python API

```python
from mga.inference import ADMETPredictor

predictor = ADMETPredictor()  # checkpoints/ 자동 탐색

# 단일 분자 예측 (결과 + substructure 이미지 포함)
result = predictor.predict_single("CC(=O)OC1=CC=CC=C1C(=O)O")
# result: [{"SMILES": "...", "Predict": [{"task": "Caco-2 Permeability", "value": -5.2, ...}, ...]}]

# 대량 배치 예측 (DataFrame 반환)
import pandas as pd
df = predictor.predict_batch(["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"])
print(df.head())
```

**사용자 정의 체크포인트 경로 지정**:
```python
predictor = ADMETPredictor(
    checkpoints_dir="path/to/checkpoints",
    device="cuda",  # 또는 "cpu"
    seed=42,
)
```

---

## CLI Usage

### 추론 (inference)

```bash
# 단일 분자 → JSON
python scripts/inference.py \
    --input "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --output result.json

# 배치 (CSV 파일) → CSV
python scripts/inference.py \
    --input data/compounds.csv \
    --output results.csv

# Docker 모드 (이미지 파일로 저장)
python scripts/inference.py \
    --input data.csv \
    --output results.csv \
    --docker \
    --output-dir ./images
```

**주요 옵션**:

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input` | SMILES 문자열 또는 CSV 파일 경로 | 필수 |
| `--output` | 결과 파일 경로 (.json / .csv) | — |
| `--mode` | `single` 또는 `batch` (자동 감지) | auto |
| `--image-mode` | `base64` / `file` / `auto` | auto |
| `--batch-size` | DataLoader 배치 크기 | 32 |
| `--checkpoints-dir` | 체크포인트 디렉토리 경로 | `./checkpoints` |
| `--device` | `cuda` 또는 `cpu` | auto |

---

## Streamlit UI

```bash
# 의존성 설치
pip install -e ".[serve]"

# 실행
streamlit run serve/app.py
```

브라우저에서 `http://localhost:8501` 접속 후 SMILES를 입력하면 52개 ADMET 특성과 radar chart를 확인할 수 있습니다.

---

## Docker

### 빌드 및 실행

```bash
# 빌드 (모노레포 루트에서 실행)
docker compose -f serve/docker-compose.yml build

# 개발 모드 (interactive bash)
docker compose -f serve/docker-compose.yml up

# 배치 추론 직접 실행
docker run --gpus all mga-inference:latest \
    --input /data/compounds.csv \
    --output /data/results.csv
```

> **Note**: GPU 사용을 위해 NVIDIA Container Toolkit이 설치되어 있어야 합니다.

---

## Training

### 1. 데이터 준비

```bash
# CSV 파일 → 바이너리 그래프 포맷
python scripts/prepare_data.py from-csv \
    -i data/raw.csv \
    -o data/processed.bin

# TDC 단일 태스크
python scripts/prepare_data.py from-tdc \
    -d CYP2C9_Veith \
    -o data/cyp2c9.bin

# TDC 멀티 태스크
python scripts/prepare_data.py from-tdc \
    -d CYP2C9_Veith,CYP3A4_Veith \
    -o data/cyp.bin

# 사용 가능한 TDC 데이터셋 목록
python scripts/prepare_data.py list-tdc
```

### 2. 학습 실행

```bash
# 기본 학습 (wandb 활성화)
python scripts/train.py --config config/config_admet.yaml

# wandb 없이 실행
python scripts/train.py --config config/config_admet.yaml --no-wandb

# 하이퍼파라미터 덮어쓰기
python scripts/train.py \
    --config config/config_admet.yaml \
    --epochs 200 \
    --batch-size 256 \
    --lr 0.0005
```

### 3. Config 파일 설명

`config/config_admet.yaml` 기준 주요 파라미터:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `num_epochs` | 500 | 최대 학습 에폭 |
| `patience` | 50 | Early stopping 인내 횟수 |
| `batch_size` | 512 | 미니배치 크기 |
| `lr` | 0.001 | 학습률 |
| `in_feats` | 40 | 원자 피처 차원 |
| `rgcn_hidden_feats` | [64, 64] | RGCN 은닉층 차원 |
| `classifier_hidden_feats` | 64 | 분류기 은닉층 차원 |
| `task_class` | `"classification_regression"` | 태스크 유형 |
| `bin_path` | `./data/admet_52.bin` | 바이너리 그래프 경로 |

**사전 제공 config 파일**:
- `config_admet.yaml` — 52개 태스크 혼합 (분류+회귀)
- `config_admet_clas.yaml` — 분류 태스크 전용
- `config_admet_deeppk.yaml` — DeepPK ADME 태스크

---

## API Reference

### `ADMETPredictor`

```python
ADMETPredictor(
    checkpoints_dir: str | Path | None = None,  # None이면 ./checkpoints 자동 탐색
    device: str | None = None,                   # 'cuda' | 'cpu' | None (자동)
    seed: int = 42,
)
```

| 메서드 | 반환 | 설명 |
|--------|------|------|
| `predict_single(smiles, batch_size=32, image_mode="auto", output_dir=None)` | `list[dict]` | 단일/소량 예측 + substructure 이미지 |
| `predict_batch(smiles_list, batch_size=32, generate_images=False, ...)` | `pd.DataFrame` | 대량 배치 예측 |
| `parse_input(input_str)` | — | SMILES 문자열 또는 CSV 파일 자동 감지 |

**반환 형식 (`predict_single`)**:
```json
[{
  "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "Predict": [
    {"task": "Caco-2 Permeability", "category": "absorption", "value": -5.21,
     "threshold": "regression", "sub_structure": "<base64_svg>", "unit": "cm/s"},
    ...
  ]
}]
```

### `PredictionFormatter`

```python
from mga.inference import PredictionFormatter

fmt = PredictionFormatter()
fmt.add_prediction(smiles, task_name, category, learning_task, value, structure, unit)
fmt.save_to_json("output.json")
fmt.save_to_csv("output.csv")
```

---

## Testing

```bash
# uv 사용 (권장)
uv run --with pytest python -m pytest tests/ -v

# 또는 환경 내에서
pytest tests/test_formatter.py tests/test_registry.py -v  # 체크포인트 불필요
pytest tests/test_inference_smoke.py -v                    # 체크포인트 필요
```

| 테스트 파일 | 의존성 | 설명 |
|-------------|--------|------|
| `test_formatter.py` | 없음 | PredictionFormatter 단위 테스트 |
| `test_registry.py` | 없음 (체크포인트 있으면 경로 검증) | task_registry 무결성 |
| `test_inference_smoke.py` | 체크포인트, DGL | E2E 추론 파이프라인 |

---

## ADMET Tasks

| 카테고리 | 태스크 예시 |
|----------|-------------|
| **Absorption** | Caco-2 Permeability, Pgp substrate/inhibitor, HIA, F20%, F50% |
| **Distribution** | BBB, VDss, Fu |
| **Metabolism** | CYP3A4/2D6/2C9/2C19/1A2 inhibitor & substrate, OATP1B1/1B3, BCRP |
| **Excretion** | T1/2, Clearance, OCT2 |
| **Toxicity** | hERG, DILI, Ames Mutagenicity, Carcinogenicity, Skin Sensitization 등 |
| **Tox21** | NR-AhR, NR-AR, NR-ER, SR-ARE, SR-MMP, SR-p53 등 |
| **General Properties** | pKa, logD, logP, logS, Hydration Energy, Boiling/Melting Point |

---

## Author

**PharosiBio** — `mga` v0.2.0

# MGA ADMET Inference - Production Docker Image
# NVIDIA CUDA 12.1 + Miniconda + ADMET_v2 환경
FROM nvidia/cuda:12.1.1-base-ubuntu20.04

ENV PATH /opt/conda/bin:$PATH

# 시스템 의존성 + Miniconda 설치
RUN apt-get update && apt-get install -y \
    wget bzip2 libx11-6 libxau6 libxcb1 libxdmcp6 libxext6 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 프로젝트 전체 복사 (monorepo 루트 기준)
COPY . .

# conda 환경 생성 (environment.yaml은 serve/ 디렉토리에 위치)
RUN conda env create -f serve/environment.yaml && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs

# conda 환경 활성화
SHELL ["conda", "run", "-n", "ADMET_v2", "/bin/bash", "-c"]
ENV PATH /opt/conda/envs/ADMET_v2/bin:$PATH
ENV DGLBACKEND=pytorch

# mga 패키지 설치 (코어만 - wandb/optuna 제외)
RUN pip install -e . --no-deps

# 기본 진입점: CLI 배치 추론
ENTRYPOINT ["conda", "run", "-n", "ADMET_v2", "python", "scripts/inference.py"]

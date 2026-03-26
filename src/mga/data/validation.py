"""
데이터셋 유효성 검사 모듈.

학습 전 데이터 품질을 확인합니다:
- SMILES 파싱 가능 여부
- 레이블 범위 (classification: {0, 1}, regression: 극단값)
- 클래스 균형 (10:1 초과 시 pos_weight 권장 경고)
- 결측값 비율 (task별)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mga.utils.logging import get_logger

logger = get_logger(__name__)

MISSING_LABEL_VALUE = 123456


@dataclass
class TaskStats:
    task_name: str
    task_type: str  # "classification" or "regression"
    n_valid: int
    n_missing: int
    missing_pct: float
    # Classification only
    n_positive: int = 0
    n_negative: int = 0
    class_ratio: float = 1.0  # neg / pos (>1 means more negative)
    # Regression only
    value_min: float = float("nan")
    value_max: float = float("nan")
    value_mean: float = float("nan")
    value_std: float = float("nan")


@dataclass
class ValidationReport:
    total_samples: int
    valid_smiles: int
    failed_smiles: List[str] = field(default_factory=list)
    per_task_stats: Dict[str, TaskStats] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def n_failed_smiles(self) -> int:
        return len(self.failed_smiles)

    @property
    def smiles_success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.valid_smiles / self.total_samples

    def is_valid(self) -> bool:
        """에러가 없으면 유효한 데이터셋."""
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "데이터셋 검증 리포트",
            "=" * 60,
            f"총 분자 수: {self.total_samples}",
            f"SMILES 파싱 성공: {self.valid_smiles} ({self.smiles_success_rate:.1%})",
        ]
        if self.failed_smiles:
            lines.append(f"파싱 실패: {self.n_failed_smiles}개 (첫 5개: {self.failed_smiles[:5]})")
        for task_name, stats in self.per_task_stats.items():
            lines.append(f"\n[Task: {task_name}]")
            lines.append(f"  결측값: {stats.missing_pct:.1%} ({stats.n_missing}/{stats.n_valid + stats.n_missing})")
            if stats.task_type == "classification":
                lines.append(f"  양성/음성: {stats.n_positive}/{stats.n_negative} (비율: {stats.class_ratio:.2f}:1)")
            else:
                lines.append(f"  값 범위: [{stats.value_min:.4g}, {stats.value_max:.4g}] (μ={stats.value_mean:.4g}, σ={stats.value_std:.4g})")
        if self.warnings:
            lines.append("\n[경고]")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if self.errors:
            lines.append("\n[오류]")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _validate_smiles(smiles_series: pd.Series) -> Tuple[List[str], List[str]]:
    """SMILES 파싱 가능 여부를 확인하여 (valid, failed) 목록 반환."""
    try:
        from rdkit.Chem import MolFromSmiles
    except ImportError:
        logger.warning("RDKit 없음, SMILES 유효성 검사 스킵")
        return smiles_series.tolist(), []

    valid, failed = [], []
    for smiles in smiles_series:
        mol = MolFromSmiles(str(smiles))
        if mol is None:
            failed.append(str(smiles))
        else:
            valid.append(str(smiles))
    return valid, failed


def _is_missing(value: float) -> bool:
    """결측값(123456 또는 NaN) 여부 확인."""
    return value == MISSING_LABEL_VALUE or (isinstance(value, float) and np.isnan(value))


def _compute_classification_stats(
    task_name: str,
    values: np.ndarray,
    warnings: List[str],
) -> TaskStats:
    valid_mask = ~np.array([_is_missing(v) for v in values])
    valid_values = values[valid_mask]
    n_missing = int((~valid_mask).sum())
    n_valid = int(valid_mask.sum())
    missing_pct = n_missing / len(values) if len(values) > 0 else 0.0

    n_pos = int((valid_values == 1).sum())
    n_neg = int((valid_values == 0).sum())
    unexpected = set(valid_values.tolist()) - {0, 1}

    if unexpected:
        warnings.append(
            f"[{task_name}] 예상치 못한 레이블 값: {unexpected}. "
            f"분류 태스크는 {{0, 1}}만 허용됩니다."
        )

    ratio = (n_neg / n_pos) if n_pos > 0 else float("inf")
    if ratio > 10:
        warnings.append(
            f"[{task_name}] 클래스 불균형 심각 (neg:pos = {ratio:.1f}:1). "
            "use_pos_weight=true 설정을 권장합니다."
        )
    elif ratio < 0.1 and n_neg > 0:
        warnings.append(
            f"[{task_name}] 클래스 불균형 심각 (pos:neg = {1/ratio:.1f}:1)."
        )

    return TaskStats(
        task_name=task_name,
        task_type="classification",
        n_valid=n_valid,
        n_missing=n_missing,
        missing_pct=missing_pct,
        n_positive=n_pos,
        n_negative=n_neg,
        class_ratio=ratio,
    )


def _compute_regression_stats(
    task_name: str,
    values: np.ndarray,
    warnings: List[str],
) -> TaskStats:
    valid_mask = ~np.array([_is_missing(v) for v in values])
    valid_values = values[valid_mask].astype(float)
    n_missing = int((~valid_mask).sum())
    n_valid = int(valid_mask.sum())
    missing_pct = n_missing / len(values) if len(values) > 0 else 0.0

    if n_valid == 0:
        return TaskStats(
            task_name=task_name,
            task_type="regression",
            n_valid=0,
            n_missing=n_missing,
            missing_pct=missing_pct,
        )

    vmin, vmax = float(valid_values.min()), float(valid_values.max())
    vmean, vstd = float(valid_values.mean()), float(valid_values.std())

    # 극단값 경고: 값이 3 표준편차 이상 벗어나는 비율
    z_scores = np.abs((valid_values - vmean) / (vstd + 1e-8))
    extreme_pct = float((z_scores > 5).mean())
    if extreme_pct > 0.01:
        warnings.append(
            f"[{task_name}] 극단값이 {extreme_pct:.1%}로 많음 "
            f"(범위: [{vmin:.4g}, {vmax:.4g}]). 전처리를 고려하세요."
        )

    return TaskStats(
        task_name=task_name,
        task_type="regression",
        n_valid=n_valid,
        n_missing=n_missing,
        missing_pct=missing_pct,
        value_min=vmin,
        value_max=vmax,
        value_mean=vmean,
        value_std=vstd,
    )


def validate_dataset(
    df: pd.DataFrame,
    task_list: List[str],
    task_class: str = "classification",
    classification_tasks: Optional[List[str]] = None,
    regression_tasks: Optional[List[str]] = None,
    smiles_col: str = "smiles",
    warn_missing_threshold: float = 0.3,
) -> ValidationReport:
    """
    데이터셋 유효성 검사.

    Args:
        df: 입력 DataFrame (smiles 컬럼 + task 컬럼 포함)
        task_list: 검사할 태스크 컬럼 이름 목록
        task_class: "classification", "regression", "classification_regression"
        classification_tasks: 분류 태스크 이름 목록 (classification_regression 시)
        regression_tasks: 회귀 태스크 이름 목록 (classification_regression 시)
        smiles_col: SMILES 컬럼 이름 (default: "smiles")
        warn_missing_threshold: 결측값 비율 경고 임계값 (default: 0.3 = 30%)

    Returns:
        ValidationReport

    Example:
        >>> report = validate_dataset(df, task_list=["Caco2", "HIA"], task_class="classification")
        >>> print(report.summary())
        >>> if not report.is_valid():
        ...     raise ValueError("데이터 검증 실패")
    """
    warnings: List[str] = []
    errors: List[str] = []

    # ── SMILES 유효성 검사 ────────────────────────────────
    if smiles_col not in df.columns:
        errors.append(f"'{smiles_col}' 컬럼이 없습니다.")
        return ValidationReport(
            total_samples=len(df),
            valid_smiles=0,
            errors=errors,
        )

    valid_smiles, failed_smiles = _validate_smiles(df[smiles_col])
    total = len(df)

    if failed_smiles:
        ratio = len(failed_smiles) / total
        msg = f"SMILES 파싱 실패: {len(failed_smiles)}개 ({ratio:.1%})"
        if ratio > 0.05:
            errors.append(msg)
        else:
            warnings.append(msg)

    # ── 태스크 컬럼 존재 확인 ─────────────────────────────
    missing_cols = [t for t in task_list if t not in df.columns]
    if missing_cols:
        errors.append(f"태스크 컬럼 없음: {missing_cols}")

    # ── 태스크별 통계 계산 ────────────────────────────────
    per_task_stats: Dict[str, TaskStats] = {}

    # 태스크 유형 결정
    if task_class == "classification_regression":
        clas_tasks = classification_tasks or []
        reg_tasks = regression_tasks or []
    elif task_class == "classification":
        clas_tasks = task_list
        reg_tasks = []
    else:  # regression
        clas_tasks = []
        reg_tasks = task_list

    for task_name in task_list:
        if task_name not in df.columns:
            continue
        values = df[task_name].values

        if task_name in clas_tasks:
            stats = _compute_classification_stats(task_name, values, warnings)
        else:
            stats = _compute_regression_stats(task_name, values, warnings)

        per_task_stats[task_name] = stats

        # 결측값 비율 경고
        if stats.missing_pct > warn_missing_threshold:
            warnings.append(
                f"[{task_name}] 결측값 비율이 {stats.missing_pct:.1%}로 높음."
            )

    report = ValidationReport(
        total_samples=total,
        valid_smiles=len(valid_smiles),
        failed_smiles=failed_smiles,
        per_task_stats=per_task_stats,
        warnings=warnings,
        errors=errors,
    )

    # 로그로 요약 출력
    logger.info(f"데이터 검증: {total}개 분자, {len(valid_smiles)}개 유효 SMILES")
    if warnings:
        for w in warnings:
            logger.warning(w)
    if errors:
        for e in errors:
            logger.error(e)

    return report

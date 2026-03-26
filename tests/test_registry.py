"""
task_registry 무결성 테스트.

- 모든 MODEL_FILENAMES 경로가 checkpoints/ 디렉토리 내에 실제 존재하는지 확인
- DISPLAY_NAMES가 모든 TASK_LISTS 코드명을 커버하는지 확인
- get_model_paths()가 올바른 절대 경로를 반환하는지 확인
"""

from pathlib import Path

import pytest

from mga.inference.task_registry import (
    DISPLAY_NAMES,
    MODEL_FILENAMES,
    TASK_LISTS,
    TOXICITY_REG_TASKS,
    get_model_paths,
    get_display_name,
)

CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def test_model_filenames_keys_match_task_lists():
    """MODEL_FILENAMES의 키가 TASK_LISTS 키와 동일해야 함."""
    assert set(MODEL_FILENAMES.keys()) == set(TASK_LISTS.keys()), (
        f"Mismatch: {set(MODEL_FILENAMES.keys()) ^ set(TASK_LISTS.keys())}"
    )


@pytest.mark.skipif(
    not CHECKPOINTS_DIR.exists(),
    reason="checkpoints/ 디렉토리 없음 - CI 환경에서 스킵"
)
def test_checkpoint_files_exist():
    """실제 .pth 파일이 checkpoints/ 하위에 존재해야 함."""
    model_paths = get_model_paths(CHECKPOINTS_DIR)
    missing = [
        (task, path)
        for task, path in model_paths.items()
        if not Path(path).exists()
    ]
    assert not missing, f"Missing checkpoint files:\n" + "\n".join(
        f"  {task}: {path}" for task, path in missing
    )


def test_display_names_cover_all_tasks():
    """DISPLAY_NAMES가 모든 TASK_LISTS 코드명을 포함해야 함 (TOXICITY_REG_TASKS 제외)."""
    all_tasks = {
        task
        for tasks in TASK_LISTS.values()
        for task in tasks
        if task not in TOXICITY_REG_TASKS
    }
    missing = all_tasks - set(DISPLAY_NAMES.keys())
    assert not missing, f"DISPLAY_NAMES에 누락된 태스크: {sorted(missing)}"


def test_get_display_name_returns_fallback():
    """알 수 없는 코드명에 대해 원래 이름을 반환해야 함."""
    assert get_display_name("unknown_task_xyz") == "unknown_task_xyz"


def test_get_display_name_known():
    """알려진 태스크에 대해 표시명을 반환해야 함."""
    assert get_display_name("caco2_reg") == "Caco-2 Permeability"


def test_get_model_paths_absolute():
    """get_model_paths()가 절대 경로 문자열을 반환해야 함."""
    paths = get_model_paths("/tmp/ckpt")
    for task, path in paths.items():
        assert path.startswith("/tmp/ckpt"), f"{task} path not under base dir: {path}"


def test_toxicity_reg_tasks_subset_of_toxicity():
    """TOXICITY_REG_TASKS가 TASK_LISTS['toxicity'] 의 부분집합이어야 함."""
    assert TOXICITY_REG_TASKS <= set(TASK_LISTS["toxicity"])

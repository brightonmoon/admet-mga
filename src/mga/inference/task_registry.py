"""
태스크 레지스트리 - 모델 경로, 태스크 목록, 하이퍼파라미터의 단일 소스.
"""

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# 모델 파일명 (checkpoints/ 디렉토리 기준 상대 경로)
# ---------------------------------------------------------------------------
MODEL_FILENAMES = {
    "caco2":              "single_model/caco2_model.pth",
    "absorption":         "absorption_model.pth",
    "distribution":       "distribution_model.pth",
    "metabolism":         "metabolism_model.pth",
    "half_life":          "single_model/half_life_model.pth",
    "excretion":          "excretion_model.pth",
    "tox21":              "tox21_model.pth",
    "toxicity":           "toxicity_model.pth",
    "pka":                "single_model/pka_model.pth",
    "general_properties": "general_properties_model.pth",
}


def get_model_paths(checkpoints_dir: str | Path) -> dict[str, str]:
    """checkpoints 디렉토리 경로를 받아 각 태스크의 전체 모델 경로 반환."""
    base = Path(checkpoints_dir)
    return {task: str(base / rel) for task, rel in MODEL_FILENAMES.items()}


# ---------------------------------------------------------------------------
# 태스크 목록 (코드명 기준 - CSV 저장, 후처리 로직에 사용)
# ---------------------------------------------------------------------------
TASK_LISTS: dict[str, list[str]] = {
    "caco2":     ["caco2_reg"],
    "absorption": ["pgp_substrate", "pgp_inhibitor", "ob", "f20", "hia"],
    "distribution": ["bbb_logbb", "vd", "fu"],
    "metabolism": [
        "oatp1b3", "oatp1b1", "cyp3a4_inhibitor", "cyp2d6_inhibitor",
        "cyp2c9_inhibitor", "cyp2c19_inhibitor", "cyp1a2_inhibitor",
        "cyp3a4_substrate", "cyp2d6_substrate", "cyp2c9_substrate",
        "cyp2c19_substrate", "cyp1a2_substrate", "bcrp",
    ],
    "half_life":  ["t0.5"],
    "excretion":  ["oct2", "cl"],
    "tox21": [
        "nr_ahr", "nr_ar", "nr_ar_lbd", "nr_er", "nr_er_lbd",
        "nr_tr", "nr_gr", "nr_ppar_gamma", "nr_aromatase",
        "sr_are", "sr_atad5", "sr_hse", "sr_mmp", "sr_p53",
    ],
    "toxicity": [
        "skin_sens", "respiratory_tox", "micronucleus_tox", "herg",
        "h_ht", "eye_irritation", "eye_corrosion", "dili",
        "crustacean", "carcinogenicity", "biodegradation", "bee_tox",
        "avian_tox", "ames",
        # 회귀 태스크 (단일 처리 시 포맷팅에서 제외)
        "rat_acute_reg", "fdamdd_reg", "fm_reg", "rat_chronic",
        "bioconcF", "lc50dm", "pyriformis_reg",
    ],
    "pka":               ["pka"],
    "general_properties": ["pkb", "logd", "logp", "logs", "logvp", "hydrationE", "bp", "mp"],
}

# 단일 처리 시 결과에서 제외할 회귀 태스크 (toxicity 내부)
TOXICITY_REG_TASKS = frozenset([
    "rat_acute_reg", "fdamdd_reg", "fm_reg", "rat_chronic",
    "bioconcF", "lc50dm", "pyriformis_reg",
])

# ---------------------------------------------------------------------------
# 코드명 -> 표시명 매핑 (Streamlit UI용)
# ---------------------------------------------------------------------------
DISPLAY_NAMES: dict[str, str] = {
    "caco2_reg":          "Caco-2 Permeability",
    "pgp_substrate":      "Pgp substrate",
    "pgp_inhibitor":      "Pgp inhibitor",
    "ob":                 "F50%",
    "f20":                "F20%",
    "hia":                "HIA",
    "bbb_logbb":          "BBB",
    "vd":                 "VDss",
    "fu":                 "Fu",
    "oatp1b3":            "OATP1B3 inhibitor",
    "oatp1b1":            "OATP1B1 inhibitor",
    "cyp3a4_inhibitor":   "CYP3A4 inhibitor",
    "cyp2d6_inhibitor":   "CYP2D6 inhibitor",
    "cyp2c9_inhibitor":   "CYP2C9 inhibitor",
    "cyp2c19_inhibitor":  "CYP2C19 inhibitor",
    "cyp1a2_inhibitor":   "CYP1A2 inhibitor",
    "cyp3a4_substrate":   "CYP3A4 substrate",
    "cyp2d6_substrate":   "CYP2D6 substrate",
    "cyp2c9_substrate":   "CYP2C9 substrate",
    "cyp2c19_substrate":  "CYP2C19 substrate",
    "cyp1a2_substrate":   "CYP1A2 substrate",
    "bcrp":               "BCRP inhibitor",
    "t0.5":               "T 1/2",
    "oct2":               "OCT2 inhibitor",
    "cl":                 "Clearance",
    "pka":                "pKa",
    "pkb":                "pkb",
    "logd":               "logD",
    "logp":               "logP",
    "logs":               "logS",
    "logvp":              "logVP",
    "hydrationE":         "Hydration Energy",
    "bp":                 "Boiling Point",
    "mp":                 "Melting Point",
}


def get_display_name(code_name: str) -> str:
    """코드명을 표시명으로 변환. 매핑에 없으면 코드명 그대로 반환."""
    return DISPLAY_NAMES.get(code_name, code_name)


# ---------------------------------------------------------------------------
# 모델 하이퍼파라미터
# ---------------------------------------------------------------------------
TASK_PARAMS: dict[str, dict] = {
    "caco2":             {"hidden_feats": 128, "dropout": 0.1},
    "absorption":        {"hidden_feats": 128, "dropout": 0.1},
    "distribution":      {"hidden_feats": 128, "dropout": 0.1},
    "metabolism":        {"hidden_feats": 128, "dropout": 0.1},
    "half_life":         {"hidden_feats": 64,  "dropout": 0.2},
    "excretion":         {"hidden_feats": 128, "dropout": 0.1},
    "tox21":             {"hidden_feats": 128, "dropout": 0.2},
    "toxicity":          {"hidden_feats": 128, "dropout": 0.2},
    "pka":               {"hidden_feats": 128, "dropout": 0.1},
    "general_properties":{"hidden_feats": 64,  "dropout": 0.1},
}

# ---------------------------------------------------------------------------
# 태스크별 후처리 (sigmoid, 스케일링)
# ---------------------------------------------------------------------------

# toxicity에서 분류 태스크 수: TOXICITY_REG_TASKS를 제외한 나머지
# 이 값을 하드코딩하지 않고 리스트에서 파생
_TOXICITY_CLS_COUNT = len([t for t in TASK_LISTS["toxicity"] if t not in TOXICITY_REG_TASKS])


def _apply_sigmoid(result, task, scale: float = 1.0):
    """sigmoid 적용 + 선택적 스케일링 (배치/단일 공통 로직)."""
    match task:
        case "absorption" | "metabolism" | "half_life" | "tox21":
            result = torch.sigmoid(result) * scale
        case "distribution" | "excretion":
            clas = torch.sigmoid(result[:, :1]) * scale
            result = torch.cat((clas, result[:, 1:]), dim=1)
        case "toxicity":
            clas = torch.sigmoid(result[:, :_TOXICITY_CLS_COUNT]) * scale
            result = torch.cat((clas, result[:, _TOXICITY_CLS_COUNT:]), dim=1)
        case _:
            pass  # caco2, pka, general_properties: 후처리 없음
    return result


def apply_postprocessing_batch(result, task):
    """배치 모드: sigmoid + *100 스케일링 (백분율)."""
    return _apply_sigmoid(result, task, scale=100.0)


def apply_postprocessing_single(result, task):
    """단일 모드: sigmoid만 적용 (스케일링은 get_task_meta의 scale로 처리)."""
    return _apply_sigmoid(result, task, scale=1.0)


# ---------------------------------------------------------------------------
# 태스크별 포맷팅 메타데이터 (단위, 학습 타입 등)
# ---------------------------------------------------------------------------

def get_task_meta(task: str, task_name: str) -> dict:
    """
    태스크와 태스크명을 받아 포맷팅에 필요한 메타데이터 반환.
    Returns: {"category": str, "learning_task": str, "unit": str, "scale": float}
    scale: predict_value에 곱할 배수 (1.0 = 그대로, 100.0 = 백분율)
    """
    match task:
        case "caco2":
            return {"category": "absorption", "learning_task": "regression", "unit": "log cm/s", "scale": 1.0}
        case "half_life":
            return {"category": "excretion", "learning_task": "classification", "unit": "% (>= 5h)", "scale": 100.0}
        case "pka":
            return {"category": "general_properties", "learning_task": "regression", "unit": "pKa", "scale": 1.0}
        case "distribution":
            if task_name == "bbb_logbb":
                return {"category": "distribution", "learning_task": "classification", "unit": "%", "scale": 100.0}
            elif task_name == "vd":
                return {"category": "distribution", "learning_task": "regression", "unit": "L/kg", "scale": 1.0}
            else:
                return {"category": "distribution", "learning_task": "regression", "unit": "%", "scale": 100.0}
        case "excretion":
            if task_name == "oct2":
                return {"category": "excretion", "learning_task": "classification", "unit": "%", "scale": 100.0}
            else:
                return {"category": "excretion", "learning_task": "regression", "unit": "ml/min/kg", "scale": 1.0}
        case "toxicity":
            return {"category": "toxicity", "learning_task": "classification", "unit": "%", "scale": 100.0}
        case "absorption" | "metabolism" | "tox21":
            return {"category": task, "learning_task": "classification", "unit": "%", "scale": 100.0}
        case "general_properties":
            unit_map = {
                "pkb": "pkb",
                "logd": "log mol/L", "logp": "log mol/L", "logs": "log mol/L",
                "logvp": "log kPa",
                "hydrationE": "kcal/mol",
                "bp": "°C", "mp": "°C",
            }
            return {"category": "general_properties", "learning_task": "regression",
                    "unit": unit_map.get(task_name, ""), "scale": 1.0}
        case _:
            return {"category": task, "learning_task": "regression", "unit": "", "scale": 1.0}

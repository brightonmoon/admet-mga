"""
시각화 유틸리티 - 원자 가중치 기반 서브구조 강조 이미지 생성.
mga_inference/src/utils.py에서 이전.
"""

import base64
import hashlib
import os
from pathlib import Path

import dgl
import matplotlib.cm as cm
import matplotlib.colors
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# 모듈 수준에서 1회 생성 (weight_visualize_string 호출마다 재생성 방지)
_NORM = matplotlib.colors.Normalize(vmin=0, vmax=1)
_CMAP = matplotlib.colormaps["Oranges"]
_PLT_COLORS = cm.ScalarMappable(norm=_NORM, cmap=_CMAP)


def weight_visualize_string(smiles: str, atom_weight) -> str:
    """
    원자 attention 가중치를 기반으로 서브구조를 강조한 SVG를 Base64로 반환.

    Args:
        smiles: SMILES 문자열
        atom_weight: 원자별 가중치 텐서 (shape: [num_atoms])

    Returns:
        Base64 인코딩된 SVG 문자열
    """
    atom_weight = atom_weight.cpu()
    atom_weight_list = atom_weight.squeeze().numpy().tolist()

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    num_atoms = mol.GetNumAtoms()
    if len(atom_weight_list) != num_atoms:
        raise ValueError(
            f"Atom weight size ({len(atom_weight_list)}) != num atoms ({num_atoms})"
        )

    max_idx = atom_weight_list.index(max(atom_weight_list))
    significant_weight = atom_weight[max_idx].item()

    atom_new_weight = [0.0] * num_atoms

    atom = mol.GetAtomWithIdx(max_idx)
    neighbors_1 = [x.GetIdx() for x in atom.GetNeighbors()]

    neighbors_2 = []
    for idx1 in neighbors_1:
        neighbors_2 += [x.GetIdx() for x in mol.GetAtomWithIdx(idx1).GetNeighbors()]
    if max_idx in neighbors_2:
        neighbors_2.remove(max_idx)

    neighbors_3 = []
    for idx2 in neighbors_2:
        neighbors_3 += [x.GetIdx() for x in mol.GetAtomWithIdx(idx2).GetNeighbors()]
    neighbors_3 = [x for x in neighbors_3 if x not in neighbors_1]

    for i in neighbors_3:
        atom_new_weight[i] = significant_weight * 0.5
    for i in neighbors_2:
        atom_new_weight[i] = significant_weight
    for i in neighbors_1:
        atom_new_weight[i] = significant_weight
    atom_new_weight[max_idx] = significant_weight

    atom_colors = {
        i: _PLT_COLORS.to_rgba(float(atom_new_weight[i])) for i in range(num_atoms)
    }
    bond_colors = {}
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_colors[i] = _PLT_COLORS.to_rgba((atom_new_weight[u] + atom_new_weight[v]) / 2)

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(6)
    drawer.DrawMolecule(
        rdMolDraw2D.PrepareMolForDrawing(mol),
        highlightAtoms=range(num_atoms),
        highlightBonds=range(mol.GetNumBonds()),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    svg_bytes = drawer.GetDrawingText().replace("svg:", "").encode("utf-8")
    return base64.b64encode(svg_bytes).decode("utf-8")


def return_result_supgraph(model, smiles: list, bg, atom_feats, bond_feats, task_list: list):
    """
    모델 순전파 + 원자 attention 기반 서브구조 이미지 생성.

    Returns:
        (result tensor, images) where images[mol_idx] = [base64_svg per task]
    """
    result, atom_weight_list, node_feats = model(bg, atom_feats, bond_feats, norm=None)

    # task 루프 기준으로 unbatch: O(n_mols × n_tasks) → O(n_tasks) unbatch 호출
    img_list = [[None] * len(task_list) for _ in range(len(smiles))]
    for task_index, _ in enumerate(task_list):
        bg.ndata["w"] = atom_weight_list[task_index]
        bg.ndata["feats"] = node_feats
        unbatch_bg = dgl.unbatch(bg)
        for mol_index, atom_smiles in enumerate(smiles):
            one_atom_weight = unbatch_bg[mol_index].ndata["w"]
            img_list[mol_index][task_index] = weight_visualize_string(atom_smiles, one_atom_weight)
    return result, img_list


def return_result(model, bg, atom_feats, bond_feats):
    """이미지 없이 예측값만 반환 (배치 모드용)."""
    return model(bg, atom_feats, bond_feats, norm=None)


class ImageHandler:
    """이미지 처리 전략 패턴 - Base64 반환 또는 파일 저장."""

    def __init__(self, mode: str = "auto", output_dir: str | Path | None = None):
        """
        Args:
            mode: 'base64' | 'file' | 'auto' (Docker 감지 시 file, 아니면 base64)
            output_dir: 파일 저장 디렉토리 (file 모드 필수)
        """
        if mode == "auto":
            mode = "file" if self._is_docker() else "base64"

        self.mode = mode
        self.output_dir = Path(output_dir) if output_dir else None

        if mode == "file":
            if not output_dir:
                raise ValueError("output_dir is required for file mode")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "images").mkdir(exist_ok=True)

    @staticmethod
    def _is_docker() -> bool:
        return (
            os.path.exists("/.dockerenv")
            or os.getenv("DOCKER_CONTAINER", "").lower() == "true"
            or os.getenv("DOCKER", "").lower() == "true"
        )

    @staticmethod
    def _generate_filename(smiles: str, task: str, task_name: str) -> str:
        smiles_hash = hashlib.md5(smiles.encode(), usedforsecurity=False).hexdigest()[:8]
        safe_name = task_name.replace("/", "_").replace("\\", "_")
        return f"{smiles_hash}_{task}_{safe_name}.svg"

    def process_image(self, base64_svg: str, smiles: str, task: str, task_name: str) -> str:
        """
        Returns:
            Base64 문자열 (base64 모드) 또는 파일 경로 문자열 (file 모드)
        """
        if self.mode == "base64":
            return base64_svg

        filename = self._generate_filename(smiles, task, task_name)
        file_path = self.output_dir / "images" / filename
        svg_data = base64.b64decode(base64_svg)
        with open(file_path, "wb") as f:
            f.write(svg_data)
        return str(Path("images") / filename)

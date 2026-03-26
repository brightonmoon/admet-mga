"""
데이터 파이프라인 단위 테스트.
"""

from __future__ import annotations

import pytest
import torch


class TestGraphConstruction:
    def test_ethanol_graph(self, mini_graph):
        """CCO (에탄올): 3 atoms, 4 edges (bidirectional)."""
        g = mini_graph
        assert g.num_nodes() == 3  # C, C, O
        assert g.num_edges() == 4  # 2 bonds × 2 directions

    def test_atom_features_shape(self, mini_graph):
        """원자 특징 shape: [n_atoms, in_feats]."""
        g = mini_graph
        atom_feats = g.ndata["atom"]
        assert atom_feats.shape[0] == g.num_nodes()
        assert atom_feats.shape[1] == 40  # default in_feats

    def test_edge_type_present(self, mini_graph):
        """엣지 타입 데이터(etype)가 존재해야 함."""
        g = mini_graph
        assert "etype" in g.edata

    def test_invalid_smiles_raises(self):
        """잘못된 SMILES는 ValueError를 발생시켜야 함."""
        from mga.data.dataset import construct_graph_from_smiles
        with pytest.raises((ValueError, RuntimeError)):
            construct_graph_from_smiles("INVALID_SMILES_XYZ")

    def test_benzene_graph(self):
        """벤젠(c1ccccc1): 6 atoms, 12 edges."""
        from mga.data.dataset import construct_graph_from_smiles
        g = construct_graph_from_smiles("c1ccccc1")
        assert g.num_nodes() == 6
        assert g.num_edges() == 12

    def test_methane_single_atom(self):
        """메탄(C): 1 atom, 0 edges (self-loop은 제외)."""
        from mga.data.dataset import construct_graph_from_smiles
        g = construct_graph_from_smiles("C")
        assert g.num_nodes() == 1


class TestBuildDataset:
    def test_dataset_length(self, mini_dataset):
        """build_dataset: 10개 분자 → 10개 항목."""
        assert len(mini_dataset) == 10

    def test_dataset_element_structure(self, mini_dataset):
        """각 항목: (smiles, graph, labels, mask, split_index)."""
        smiles, graph, labels, mask, split = mini_dataset[0]
        assert isinstance(smiles, str)
        assert hasattr(graph, "num_nodes")  # DGL graph
        assert isinstance(labels, list)
        assert isinstance(mask, list)
        assert isinstance(split, str)

    def test_mask_marks_missing_labels(self, mini_dataset):
        """123456 값은 mask=0으로 표시되어야 함."""
        import numpy as np
        # mini_dataset has 123456 at index 2 (task_0) and 5 (task_1)
        for item in mini_dataset:
            _, _, labels, mask, _ = item
            for i, (lbl, msk) in enumerate(zip(labels, mask)):
                if lbl == 123456:
                    assert msk == 0, f"label=123456 should have mask=0, got mask={msk}"

    def test_build_dataset_from_dataframe(self):
        """CSV → dataset 엔드-투-엔드 파이프라인."""
        import pandas as pd
        from mga.data.dataset import build_dataset

        df = pd.DataFrame({
            "smiles": ["CCO", "CC(=O)O", "c1ccccc1"],
            "label": [0, 1, 0],
            "group": ["training", "training", "valid"],
        })
        dataset = build_dataset(df, task_list=["label"])
        assert len(dataset) == 3
        smiles, graph, labels, mask, split = dataset[0]
        assert smiles == "CCO"
        assert len(labels) == 1

    def test_labels_count_equals_tasks(self, mini_dataset):
        """레이블 수 = task_list 길이 = 2."""
        _, _, labels, mask, _ = mini_dataset[0]
        assert len(labels) == 2
        assert len(mask) == 2


class TestCollateFunctions:
    def test_collate_molgraphs_shapes(self, mini_dataset):
        """collate_molgraphs: 올바른 배치 shape 반환."""
        from mga.data.collate import collate_molgraphs

        items = [(m[0], m[1], m[2], m[3]) for m in mini_dataset[:4]]
        smiles, bg, labels, mask = collate_molgraphs(items)
        assert len(smiles) == 4
        assert labels.shape == (4, 2)
        assert mask.shape == (4, 2)
        assert bg.num_nodes() > 0

    def test_collate_returns_tensors(self, mini_dataset):
        """labels와 mask는 torch.Tensor이어야 함."""
        from mga.data.collate import collate_molgraphs

        items = [(m[0], m[1], m[2], m[3]) for m in mini_dataset[:4]]
        _, _, labels, mask = collate_molgraphs(items)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

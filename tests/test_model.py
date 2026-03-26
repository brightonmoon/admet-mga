"""
MGA 모델 아키텍처 단위 테스트.

CPU + 소규모 모델(hidden=[16,16], n_tasks=2)으로 실행.
"""

from __future__ import annotations

import pytest
import torch


def _make_batch(n_molecules: int = 4, n_tasks: int = 2):
    """Build a mini batched DGL graph from simple SMILES."""
    import dgl
    from mga.data.dataset import construct_graph_from_smiles
    from mga.data.collate import collate_molgraphs

    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1", "C1CCCCC1"][:n_molecules]
    graphs = [construct_graph_from_smiles(s) for s in smiles_list]
    labels = torch.zeros(n_molecules, n_tasks)
    mask = torch.ones(n_molecules, n_tasks)

    items = [(s, g, labels[i].tolist(), mask[i].tolist()) for i, (s, g) in enumerate(zip(smiles_list, graphs))]
    _, bg, labels_t, mask_t = collate_molgraphs(items)
    atom_feats = bg.ndata.pop("atom").float()
    bond_feats = bg.edata.pop("etype").long()
    return bg, atom_feats, bond_feats


class TestMGA:
    def test_forward_shape(self, mini_model):
        """Output shape should be [batch, n_tasks]."""
        bg, atom_feats, bond_feats = _make_batch(4, 2)
        out = mini_model(bg, atom_feats, bond_feats)
        assert out.shape == (4, 2)

    def test_forward_no_nan(self, mini_model):
        """Output should not contain NaN or Inf."""
        bg, atom_feats, bond_feats = _make_batch(4, 2)
        out = mini_model(bg, atom_feats, bond_feats)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_return_weight(self, mini_model):
        """return_weight=True should return 3-tuple (logits, weights, batched_graph)."""
        mini_model.return_weight = True
        bg, atom_feats, bond_feats = _make_batch(4, 2)
        result = mini_model(bg, atom_feats, bond_feats)
        assert isinstance(result, tuple)
        assert len(result) == 3  # logits, atom_weights, bg
        logits = result[0]
        assert logits.shape == (4, 2)
        mini_model.return_weight = False

    def test_return_mol_embedding(self, mini_model):
        """return_mol_embedding=True should include embedding in output."""
        mini_model.return_mol_embedding = True
        bg, atom_feats, bond_feats = _make_batch(4, 2)
        result = mini_model(bg, atom_feats, bond_feats)
        assert isinstance(result, tuple)
        logits = result[0]
        assert logits.shape == (4, 2)
        mini_model.return_mol_embedding = False

    def test_single_atom_molecule(self, mini_model):
        """Methane (single atom) should not crash."""
        import dgl
        from mga.data.dataset import construct_graph_from_smiles
        from mga.data.collate import collate_molgraphs

        g = construct_graph_from_smiles("C")
        items = [("C", g, [0.0, 0.0], [1.0, 1.0])]
        _, bg, _, _ = collate_molgraphs(items)
        atom_feats = bg.ndata.pop("atom").float()
        bond_feats = bg.edata.pop("etype").long()
        out = mini_model(bg, atom_feats, bond_feats)
        assert out.shape == (1, 2)

    def test_batch_size_one(self, mini_model):
        """Single molecule batch should work."""
        bg, atom_feats, bond_feats = _make_batch(1, 2)
        out = mini_model(bg, atom_feats, bond_feats)
        assert out.shape == (1, 2)


class TestMGATest:
    def test_forward_shape(self, mini_model_test):
        """MGATest output shape should be [batch, n_tasks]."""
        bg, atom_feats, bond_feats = _make_batch(4, 2)
        out = mini_model_test(bg, atom_feats, bond_feats)
        assert out.shape == (4, 2)

    def test_forward_no_nan(self, mini_model_test):
        """MGATest should not produce NaN."""
        bg, atom_feats, bond_feats = _make_batch(4, 2)
        out = mini_model_test(bg, atom_feats, bond_feats)
        assert not torch.isnan(out).any()

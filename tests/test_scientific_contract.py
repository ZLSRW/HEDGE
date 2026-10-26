import unittest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from hedge.config import HyperParams, RELATION_LIST
from hedge.model import HEDGE
from hedge.layers import ChannelAttention, ConvergenceGate
from hedge.objectives import FixedNoiseMapping, consistency, supervised, objective
from hedge.training import graph_intervention, setup_component
from hedge.analysis import bh_adjust, effect_space, mapped_enrichment, category_jaccard
torch.set_num_threads(2)

def graph():
    torch.manual_seed(17)
    n = 12
    edge = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 5, 4, 7, 8, 9, 10, 11, 6]])
    typ = torch.tensor([0, 0, 1, 1, 3, 3, 4, 4, 4, 4, 4, 4])
    edge = torch.cat([edge, edge.flip(0)], dim=1)
    typ = typ.repeat(2)
    gm = torch.arange(n) >= 6
    labels = torch.full((n,), -1.0)
    labels[gm] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    counts = torch.zeros(n, 7)
    counts[gm, :3] = torch.tensor([1.0, 2.0, 0.0])
    return (Data(x=torch.randn(n, 64), edge_index=edge, edge_type=typ, edge_weight=torch.ones(24)), gm, labels, counts)

class Contract(unittest.TestCase):

    def test_nine_slots_do_not_imply_nine_edge_relations(self):
        att = ChannelAttention(64, 9, heads=4, attn_drop=0).eval()
        torch.manual_seed(5)
        z, a = att({i: torch.randn(7, 64) for i in range(5)})
        self.assertEqual(a.shape, (4, 9, 7))
        self.assertTrue(torch.allclose(a.sum(1), torch.ones(4, 7), atol=1e-06))
        self.assertTrue(bool((a[:, 5:] > 0).all()))
        self.assertTrue(torch.equal(a[:, 5], a[:, 8]))
        self.assertEqual(RELATION_LIST[4], 'ppi')
        self.assertEqual(z.shape, (7, 64))

    def test_gate_is_row_mixing_then_weighted_sum(self):
        gate = ConvergenceGate(64)
        inputs = [torch.randn(8, 64) for _ in range(3)]
        counts = torch.rand(8, 7)
        fused, a = gate(*inputs, counts)
        expected = (torch.einsum('ij,njh->nih', a['A_static'], torch.stack(inputs, 1)) * a['gates'].unsqueeze(-1)).sum(1)
        self.assertTrue(torch.allclose(fused, expected))
        self.assertTrue(bool(((a['gates'] >= 0) & (a['gates'] <= 1.5)).all()))

    def test_probability_consistency_is_not_full_binary_kl(self):
        a = torch.logit(torch.tensor([0.2, 0.7]))
        b = torch.logit(torch.tensor([0.6, 0.8]))
        mask = torch.ones(2, dtype=torch.bool)
        expected = 0.5 * np.mean(np.array([0.2, 0.7]) * np.log(np.array([0.2, 0.7]) / np.array([0.6, 0.8])) + np.array([0.6, 0.8]) * np.log(np.array([0.6, 0.8]) / np.array([0.2, 0.7])))
        self.assertAlmostEqual(consistency(a, b, mask).item(), expected, places=6)

    def test_empty_label_mask_is_finite_and_zero(self):
        a = torch.randn(4, requires_grad=True)
        m = torch.zeros(4, dtype=torch.bool)
        self.assertEqual(supervised(a, torch.ones(4), m).item(), 0)
        self.assertEqual(consistency(a, a, m).item(), 0)

    def test_fixed_mapping_passes_encoder_gradient(self):
        mapping = FixedNoiseMapping(64)
        before = {k: v.clone() for k, v in mapping.state_dict().items()}
        z = torch.randn(5, 64, requires_grad=True)
        loss = mapping(z, torch.zeros_like(z)).square().mean()
        loss.backward()
        self.assertGreater(z.grad.abs().sum().item(), 0)
        self.assertTrue(all((p.grad is None for p in mapping.parameters())))
        for k, v in before.items():
            self.assertTrue(torch.equal(v, mapping.state_dict()[k]))

    def test_full_objective_backpropagates(self):
        d, gm, y, c = graph()
        hp = HyperParams(num_layers=2)
        model = HEDGE(hp)
        mapping = FixedNoiseMapping(64)
        loss, parts = objective(model, d, y, gm, c, gm, hp, mapping)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {'supervised', 'smooth', 'consistency', 'invariance', 'noise'})
        loss.backward()
        self.assertGreater(model.gene_proj.weight.grad.abs().sum().item(), 0)

    def test_empty_supervision_keeps_only_smoothing(self):
        d, gm, y, c = graph()
        hp = HyperParams(num_layers=1)
        model = HEDGE(hp)
        loss, parts = objective(model, d, y, gm, c, torch.zeros_like(gm), hp, FixedNoiseMapping(64))
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), parts['smooth'].item(), places=5)

    def test_intervention_preserves_ld_ppi_nodes_and_features(self):
        d, gm, y, c = graph()
        new, counts, ab = graph_intervention(d, c, (False, True, False))
        self.assertTrue(torch.equal(d.x, new.x))
        self.assertEqual(set(new.edge_type.tolist()), {1, 3, 4})
        self.assertTrue(torch.equal(d.edge_index[:, d.edge_type >= 3], new.edge_index[:, new.edge_type >= 3]))
        self.assertEqual(float(counts[:, [0, 2, 3, 4, 5, 6]].sum()), 0)

    def test_graph_propagation_ablation_disables_appnp(self):
        hp, ab = setup_component(HyperParams(), 'minus_rgcn')
        self.assertTrue(ab['rgcn'] and ab['appnp_off'])
        self.assertEqual(hp.appnp_K, 0)

    def test_bh_adjustment_matches_hand_calculation(self):
        np.testing.assert_allclose(bh_adjust([0.01, 0.04, 0.03, 0.2]), [0.04, 0.0533333333, 0.0533333333, 0.2])

    def test_multicategory_term_counts_are_not_gene_counts(self):
        x = pd.DataFrame({'Term': ['synapse immune', 'immune', 'other'], 'Adjusted P-value': [0.01, 0.1, 0.9]})
        y = mapped_enrichment(x, {'A': 'immune', 'B': 'synapse'}).set_index('category_key')
        self.assertEqual(y.loc['A', 'n_terms'], 2)
        self.assertEqual(y.loc['B', 'n_terms'], 1)
        self.assertAlmostEqual(y.loc['A', 'sum_neglog10_q'], 3)

    def test_jaccard_uses_positive_universe(self):
        m = category_jaccard({'a': {'x', 'y', 'outside'}, 'b': {'y', 'z', 'outside'}}, {'x', 'y', 'z'})
        self.assertAlmostEqual(m.loc['a', 'b'], 1 / 3)

    def test_positive_effect_vectors_match_saved_normalized_export(self):
        root = Path(__file__).resolve().parents[1] / 'results/effect_space'
        probs = pd.read_csv(root / 'per_gene_probs_raw.csv')
        _, norm, z = effect_space(probs)
        frozen = pd.read_csv(root / 'per_gene_probs_effects_normalized.csv')
        columns = ['effect_me', 'effect_h', 'effect_circ', 'drop_all', 'lift_only_me', 'lift_only_h', 'lift_only_circ']
        np.testing.assert_allclose(norm.sort_values('gene_unified_id')[columns], frozen.sort_values('gene_unified_id')[columns], atol=0.00011)
        self.assertEqual(len(norm), 4933)
        self.assertEqual(z.shape, (4933, 7))

    def test_grid_cli_uses_fixed_configuration(self):
        from unittest.mock import patch
        import main as entry
        with patch('hedge.data.load_csv_graph', return_value=object()), patch('hedge.training.run') as run, patch.object(sys, 'argv', ['main.py', 'grid']):
            entry.main()
        self.assertEqual(run.call_count, 40)
        settings = [call.args[1] for call in run.call_args_list]
        self.assertEqual({(h.appnp_K, h.num_layers, h.enable_convergence_gate) for h in settings}, {(k, l, g) for k in [1, 2, 4, 6, 8] for l in [2, 3, 4, 6] for g in [False, True]})
        self.assertTrue(all(((h.basis, h.attn_heads, h.appnp_alpha, h.edge_drop_rate, h.smooth_lambda) == (4, 2, 0.3, 0.1, 0.0) for h in settings)))

    def test_umap_duplicate_id_pairs_do_not_create_false_coordinate_difference(self):
        from hedge.reference import verify_umap_coordinate_sets
        a = pd.DataFrame({'gene_index': [1, 1, 2], 'umap_1': [0.0, 3.0, 5.0], 'umap_2': [2.0, 4.0, 6.0]})
        b = pd.concat([a, a[a.gene_index == 1]], ignore_index=True)
        b['umap_1'] = b.umap_1 + 5e-07
        result = verify_umap_coordinate_sets(b, a)
        self.assertTrue(result['coordinates_match_within_export_precision'])
        self.assertLessEqual(result['max_abs_matching_coordinate_difference'], 1e-06)
if __name__ == '__main__':
    unittest.main(verbosity=2)

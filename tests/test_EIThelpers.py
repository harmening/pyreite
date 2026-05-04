import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyreite.EIThelpers import (
    EIT_protocol,
    apply_protocol,
    build_full_protocol,
    compute_fim_from_J,
    compute_fim_from_blocks,
    d_optimality,
    greedy_select_blocks_doptimal,
    greedy_select_until_full,
    info_metric,
    scale_experiment_to_simulation,
)


def test_EIT_protocol():
    num_elec = np.random.randint(100)
    ND2V = EIT_protocol(num_elec, n_freq=1, protocol='all_realistic')
    assert np.sum(ND2V) == num_elec*(num_elec-1)/2 * (num_elec-2)
    assert len(ND2V) == num_elec**3
    n_freq = np.random.randint(10)
    ND2V = EIT_protocol(num_elec, n_freq=n_freq, protocol='all')
    assert len(ND2V) == np.sum(ND2V) == (n_freq*num_elec**3)


# ---- apply_protocol ----

def test_apply_protocol_basic():
    n_elec = 5
    V4 = np.arange(n_elec**3, dtype=float).reshape(1, n_elec, n_elec, n_elec)
    protocol = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    result = apply_protocol(V4, protocol)
    expected = np.array([
        V4[0, 0, 1, 2] - V4[0, 0, 1, 3],
        V4[0, 1, 2, 3] - V4[0, 1, 2, 4],
    ])
    assert_allclose(result, expected)

def test_apply_protocol_3d_input_promoted():
    V3 = np.random.rand(4, 4, 4)
    protocol = np.array([[0, 1, 2, 3]])
    out_3d = apply_protocol(V3, protocol)
    out_4d = apply_protocol(V3[np.newaxis], protocol)
    assert_allclose(out_3d, out_4d)


# ---- build_full_protocol ----

def test_build_full_protocol_shape_and_count():
    n = 5
    p = build_full_protocol(n)
    assert p.shape[1] == 4
    # C(n, 2) source pairs * C(n-2, 2) measurement pairs
    expected = (n * (n - 1) // 2) * ((n - 2) * (n - 3) // 2)
    assert p.shape[0] == expected

def test_build_full_protocol_row_validity():
    p = build_full_protocol(6)
    assert (p[:, 0] < p[:, 1]).all(), "source < sink required"
    assert (p[:, 2] < p[:, 3]).all(), "m1 < m2 required"
    for row in p:
        assert len(set(row.tolist())) == 4, "all four electrodes must be distinct"


# ---- scale_experiment_to_simulation ----

def test_scale_experiment_to_simulation_matches_moments():
    sim = np.random.randn(50) * 2.0 + 5.0
    exp = np.random.randn(50) * 0.3 - 1.0
    scaled = scale_experiment_to_simulation(exp, sim)
    assert_allclose(np.mean(scaled), np.mean(sim), atol=1e-10)
    assert_allclose(np.std(scaled), np.std(sim), atol=1e-10)


# ---- FIM helpers ----

def test_compute_fim_from_J():
    J = np.random.randn(20, 4)
    F = compute_fim_from_J(J, noise_var=2.0)
    assert F.shape == (4, 4)
    assert_allclose(F, J.T @ J / 2.0)
    assert_allclose(F, F.T)

def test_compute_fim_from_blocks_equals_concatenated():
    J1 = np.random.randn(10, 3)
    J2 = np.random.randn(15, 3)
    F_blocks = compute_fim_from_blocks([J1, J2], noise_var=1.5)
    F_full = compute_fim_from_J(np.vstack([J1, J2]), noise_var=1.5)
    assert_allclose(F_blocks, F_full)

def test_compute_fim_from_blocks_empty_raises():
    with pytest.raises(ValueError):
        compute_fim_from_blocks([])


# ---- d_optimality / info_metric ----

def test_d_optimality_identity_zero():
    assert_allclose(d_optimality(np.eye(3), reg=0), 0.0, atol=1e-10)

def test_d_optimality_singular_neg_inf():
    assert d_optimality(np.zeros((3, 3)), reg=0) == -np.inf

def test_info_metric_min_eig():
    F = np.diag([1.0, 5.0, 9.0])
    assert_allclose(info_metric(F, metric='min_eig', reg=0), 1.0)

def test_info_metric_logdet_matches_d_optimality():
    F = np.diag([2.0, 3.0, 5.0])
    assert_allclose(info_metric(F, metric='logdet', reg=0),
                    d_optimality(F, reg=0))

def test_info_metric_unknown_metric_raises():
    with pytest.raises(ValueError):
        info_metric(np.eye(2), metric='nonsense')


# ---- greedy block selection ----

def test_greedy_select_blocks_doptimal_basic():
    J_blocks = [np.random.randn(8, 3) * (i + 1) for i in range(5)]
    selected, F, history = greedy_select_blocks_doptimal(J_blocks, num_select=3)
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert len(history) == 3
    assert F.shape == (3, 3)

def test_greedy_select_blocks_doptimal_too_many_raises():
    J_blocks = [np.random.randn(5, 2) for _ in range(2)]
    with pytest.raises(ValueError):
        greedy_select_blocks_doptimal(J_blocks, num_select=5)

def test_greedy_select_until_full_stops_at_target():
    J_blocks = [np.random.randn(10, 3) for _ in range(8)]
    selected, F_sub, history, F_all, target_val = greedy_select_until_full(
        J_blocks, target_frac=0.95, metric='logdet'
    )
    assert F_all.shape == (3, 3)
    if history:
        last = history[-1]
        assert last['fraction_of_full'] >= 0.95 or len(selected) == len(J_blocks)

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from tests.data_for_testing import *
from pyreite.OpenMEEGHead import OpenMEEGHead, om2np
from pyreite.optimizers import *
from collections import OrderedDict


def moore_penrose(A, b):
	# Moore-Penrose generalized inverse
	x = np.dot(np.linalg.inv(A.conj().T.dot(A)), A.conj().T.dot(b))
	return x

def newton(J, H):
    new_J = np.zeros(J.shape) 
    for i in range(J.shape[0]):
        new_J[i,:] = J[i].dot(np.linalg.pinv(H[:,:,i]))
    return new_J

def head_from_bnds(bnds):
    mesh_names = ['bnd%d' % i for i in range(len(bnds))]
    geom = OrderedDict([(tissue, bnd) for tissue, bnd in zip(mesh_names, bnds)])
    cond = OrderedDict([(tissue, 1+np.random.rand()) for tissue in mesh_names])
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])
    head = OpenMEEGHead(cond, geom, elecs)
    return head


def add_noise(x):
    x_shape = x.shape
    x = x.flatten()
    # noise as in Malone 2014
    # proportional noise 
    std_dev_prop = (0.02/100) # 0.02%
    noise_prop = np.array([e*np.random.normal(loc=0.0, scale=np.abs(e)*\
                                                             std_dev_prop) \
                           for e in x])
    # additive noise 
    std_dev_add = 5 / 10**6 # 5 micro Volt
    noise_add = np.random.normal(loc=0.0, scale=std_dev_add)

    #print(x)
    for ii in range(len(x)):
        if x[ii] != 0.0:
            x[ii] += noise_prop[ii] + noise_add
    #print(x)
    x = x.reshape(x_shape)
    return x


def test_loss_residuals():
    bnds = simple_test_shapes(num_nested_meshes=2)
    head = head_from_bnds(bnds)
    V_experiment = add_noise(head.V)
    ND2V = EIT_protocol(num_elec=head.n_electrodes, n_freq=1, \
                        protocol='all_realistic')
    V_experiment = V_experiment.flatten()[ND2V]
    dV = loss_residuals(head.cond, head, V_experiment)
    print(np.sum(dV))
    print(np.min(dV), np.max(dV), np.std(dV))
    assert np.sum(np.abs(dV))

def test_jac():
    bnds = simple_test_shapes(num_nested_meshes=2)
    head = head_from_bnds(bnds)
    J = jac(head.cond, head, None)
    assert (J != 0.0).all()
    head = head_from_bnds(bnds)
    new_J = jac(head.cond, head, None)
    assert (J != new_J).all()

def test_hess():
    bnds = simple_test_shapes(num_nested_meshes=2)
    head = head_from_bnds(bnds)
    H = hess(head.cond, head, None)
    assert (H != 0.0).all()
    head = head_from_bnds(bnds)
    new_H = hess(head.cond, head, None)
    assert (H != new_H).all()

def test_jac_hess():
    bnds = simple_test_shapes(num_nested_meshes=2)
    head = head_from_bnds(bnds)
    J, H = jac_hess(head.cond, head, None)
    assert (J != 0.0).all()
    assert (H != 0.0).all()
    head = head_from_bnds(bnds)
    new_J, new_H = jac_hess(head.cond, head, None)
    assert (J != new_J).all()
    assert (H != new_H).all()

def test_levenberg_marquardt_hessian():
    bnds = simple_test_shapes(num_nested_meshes=2)
    head = head_from_bnds(bnds)
    V_experiment = add_noise(head.V)
    ND2V = EIT_protocol(num_elec=head.n_electrodes, n_freq=1, \
                        protocol='all_realistic')
    V_experiment = V_experiment.flatten()[ND2V]
    b = loss_residuals(head.cond, head, V_experiment)
    A = jac(head.cond, head, None)
    dAzeros = np.zeros((len(bnds), len(bnds), len(b)))
    lamb = 0.1
    Lpr0 = np.identity(len(bnds))
    diff = levenberg_marquardt_hessian(A, dAzeros, b, lamb, Lpr0)
    diff2 = levenberg_marquardt_hessiancheck(A, dAzeros, b, lamb, Lpr0)
    lma = tikhonov(A, b, lamb, Lpr0)
    assert_array_almost_equal(lma, diff)
    assert_array_almost_equal(lma, diff2)


# ---- pure-numeric tests for the regularization / damping helpers ----

def test_tikhonov_zero_lambda_matches_lstsq():
    A = np.random.randn(10, 3)
    b = np.random.randn(10)
    Lpr0 = np.eye(3)
    x_tik = tikhonov(A, b, lamb=0.0, Lpr0=Lpr0)
    x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert_allclose(x_tik, x_lstsq, atol=1e-10)

def test_tikhonov_large_lambda_shrinks_to_zero():
    A = np.random.randn(10, 3)
    b = np.random.randn(10)
    Lpr0 = np.eye(3)
    x = tikhonov(A, b, lamb=1e6, Lpr0=Lpr0)
    assert np.linalg.norm(x) < 1e-3

def test_levenberg_marquardt_prior_centered_at_mu_equals_tikhonov():
    """When x == mu and dA == 0, prior centering and Hessian terms vanish."""
    M, P = 8, 3
    A = np.random.randn(M, P)
    b = np.random.randn(M)
    dA = np.zeros((P, P, M))  # per-measurement Hessian; all-zero -> no 2nd-order
    x = np.random.randn(P)
    mu = x.copy()
    Lpr0 = np.eye(P)
    step = levenberg_marquardt_hessiancheck_prior_centered(
        A, dA, b, lamb=0.1, Lpr0=Lpr0, x=x, mu=mu
    )
    expected = tikhonov(A, b, lamb=0.1, Lpr0=Lpr0)
    assert_allclose(step, expected, atol=1e-10)

def test_levenberg_marquardt_prior_centered_accepts_vector_Lpr0():
    A = np.random.randn(8, 3)
    b = np.random.randn(8)
    dA = np.zeros((3, 3, 8))
    x = np.zeros(3)
    mu = np.zeros(3)
    step_vec = levenberg_marquardt_hessiancheck_prior_centered(
        A, dA, b, 0.1, np.array([1.0, 1.0, 1.0]), x, mu
    )
    step_mat = levenberg_marquardt_hessiancheck_prior_centered(
        A, dA, b, 0.1, np.eye(3), x, mu
    )
    assert_allclose(step_vec, step_mat)

def test_levenberg_marquardt_noser_shape():
    A = np.random.randn(10, 4)
    dA = np.zeros((4, 4, 10))   # per-measurement Hessian, shape (P, P, M)
    b = np.random.randn(10)
    out = levenberg_marquardt_hessian_noser(A, dA, b, lamb=0.1)
    assert out.shape == (4,)

def test_build_Lpr0_from_J_matches_formula():
    J = np.random.randn(10, 3)
    Lpr0, diagJ = build_Lpr0_from_J(J)
    expected_diagJ = np.maximum(np.einsum("ij,ij->j", J, J), 1e-12)
    assert_allclose(diagJ, expected_diagJ)
    assert_allclose(Lpr0, 1.0 / np.sqrt(diagJ))

def test_build_Lpr0_from_J_floor_protects_zero_J():
    J = np.zeros((10, 3))
    _, diagJ = build_Lpr0_from_J(J, floor=1e-3)
    assert_allclose(diagJ, np.full(3, (1e-3) ** 2))

def test_build_Lpr0_from_J_max_with_prev_diag():
    J = np.ones((5, 3))
    prev_diag = np.array([100.0, 0.0, 1.0])
    _, diagJ = build_Lpr0_from_J(J, prev_diag=prev_diag)
    assert_allclose(diagJ, np.array([100.0, 5.0, 5.0]))

def test_build_Lpr0_combined_matches_formula():
    J = np.random.randn(8, 2)
    sigma_x = np.array([0.1, 0.5])
    Lambda_total, diagJ = build_Lpr0_combined(J, sigma_x)
    expected_sens = 1.0 / np.maximum(np.einsum("ij,ij->j", J, J), 1e-12)
    expected_prior = 1.0 / sigma_x ** 2
    assert_allclose(Lambda_total, expected_sens + expected_prior)


import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from tests.data_for_testing import *
from pyreite.OpenMEEGHead import OpenMEEGHead, om2np
from pyreite.material_derivative import *
from collections import OrderedDict


def head_from_bnds(bnds):
    mesh_names = ['bnd%d' % i for i in range(len(bnds))]
    geom = OrderedDict([(shell, bnd) for shell, bnd in zip(mesh_names, bnds)])
    cond = OrderedDict([(shell, 1+np.random.rand()) for shell in mesh_names])
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])[::800,:] # 7 elecs
    head = OpenMEEGHead(cond, geom, elecs)
    return head

def small_cond_pertubation(head, mesh_nb):
    f_x = head.A
    cond = head.cond
    # small pertubation
    e = {shell: 0.0 for shell in cond.keys()}
    eps = np.sqrt(10**(-12))
    e['bnd%d' % mesh_nb] = eps
    new_cond = {shell: cond[shell]+e[shell] for shell in head.mesh_names}
    #head_eps = OpenMEEGHead(new_cond, geom, elecs)
    # head = OpenMEEGHead(new_cond, geom, elecs)
    #f_x_eps = head_eps.A
    head.set_cond(new_cond)
    f_x_eps = head.A
    diff_right = (f_x - f_x_eps) / eps

    new_cond = {shell: cond[shell]-e[shell] for shell in head.mesh_names}
    head.set_cond(new_cond) ##
    f_x_eps = head.A ##
    diff_left = (f_x_eps - f_x) / eps

    diff = -(diff_right+diff_left)/2
    return diff


def test_first_derivatives():
    num_meshes = np.random.randint(1, 4)
    bnds = simple_test_shapes(num_nested_meshes=num_meshes)
    mesh_names = ['bnd%d' % i for i in range(len(bnds))]
    geom = OrderedDict([(shell, bnd) for shell, bnd in zip(mesh_names, bnds)])
    cond = OrderedDict([(shell, 1+np.random.rand()) for shell in mesh_names])
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])[::800,:] #7 elecs
    head = OpenMEEGHead(cond, geom, elecs)
    derivs = first_derivatives(head, elecs, head.A, head.Ainv, head.h2em, \
                               head.eitsm, head.ind)
    assert len(derivs) == 4
    dEIT, dV4dsi, dA1dsi, dadsi = derivs
    if num_meshes == 1:
        assert isinstance(dV4dsi, np.ndarray)
        assert isinstance(dA1dsi, np.ndarray)
        assert isinstance(dadsi, np.ndarray)
    else:
        assert len(dV4dsi) == num_meshes
        assert len(dA1dsi) == num_meshes
        assert len(dadsi) == num_meshes


def test_second_derivatives():
    num_meshes = np.random.randint(1, 4)
    bnds = simple_test_shapes(num_nested_meshes=num_meshes)
    mesh_names = ['bnd%d' % i for i in range(len(bnds))]
    geom = OrderedDict([(shell, bnd) for shell, bnd in zip(mesh_names, bnds)])
    cond = OrderedDict([(shell, 1+np.random.rand()) for shell in mesh_names])
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])[::800,:] #7 elecs
    head = OpenMEEGHead(cond, geom, elecs)
    dEIT, _, dA1dsi, dadsi = first_derivatives(head, elecs, head.A, head.Ainv,\
                                               head.h2em, head.eitsm, head.ind)
    d2Vdsidsj = second_derivatives(head, elecs, head.A, head.Ainv, head.h2em,
                                   head.eitsm, head.ind, dEIT, dA1dsi, dadsi)
    assert len(d2Vdsidsj) == num_meshes**2

def test_dAds1():
    bnds = simple_test_shapes(num_nested_meshes=5)[-2:]
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names] 
    dads1 = dAds1(cond_list, head.ind, head.A)
    diff = small_cond_pertubation(head, mesh_nb=0)
    assert np.sum(np.abs(diff)) != 0
    assert_array_almost_equal(dads1[np.ix_(head.ind['V'][0], head.ind['V'][0])],
                              diff[np.ix_(head.ind['V'][0], head.ind['V'][0])],
                              6)


def test_dAds2():
    bnds = simple_test_shapes(num_nested_meshes=5)[-4:]
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names] 
    dads2 = dAds2(cond_list, head.ind, head.A)
    diff = small_cond_pertubation(head, mesh_nb=1)
    assert np.sum(np.abs(diff)) != 0
    assert_array_almost_equal(dads2[np.ix_(head.ind['V'][1], head.ind['V'][1])],
                              diff[np.ix_(head.ind['V'][1], head.ind['V'][1])],
                              6)


def test_dAds3():
    bnds = fast_simple_test_shapes(num_nested_meshes=8)[-4:]
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names] 
    dads3 = dAds3(cond_list, head.ind, head.A)
    diff = small_cond_pertubation(head, mesh_nb=2)
    assert np.sum(np.abs(diff)) != 0
    assert_array_almost_equal(dads3[np.ix_(head.ind['p'][2], head.ind['p'][2])],
                              diff[np.ix_(head.ind['p'][2], head.ind['p'][2])],
                              6)


def test_dAds4():
    bnds = fast_simple_test_shapes(num_nested_meshes=8)[-5:]
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names] 
    dads4 = dAds4(cond_list, head.ind, head.A)
    diff = small_cond_pertubation(head, mesh_nb=3)
    assert np.sum(np.abs(diff)) != 0
    assert_array_almost_equal(dads4[np.ix_(head.ind['p'][3], head.ind['p'][3])],
                              diff[np.ix_(head.ind['p'][3], head.ind['p'][3])],
                              6)

# 2nd
def test_dAds1ds1():
    bnds = simple_test_shapes(num_nested_meshes=2)
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names] 
    ind = head.ind
    dads1 = dAds1(cond_list, ind, head.A)
    da2ds1ds1 = dAds1ds1(cond_list, ind, dads1)

    da2ds1ds1_from_hm = np.zeros((head.A.shape), dtype='float64')
    S11 = head.A[np.ix_(ind['p'][0], ind['p'][0])] / (1/cond_list[0] + \
                                                      1/cond_list[1])
    da2ds1ds1_from_hm[np.ix_(ind['p'][0],ind['p'][0])] = 2 * cond_list[0]**(-3) * S11

    assert_array_almost_equal(da2ds1ds1, da2ds1ds1_from_hm)

def test_dAds2ds2():
    bnds = simple_test_shapes(num_nested_meshes=4)
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names]
    ind = head.ind
    dads2 = dAds2(cond_list, ind, head.A)
    da2ds2ds2 = dAds2ds2(cond_list, ind, dads2)

    da2ds2ds2_from_hm = np.zeros((head.A.shape), dtype='float64')
    S11 = head.A[np.ix_(ind['p'][0], ind['p'][0])] / (1/cond_list[0] + \
                                                      1/cond_list[1])
    da2ds2ds2_from_hm[np.ix_(ind['p'][0],ind['p'][0])] = 2 * cond_list[1]**(-3) * S11
    S12 = head.A[np.ix_(ind['p'][0], ind['p'][1])] * (-cond_list[1])
    da2ds2ds2_from_hm[np.ix_(ind['p'][0],ind['p'][1])] = -2 * cond_list[1]**(-3) * S12
    S21 = head.A[np.ix_(ind['p'][1], ind['p'][0])] * (-cond_list[1])
    da2ds2ds2_from_hm[np.ix_(ind['p'][1],ind['p'][0])] = -2 * cond_list[1]**(-3) * S21
    S22 = head.A[np.ix_(ind['p'][1], ind['p'][1])] / (1/cond_list[1] + \
                                                      1/cond_list[2])
    da2ds2ds2_from_hm[np.ix_(ind['p'][1],ind['p'][1])] = 2 * cond_list[1]**(-3) * S22

    assert_array_almost_equal(da2ds2ds2, da2ds2ds2_from_hm)

def test_dAds3ds3():
    bnds = simple_test_shapes(num_nested_meshes=4)
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names]
    ind = head.ind
    dads3 = dAds3(cond_list, ind, head.A)
    da2ds3ds3 = dAds3ds3(cond_list, ind, dads3)

    da2ds3ds3_from_hm = np.zeros((head.A.shape), dtype='float64')
    S22 = head.A[np.ix_(ind['p'][1], ind['p'][1])] / (1/cond_list[1] + \
                                                      1/cond_list[2])
    da2ds3ds3_from_hm[np.ix_(ind['p'][1],ind['p'][1])] = 2 * cond_list[2]**(-3) * S22
    S23 = head.A[np.ix_(ind['p'][1], ind['p'][2])] * (-cond_list[2])
    da2ds3ds3_from_hm[np.ix_(ind['p'][1],ind['p'][2])] = -2 * cond_list[2]**(-3) * S23
    S32 = head.A[np.ix_(ind['p'][2], ind['p'][1])] * (-cond_list[2])
    da2ds3ds3_from_hm[np.ix_(ind['p'][2],ind['p'][1])] = -2 * cond_list[2]**(-3) * S32
    S33 = head.A[np.ix_(ind['p'][2], ind['p'][2])] / (1/cond_list[2] + \
                                                      1/cond_list[3])
    da2ds3ds3_from_hm[np.ix_(ind['p'][2],ind['p'][2])] = 2 * cond_list[2]**(-3) * S33

    assert_array_almost_equal(da2ds3ds3, da2ds3ds3_from_hm)

def test_dAds4ds4():
    bnds = simple_test_shapes(num_nested_meshes=4)
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names]
    ind = head.ind
    dads4 = dAds4(cond_list, ind, head.A)
    da2ds4ds4 = dAds4ds4(cond_list, ind, dads4)

    da2ds4ds4_from_hm = np.zeros((head.A.shape), dtype='float64')
    S33 = head.A[np.ix_(ind['p'][2], ind['p'][2])] / (1/cond_list[2] + \
                                                      1/cond_list[3])
    da2ds4ds4_from_hm[np.ix_(ind['p'][2],ind['p'][2])] = 2 * cond_list[3]**(-3) * S33

    assert_array_almost_equal(da2ds4ds4, da2ds4ds4_from_hm)


def test_jacobian():
    bnds = simple_test_shapes(num_nested_meshes=4)
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names]
    j = jacobian(cond_list, head, return_model=False)
    assert len(j.shape) == 2
    assert j.shape[1] == 4


@pytest.mark.parametrize("num_meshes", [2, 3])
def test_jacobian_finite_difference(num_meshes):
    """Validate analytic jacobian against central-difference numerical jacobian.

    Catches sign flips, tissue-index drift, and wrong-branch selection in the
    per-mesh-count code paths of `material_derivative.jacobian`.
    """
    bnds = simple_test_shapes(num_nested_meshes=num_meshes)
    mesh_names = ['bnd%d' % i for i in range(num_meshes)]
    geom = OrderedDict([(s, b) for s, b in zip(mesh_names, bnds)])
    cond = OrderedDict([(s, 1 + np.random.rand()) for s in mesh_names])
    # Use full face-center grid on outermost shell so EIT_protocol returns
    # a non-empty measurement set (head_from_bnds's [::800] slice gives
    # only 1 electrode for basic icosahedrons -> 0 measurements).
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])[::4, :]
    head = OpenMEEGHead(cond, geom, elecs)

    n_elec = head.n_electrodes
    ND2V = EIT_protocol(num_elec=n_elec, n_freq=1, protocol='all_realistic')
    assert sum(ND2V) > 0, "FD test requires non-empty measurement protocol"

    # Analytic jacobian: shape (n_meas, num_meshes), columns inside-out.
    J_analytic = jacobian(cond, head, ND2V=ND2V)

    # Numerical jacobian via central differences. Fresh OpenMEEGHead per
    # perturbation to avoid any reliance on set_cond cache invalidation.
    eps = 1e-4
    J_fd = np.zeros_like(J_analytic)
    tissues_inside_out = list(reversed(head.mesh_names))
    for i, tissue in enumerate(tissues_inside_out):
        cond_plus = OrderedDict((t, cond[t] + (eps if t == tissue else 0.0))
                                for t in cond)
        cond_minus = OrderedDict((t, cond[t] - (eps if t == tissue else 0.0))
                                 for t in cond)
        head_plus = OpenMEEGHead(cond_plus, geom, elecs)
        head_minus = OpenMEEGHead(cond_minus, geom, elecs)
        V_plus = head_plus.V.flatten()[ND2V]
        V_minus = head_minus.V.flatten()[ND2V]
        J_fd[:, i] = (V_plus - V_minus) / (2 * eps)

    # All columns must agree to near-machine precision.
    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-4, atol=1e-9)

def test_hessian():
    bnds = simple_test_shapes(num_nested_meshes=4)
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names] 
    h = hessian(cond_list, head)
    assert len(h.shape) == 3
    assert h.shape[0] == h.shape[1] == 4


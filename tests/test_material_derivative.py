import numpy as np
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
    eps = np.sqrt(pow(10, -12))
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


def test_EIT_protocol():
    num_elec = np.random.randint(100)
    ND2V = EIT_protocol(num_elec, n_freq=1, protocol='all_realistic')
    assert np.sum(ND2V) == num_elec*(num_elec-1)/2 * (num_elec-2)
    assert len(ND2V) == pow(num_elec, 3)
    n_freq = np.random.randint(10)
    ND2V = EIT_protocol(num_elec, n_freq=n_freq, protocol='all')
    assert len(ND2V) == np.sum(ND2V) == (n_freq*pow(num_elec, 3))


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
    assert len(d2Vdsidsj) == pow(num_meshes, 2)

"""
def test_jacobian():
    #check_grad - Check the supplied derivative using finite differences.
    # -> not working due to ill-posedness???
    bnds = simple_test_shapes(num_nested_meshes=4)[-2:]
    mesh_names = ['bnd%d' % i for i in range(len(bnds))]
    geom = OrderedDict([(shell, bnd) for shell, bnd in zip(mesh_names, bnds)])
    cond = OrderedDict([(shell, 1+np.random.rand()) for shell in mesh_names])
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])[::800,:] #7 elecs
    head = OpenMEEGHead(cond, geom, elecs)
    orig_cond = [head.cond[shell] for shell in head.mesh_names] # inside out
    J = jacobian(orig_cond, head).T
    ND2V = EIT_protocol(elecs.shape[0], protocol='all_realistic')
    eps = np.sqrt(pow(10, -12))
    f_x = head.V[0]#.flatten()[ND2V]
    fin_diff = []
    for i, mesh_name in enumerate(cond.keys()):
        e = {shell: 0.0 for shell in cond.keys()}
        e[mesh_name] = eps
        new_cond = OrderedDict([(shell, cond[shell]+e[shell]) for shell in \
                                head.mesh_names])
        head_plus = OpenMEEGHead(new_cond, geom, elecs)
        #head.set_cond(new_cond)
        #act_cond = [head_plus.cond[shell] for shell in head_plus.mesh_names] 
        f_x_plus_eps = head_plus.V[0]#.flatten()[ND2V]

        new_cond = OrderedDict([(shell, cond[shell]-e[shell]) for shell in \
                                head.mesh_names])
        head_minus = OpenMEEGHead(new_cond, geom, elecs)
        act_cond = [head_minus.cond[shell] for shell in head_minus.mesh_names]
        f_x_minus_eps = head_minus.V[0]#.flatten()[ND2V]

        diff = (f_x_plus_eps - f_x_minus_eps)/(2*eps)
        fin_diff.append(diff.flatten()[ND2V])
        #head.set_cond(cond)
        
        ##dV /  |V|  approx  J_condition_nb  *  J-J_perturbed / |J| +   sigma - (sigma+eps) / |sigma|
        #new_cond_values = [head_minus.cond[shell] for shell in head_minus.mesh_names] 
        #J_eps = jacobian(new_cond_values, head_minus, None).T
        #V_diff = deltaV(act_cond, head_minus, f_x)
        #left = V_diff / np.linalg.norm(V_diff[i])
        #right = (J[i]-J_eps)/np.linalg.norm(J[i]) + \
        #        (np.array(orig_cond[i])-np.array(new_cond_values[i]))/np.linalg.norm(orig_cond[i])
        #print(left)
        #print(np.linalg.norm(J[i])*right)
        #assert_array_almost_equal(left, np.linalg.norm(J[i])*right)
        del head_plus
        del head_minus

    fin_diff = np.array(fin_diff)
    assert np.sum(fin_diff) != 0.0
"""

"""
# Testing the perturbation with taking the condition number into account
dads1_eps = dAds1(new_cond_list, head_eps.ind, om2np(head_eps.A))
left = om2np(f_x - f_x_eps) / np.linalg.norm(om2np(head.A))
right = (dads1 - dads1_eps) / np.linalg.norm(dads1) + (cond_list[0]-new_cond_list[0]) / np.linalg.norm(cond_list[0])
condition_nb = np.linalg.cond(dads1)
if abs(condition_nb) == np.inf:
    condition_nb = 1
assert_array_almost_equal(left, condition_nb*right)
"""

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
    da2ds1ds1_from_hm[np.ix_(ind['p'][0],ind['p'][0])] = 2 * pow(cond_list[0],\
                                                                 -3) * S11

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
    da2ds2ds2_from_hm[np.ix_(ind['p'][0],ind['p'][0])] = 2 * pow(cond_list[1],\
                                                                 -3) * S11
    S12 = head.A[np.ix_(ind['p'][0], ind['p'][1])] * (-cond_list[1])
    da2ds2ds2_from_hm[np.ix_(ind['p'][0],ind['p'][1])] = -2 * pow(cond_list[1],\
                                                                  -3) * S12
    S21 = head.A[np.ix_(ind['p'][1], ind['p'][0])] * (-cond_list[1])
    da2ds2ds2_from_hm[np.ix_(ind['p'][1],ind['p'][0])] = -2 * pow(cond_list[1],\
                                                                  -3) * S21
    S22 = head.A[np.ix_(ind['p'][1], ind['p'][1])] / (1/cond_list[1] + \
                                                      1/cond_list[2])
    da2ds2ds2_from_hm[np.ix_(ind['p'][1],ind['p'][1])] = 2 * pow(cond_list[1],\
                                                                 -3) * S22

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
    da2ds3ds3_from_hm[np.ix_(ind['p'][1],ind['p'][1])] = 2 * pow(cond_list[2],\
                                                                 -3) * S22
    S23 = head.A[np.ix_(ind['p'][1], ind['p'][2])] * (-cond_list[2])
    da2ds3ds3_from_hm[np.ix_(ind['p'][1],ind['p'][2])] = -2 * pow(cond_list[2],\
                                                                  -3) * S23
    S32 = head.A[np.ix_(ind['p'][2], ind['p'][1])] * (-cond_list[2])
    da2ds3ds3_from_hm[np.ix_(ind['p'][2],ind['p'][1])] = -2 * pow(cond_list[2],\
                                                                  -3) * S32
    S33 = head.A[np.ix_(ind['p'][2], ind['p'][2])] / (1/cond_list[2] + \
                                                      1/cond_list[3])
    da2ds3ds3_from_hm[np.ix_(ind['p'][2],ind['p'][2])] = 2 * pow(cond_list[2],\
                                                                 -3) * S33

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
    da2ds4ds4_from_hm[np.ix_(ind['p'][2],ind['p'][2])] = 2 * pow(cond_list[3],\
                                                                 -3) * S33

    assert_array_almost_equal(da2ds4ds4, da2ds4ds4_from_hm)


def test_jacobian():
    bnds = simple_test_shapes(num_nested_meshes=4)
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names] 
    j = jacobian(cond_list, head, return_model=False)
    assert len(j.shape) == 2
    assert j.shape[1] == 4

def test_hessian():
    bnds = simple_test_shapes(num_nested_meshes=4)
    head = head_from_bnds(bnds)
    cond_list = [head.cond[shell] for shell in head.mesh_names] 
    h = hessian(cond_list, head)
    assert len(h.shape) == 3
    assert h.shape[0] == h.shape[1] == 4


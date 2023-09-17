import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
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
    std_dev_add = 5 / pow(10,6) # 5 micro Volt
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


from __future__ import print_function
from collections import OrderedDict
import itertools
import numpy as np, openmeeg as om
GAUSS_ORDER = 3


def EIT_protocol(num_elec, n_freq=1, protocol='all'):
    if protocol == 'all':
        return [True for _ in range(n_freq*num_elec*num_elec*num_elec)]
    if protocol == 'all_realistic':
        ND2V = []
        for Source, Sink in itertools.product(range(num_elec), range(num_elec)):
            if Source < Sink:
                for measure in range(num_elec):
                    if measure not in [Source, Sink]:
                        ND2V.append(True)
                    else:
                        ND2V.append(False)
            else:
                for measure in range(num_elec):
                    ND2V.append(False)
        assert np.sum(ND2V) == num_elec*(num_elec-1)/2 * (num_elec-2)
        assert len(ND2V) == num_elec*num_elec*num_elec
        return ND2V



def first_derivatives(head, electrodes, hm, hminv, h2em, eitsm, ind):
    # Conductivity values for meshes (from out to inside)
    cond = [head.cond[tissue] for tissue in reversed(head.mesh_names)] #inside out
    #cond = [head.cond[tissue] for tissue in head.mesh_names] #outside in
    num_meshes = len(cond)

    # Derivative of the EIT right hand side
    dEIT = np.zeros((eitsm.shape))
    #dEIT[ind['p'][-2],:] = -pow(cond[-1], -1) * eitsm[ind['p'][-2],:] # S23
    #dEIT[ind['p'][-1],:] = -pow(cond[-1], -2) * eitsm[ind['p'][-1],:] # S23
    dEIT[ind['p'][-1],:] = -eitsm[ind['p'][-1],:] / cond[-1] # S23

    # can be precomputed
    dCds = hminv.dot(dEIT)

    #dV4ds1
    dads1 = dAds1(cond, ind, hm)
    dA1ds1 = -hminv.dot(dads1.dot(hminv)) # needed for hessian
    dV4ds1 = dA1ds1.dot(eitsm)
    dV4ds1 += dCds if num_meshes == 1 else 0
    dV4ds1 = h2em.dot(dV4ds1)

    #dV4ds2
    if num_meshes > 1:
        dads2 = dAds2(cond, ind, hm)
        dA1ds2 = -hminv.dot(dads2.dot(hminv)) # needed for hessian
        dV4ds2 = dA1ds2.dot(eitsm)
        dV4ds2 += dCds if num_meshes == 2 else 0
        dV4ds2 = h2em.dot(dV4ds2)
    else:
        head.first_derivatives = (dEIT, (dV4ds1), (dA1ds1), (dads1))
        return dEIT, (dV4ds1), (dA1ds1), (dads1)
    #dV4ds3
    if num_meshes > 2:
        dads3 = dAds3(cond, ind, hm)
        dA1ds3 = -hminv.dot(dads3.dot(hminv)) # needed for hessian
        dV4ds3 = dA1ds3.dot(eitsm)
        dV4ds3 += dCds if num_meshes == 3 else 0
        dV4ds3 = h2em.dot(dV4ds3)
    else:
        head.first_derivatives = (dEIT, (dV4ds1, dV4ds2), (dA1ds1, dA1ds2), (dads1, dads2))
        return dEIT, (dV4ds1, dV4ds2), (dA1ds1, dA1ds2), (dads1, dads2)
    #dV4ds4
    if num_meshes > 3:
        dads4 = dAds4(cond, ind, hm)
        dA1ds4 = -hminv.dot(dads4.dot(hminv)) # needed for hessian
        dV4ds4 = dA1ds4.dot(eitsm)
        dV4ds4 += dCds if num_meshes == 4 else 0
        dV4ds4 = h2em.dot(dV4ds4)
    else:
        head.first_derivatives = (dEIT, (dV4ds1, dV4ds2, dV4ds3), (dA1ds1, dA1ds2, dA1ds3), \
                                  (dads1, dads2, dads3))
        return dEIT, (dV4ds1, dV4ds2, dV4ds3), (dA1ds1, dA1ds2, dA1ds3), \
               (dads1, dads2, dads3)
    if num_meshes > 4:
        raise NotImplementedError
    else:
        return dEIT, (dV4ds1, dV4ds2, dV4ds3, dV4ds4), (dA1ds1, dA1ds2, dA1ds3,
                dA1ds4), (dads1, dads2, dads3, dads4)


def second_derivatives(head, electrodes, hm, hminv, h2em, eitsm, ind, dEIT,
                       dA1ds, dads):
    cond = [head.cond[tissue] for tissue in reversed(head.mesh_names)] # inside out
    #cond = [head.cond[tissue] for tissue in head.mesh_names] # outside to inside
    num_meshes = len(cond)
    if num_meshes == 1:
        dA1ds1 = dA1ds
        dads1 = dads
    elif num_meshes == 2:
        dA1ds1, dA1ds2 = dA1ds
        dads1, dads2 = dads
    elif num_meshes == 3:
        dA1ds1, dA1ds2, dA1ds3 = dA1ds
        dads1, dads2, dads3 = dads
    elif num_meshes == 4:
        dA1ds1, dA1ds2, dA1ds3, dA1ds4 = dA1ds
        dads1, dads2, dads3, dads4 = dads
    else:
        raise NotImplementedError

    d2EIT = -2 * dEIT / cond[-1]

    # can be precomputed
    hminveitsm = hminv.dot(eitsm)

    da2ds1ds1 = dAds1ds1(cond, ind, dads1)

    #d2Vds
    d2Vds1ds1 = (-2*dA1ds1.dot(dads1)-hminv.dot(da2ds1ds1)).dot(hminveitsm)
    d2Vds1ds1 += - dA1ds1.dot(dEIT) + dA1ds1.dot(dEIT) + hminv.dot(d2EIT) if \
                num_meshes == 1 else 0
    d2Vds1ds1 = h2em.dot(d2Vds1ds1)
    if num_meshes > 1:
        da2ds2ds2 = dAds2ds2(cond, ind, dads2)
        d2Vds1ds2 = (- dA1ds1.dot(dads2)-dA1ds2.dot(dads1)).dot(hminveitsm)
        d2Vds1ds2 += dA1ds1.dot(dEIT) + 0 + 0 if num_meshes == 2 else 0
        d2Vds1ds2 = h2em.dot(d2Vds1ds2)
        d2Vds2ds1 = (- dA1ds2.dot(dads1)-dA1ds1.dot(dads2)).dot(hminveitsm)
        d2Vds2ds1 += 0 + dA1ds1.dot(dEIT) + 0 if num_meshes == 2 else 0
        d2Vds2ds1 = h2em.dot(d2Vds2ds1)
        d2Vds2ds2 = (-2*dA1ds2.dot(dads2)-hminv.dot(da2ds2ds2)).dot(hminveitsm)
        d2Vds2ds2 += dA1ds2.dot(dEIT) + dA1ds2.dot(dEIT) + hminv.dot(d2EIT) if \
                    num_meshes == 2 else 0
        d2Vds2ds2 = h2em.dot(d2Vds2ds2)
    else:
        return d2Vds1ds1
    if num_meshes > 2:
        da2ds3ds3 = dAds3ds3(cond, ind, dads3)
        d2Vds1ds3 = (-dA1ds1.dot(dads3)-dA1ds3.dot(dads1)).dot(hminveitsm)
        d2Vds1ds3 += dA1ds1.dot(dEIT) + 0 + 0 if num_meshes == 3 else 0
        d2Vds1ds3 = h2em.dot(d2Vds1ds3)
        d2Vds3ds1 = (-dA1ds3.dot(dads1)-dA1ds1.dot(dads3)).dot(hminveitsm)
        d2Vds3ds1 += 0 + dA1ds1.dot(dEIT) + 0 if num_meshes == 3 else 0
        d2Vds3ds1 = h2em.dot(d2Vds3ds1)
        d2Vds2ds3 = (-dA1ds2.dot(dads3)-dA1ds3.dot(dads2)).dot(hminveitsm)
        d2Vds2ds3 += dA1ds2.dot(dEIT) + 0 + 0 if num_meshes == 3 else 0
        d2Vds2ds3 = h2em.dot(d2Vds2ds3)
        d2Vds3ds2 = (-dA1ds3.dot(dads2)-dA1ds2.dot(dads3)).dot(hminveitsm)
        d2Vds3ds2 += 0 + dA1ds2.dot(dEIT) + 0 if num_meshes == 3 else 0
        d2Vds3ds2 = h2em.dot(d2Vds3ds2)
        d2Vds3ds3 = (-2*dA1ds3.dot(dads3)-hminv.dot(da2ds3ds3)).dot(hminveitsm)
        d2Vds3ds3 += dA1ds3.dot(dEIT) + dA1ds3.dot(dEIT) + \
                     hminv.dot(d2EIT) if num_meshes == 3 else 0
        d2Vds3ds3 = h2em.dot(d2Vds3ds3)
    else:
        return (d2Vds1ds1, d2Vds1ds2, d2Vds2ds1, d2Vds2ds2)
    if num_meshes > 3:
        da2ds4ds4 = dAds4ds4(cond, ind, dads4)
        d2Vds1ds4 = (-dA1ds1.dot(dads4)-dA1ds4.dot(dads1)).dot(hminveitsm)
        d2Vds1ds4 += dA1ds1.dot(dEIT) + 0 + 0 if num_meshes == 4 else 0
        d2Vds1ds4 = h2em.dot(d2Vds1ds4)
        d2Vds4ds1 = (-dA1ds4.dot(dads1)-dA1ds1.dot(dads4)).dot(hminveitsm)
        d2Vds4ds1 += 0 + dA1ds1.dot(dEIT) + 0 if num_meshes == 4 else 0
        d2Vds4ds1 = h2em.dot(d2Vds4ds1)
        d2Vds2ds4 = (-dA1ds2.dot(dads4)-dA1ds4.dot(dads2)).dot(hminveitsm)
        d2Vds2ds4 += dA1ds2.dot(dEIT) + 0 + 0 if num_meshes == 4 else 0
        d2Vds2ds4 = h2em.dot(d2Vds2ds4)
        d2Vds4ds2 = (-dA1ds4.dot(dads2)-dA1ds2.dot(dads4)).dot(hminveitsm)
        d2Vds4ds2 += 0 + dA1ds2.dot(dEIT) + 0 if num_meshes == 4 else 0
        d2Vds4ds2 = h2em.dot(d2Vds4ds2)
        d2Vds3ds4 = (-dA1ds3.dot(dads4)-dA1ds4.dot(dads3)).dot(hminveitsm)
        d2Vds3ds4 += dA1ds3.dot(dEIT) + 0 + 0 if num_meshes == 4 else 0
        d2Vds3ds4 = h2em.dot(d2Vds3ds4)
        d2Vds4ds3 = (-dA1ds4.dot(dads3)-dA1ds3.dot(dads4)).dot(hminveitsm)
        d2Vds4ds3 += 0 + dA1ds3.dot(dEIT) + 0 if num_meshes == 4 else 0
        d2Vds4ds3 = h2em.dot(d2Vds4ds3)
        d2Vds4ds4 = (-2*dA1ds4.dot(dads4)-hminv.dot(da2ds4ds4)).dot(hminveitsm)
        d2Vds4ds4 += dA1ds4.dot(dEIT) + dA1ds4.dot(dEIT) + \
                     hminv.dot(d2EIT) if num_meshes == 4 else 0
        d2Vds4ds4 = h2em.dot(d2Vds4ds4)
    else:
        return (d2Vds1ds1, d2Vds1ds2, d2Vds1ds3, d2Vds2ds1, d2Vds2ds2,
                d2Vds2ds3, d2Vds3ds1, d2Vds3ds2, d2Vds3ds3)
    if num_meshes > 4:
        raise NotImplementedError
    else:
        return (d2Vds1ds1, d2Vds1ds2, d2Vds1ds3, d2Vds1ds4, d2Vds2ds1,
                d2Vds2ds2, d2Vds2ds3, d2Vds2ds4, d2Vds3ds1, d2Vds3ds2,
                d2Vds3ds3, d2Vds3ds4, d2Vds4ds1, d2Vds4ds2, d2Vds4ds3,
                d2Vds4ds4)


def jacobian_per_measurements(cond, head, return_model=False):
    #_, dV4ds, _, _ = first_derivatives(head, head.sens, head.A, head.Ainv,
    #                                   head.h2em, head.eitsm, head.ind)
    dEIT, dV4ds, dA1ds, dads = first_derivatives(head, head.sens, head.A, head.Ainv,
                                       head.h2em, head.eitsm, head.ind)
    head.first_derivatives = (dEIT, dV4ds, dA1ds, dads)
    C = head.C[0] # one freq for now
    num_meshes = len(head.cond)
    jacob=np.zeros((num_meshes, C.shape[0], C.shape[1], C.shape[0]))
    if num_meshes == 4:
        dV4ds1, dV4ds2, dV4ds3, dV4ds4 = dV4ds
        jacob[1] = np.einsum('ijl,kl', C, dV4ds2)
        jacob[2] = np.einsum('ijl,kl', C, dV4ds3)
        jacob[3] = np.einsum('ijl,kl', C, dV4ds4)
    elif num_meshes == 3:
        dV4ds1, dV4ds2, dV4ds3 = dV4ds
        jacob[1] = np.einsum('ijl,kl', C, dV4ds2)
        jacob[2] = np.einsum('ijl,kl', C, dV4ds3)
    elif num_meshes == 2:
        dV4ds1, dV4ds2 = dV4ds
        jacob[1] = np.einsum('ijl,kl', C, dV4ds2)
    elif num_meshes == 1:
        dV4ds1 = dV4ds
    else:
        raise NotImplementedError
    jacob[0] = np.einsum('ijl,kl', C, dV4ds1)

    if return_model:
        return jacob, head
    return jacob

def jacobian(cond, head, return_model=False):
    if isinstance(cond, list) or isinstance(cond, np.ndarray):
        cond = {shell: cond[i] for i, shell in enumerate(reversed(head.mesh_names))} #inside out
        #cond = {shell: cond[i] for i, shell in enumerate(head.mesh_names)} #outside in
    if any([head.cond[tissue] != cond[tissue] for tissue in head.mesh_names]):
        new_cond = OrderedDict([(tissue, cond[tissue]) for tissue in reversed(head.mesh_names)]) # inside out
        #new_cond = OrderedDict([(tissue, cond[tissue]) for tissue in head.mesh_names]) # outside in
        head.set_cond(new_cond)
        print("jacobian: SETTING NEW CONDUCTIVITY VALUES:", new_cond)

    if return_model:
        jacob_pm, head = jacobian_per_measurements(cond, head, return_model=return_model)
    else:
        jacob_pm = jacobian_per_measurements(cond, head)
    ND2V = EIT_protocol(head.n_electrodes, protocol = 'all_realistic')
    num_meshes = len(head.mesh_names)
    j = np.zeros((num_meshes, np.sum(ND2V)))
    for i, jacob in enumerate(jacob_pm):
        #j[num_meshes-i-1] = jacob.flatten()[ND2V] #out to inside
        j[i] = jacob.flatten()[ND2V] #in to outside
    j = j.T
    #condition_j = np.linalg.cond(j)
    #print('Condition number of jacobian: %f' % condition_j)
    if return_model:
        return j, head
    return j


def hessian_per_measurement(cond, head):
    dEIT, _, dA1ds, dads = first_derivatives(head, head.sens, head.A,
                                             head.Ainv, head.h2em,
                                             head.eitsm, head.ind)
    d2V4ds = second_derivatives(head, head.sens, head.A, head.Ainv, head.h2em,
                                head.eitsm, head.ind, dEIT, dA1ds, dads)
    C = head.C[0] # one freq for now
    num_meshes = len(head.mesh_names)
    hess = np.zeros((num_meshes,num_meshes, C.shape[0], C.shape[1], C.shape[0]))
    if num_meshes == 1:
        d2Vds1ds1 = d2V4ds
    elif num_meshes == 2:
        d2Vds1ds1, d2Vds1ds2, d2Vds2ds1, d2Vds2ds2 = d2V4ds
    elif num_meshes == 3:
        d2Vds1ds1, d2Vds1ds2, d2Vds1ds3, d2Vds2ds1, d2Vds2ds2, d2Vds2ds3, d2Vds3ds1, d2Vds3ds2, d2Vds3ds3 = d2V4ds
    elif num_meshes == 4:
        d2Vds1ds1, d2Vds1ds2, d2Vds1ds3, d2Vds1ds4, d2Vds2ds1, d2Vds2ds2, d2Vds2ds3, d2Vds2ds4, d2Vds3ds1, d2Vds3ds2, d2Vds3ds3, d2Vds3ds4, d2Vds4ds1, d2Vds4ds2, d2Vds4ds3, d2Vds4ds4 = d2V4ds
    else:
        raise NotImplementedError

    hess[0,0] = np.einsum('ijl,kl', C, d2Vds1ds1)
    if num_meshes > 1:
        hess[0,1] = np.einsum('ijl,kl', C, d2Vds1ds2)
        hess[1,0] = np.einsum('ijl,kl', C, d2Vds2ds1)
        hess[1,1] = np.einsum('ijl,kl', C, d2Vds2ds2)
        #assert (hess[1,0] == hess[0,1]).all()
    if num_meshes > 2:
        hess[1,2] = np.einsum('ijl,kl', C, d2Vds2ds3)
        hess[2,1] = np.einsum('ijl,kl', C, d2Vds3ds2)
        hess[2,2] = np.einsum('ijl,kl', C, d2Vds3ds3)
        #assert (hess[2,0] == hess[0,2]).all()
        #assert (hess[2,1] == hess[1,2]).all()
    if num_meshes > 3:
        hess[1,3] = np.einsum('ijl,kl', C, d2Vds2ds4)
        hess[2,3] = np.einsum('ijl,kl', C, d2Vds3ds4)
        hess[3,1] = np.einsum('ijl,kl', C, d2Vds4ds2)
        hess[3,2] = np.einsum('ijl,kl', C, d2Vds4ds3)
        hess[3,3] = np.einsum('ijl,kl', C, d2Vds4ds4)
        #assert (hess[3,0] == hess[0,3]).all()
        #assert (hess[3,1] == hess[1,3]).all()
        #assert (hess[3,2] == hess[2,3]).all()
    return hess

def hessian(cond, head):
    if isinstance(cond, list) or isinstance(cond, np.ndarray):
        cond = {shell: cond[i] for i, shell in enumerate(reversed(head.mesh_names))}
        #cond = {shell: cond[i] for i, shell in enumerate(head.mesh_names)} #outside in
    if any([head.cond[tissue] != cond[tissue] for tissue in head.mesh_names]):
        new_cond = OrderedDict([(tissue, cond[tissue]) for tissue in reversed(head.mesh_names)]) # inside out
        #new_cond = OrderedDict([(tissue, cond[tissue]) for tissue in head.mesh_names]) # outside in
        head.set_cond(new_cond)
        print("hessian: SETTING NEW CONDUCTIVITY VALUES:", new_cond)

    hess_pm = hessian_per_measurement(cond, head)
    ND2V = EIT_protocol(head.n_electrodes, protocol = 'all_realistic')
    num_meshes = len(head.mesh_names)
    h = np.zeros((num_meshes,num_meshes, np.sum(ND2V)))
    for i, j in itertools.product(range(num_meshes), range(num_meshes)):
        h[i,j] = hess_pm[i,j].flatten()[ND2V] # in to outside
    #condition_h = np.linalg.cond(h)
    #print('Condition number of hessian:', condition_h)
    return h




######### Frechet derivatives ############

def dAds1(cond, ind, hm):
    dads1 = np.zeros((hm.shape), dtype='float64')
    if len(cond) == 1:
        dads1[np.ix_(ind['V'][0],ind['V'][0])] = hm[np.ix_(ind['V'][0],
                                                           ind['V'][0])] / \
                                                 cond[0] #N11
        dads1[np.ix_(ind['p'][0],ind['p'][0])] = - pow(cond[0], -2) * \
                                                 hm[np.ix_(ind['p'][0],
                                                           ind['p'][0])] / \
                                                 (1.0/cond[0]) #S11
    else:
        dads1[np.ix_(ind['V'][0],ind['V'][0])] = hm[np.ix_(ind['V'][0],
                                                           ind['V'][0])] / \
                                                 (cond[0] + cond[1]) #N11
        dads1[np.ix_(ind['p'][0],ind['p'][0])] = - pow(cond[0], -2) * \
                                                 hm[np.ix_(ind['p'][0],
                                                           ind['p'][0])] / \
                                                 (1.0/cond[0] + 1.0/cond[1]) #S11
    return dads1

def dAds2(cond, ind, hm):
    dads2 = np.zeros((hm.shape), dtype='float64')
    dads2[np.ix_(ind['V'][0],ind['V'][0])] = hm[np.ix_(ind['V'][0],
                                                         ind['V'][0])] / \
                                               (cond[0] + cond[1]) #N11
    dads2[np.ix_(ind['V'][1],ind['V'][0])] = - hm[np.ix_(ind['V'][1],
                                                         ind['V'][0])] / \
                                               (-cond[1]) #N21
    dads2[np.ix_(ind['V'][0],ind['V'][1])] = - hm[np.ix_(ind['V'][0],
                                                         ind['V'][1])] / \
                                               (-cond[1]) #N12
    dads2[np.ix_(ind['p'][0],ind['p'][0])] = - pow(cond[1], -2) * \
                                               hm[np.ix_(ind['p'][0],
                                                         ind['p'][0])] / \
                                               (1.0/cond[0] + 1.0/cond[1]) #S11
    if len(cond) == 2:
        dads2[np.ix_(ind['V'][1],ind['V'][1])] = hm[np.ix_(ind['V'][1],
                                                           ind['V'][1])] / \
                                                   cond[1] #N22

    elif len(cond) > 2:
        dads2[np.ix_(ind['V'][1],ind['V'][1])] = hm[np.ix_(ind['V'][1],
                                                             ind['V'][1])] \
                                                   / (cond[1] + cond[2]) #N22
        dads2[np.ix_(ind['p'][1],ind['p'][0])] = pow(cond[1], -2) * \
                                                   hm[np.ix_(ind['p'][1],
                                                             ind['p'][0])] \
                                                   / (-1/cond[1]) #S21
        dads2[np.ix_(ind['p'][0],ind['p'][1])] = pow(cond[1], -2) * \
                                                   hm[np.ix_(ind['p'][0],
                                                             ind['p'][1])] \
                                                   / (-1/cond[1]) #S12
        dads2[np.ix_(ind['p'][1],ind['p'][1])] = - pow(cond[1], -2) * \
                                                   hm[np.ix_(ind['p'][1],
                                                             ind['p'][1])] \
                                                   / (1.0/cond[1] +
                                                      1.0/cond[2]) #S22
    return dads2

def dAds3(cond, ind, hm):
    dads3 = np.zeros((hm.shape), dtype='float64')
    dads3[np.ix_(ind['V'][1],ind['V'][1])] = hm[np.ix_(ind['V'][1],
                                                         ind['V'][1])] / \
                                               (cond[1] + cond[2]) #N22
    dads3[np.ix_(ind['V'][2],ind['V'][1])] = - hm[np.ix_(ind['V'][2],
                                                         ind['V'][1])] / \
                                               (-cond[2]) #N32
    dads3[np.ix_(ind['V'][1],ind['V'][2])] = - hm[np.ix_(ind['V'][1],
                                                         ind['V'][2])] / \
                                               (-cond[2]) #N23
    dads3[np.ix_(ind['p'][1],ind['p'][1])] = - pow(cond[2], -2) * \
                                               hm[np.ix_(ind['p'][1],
                                                         ind['p'][1])] / \
                                               (1.0/cond[1] + 1.0/cond[2]) #S22
    if len(cond) == 3:
        dads3[np.ix_(ind['V'][2],ind['V'][2])] = hm[np.ix_(ind['V'][2],
                                                             ind['V'][2])] / \
                                                   cond[2] #N33
    elif len(cond) > 3:
        dads3[np.ix_(ind['V'][2],ind['V'][2])] = hm[np.ix_(ind['V'][2],
                                                             ind['V'][2])] / \
                                                   (cond[2] + cond[3]) #N33
        dads3[np.ix_(ind['p'][2],ind['p'][1])] = pow(cond[2], -2) * \
                                                   hm[np.ix_(ind['p'][2],
                                                             ind['p'][1])] \
                                                   / (-1/cond[2]) #S32
        dads3[np.ix_(ind['p'][1],ind['p'][2])] = pow(cond[2], -2) * \
                                                   hm[np.ix_(ind['p'][1],
                                                             ind['p'][2])] \
                                                   / (-1/cond[2]) #S23
        dads3[np.ix_(ind['p'][2],ind['p'][2])] = - pow(cond[2], -2) * \
                                                   hm[np.ix_(ind['p'][2],
                                                             ind['p'][2])] / \
                                                   (1.0/cond[2] + 1.0/cond[3])
                                                   #S33
    return dads3

def dAds4(cond, ind, hm):
    dads4 = np.zeros((hm.shape), dtype='float64')
    dads4[np.ix_(ind['V'][2],ind['V'][2])] = hm[np.ix_(ind['V'][2],
                                                         ind['V'][2])] / \
                                               (cond[2] + cond[3]) #N33
    dads4[np.ix_(ind['V'][3],ind['V'][2])] = - hm[np.ix_(ind['V'][3],
                                                         ind['V'][2])] / \
                                               (-cond[3]) #N43
    dads4[np.ix_(ind['V'][2],ind['V'][3])] = - hm[np.ix_(ind['V'][2],
                                                         ind['V'][3])] / \
                                               (-cond[3]) #N34
    dads4[np.ix_(ind['p'][2],ind['p'][2])] = - pow(cond[3], -2) * \
                                               hm[np.ix_(ind['p'][2],
                                                         ind['p'][2])] / \
                                               (1.0/cond[2] + 1.0/cond[3]) #S33
    if len(cond) == 4:
        dads4[np.ix_(ind['V'][3],ind['V'][3])] = hm[np.ix_(ind['V'][3],
                                                             ind['V'][3])] / \
                                                   cond[3] #N44
    elif len(cond) > 4:
        dads4[np.ix_(ind['V'][3],ind['V'][3])] = hm[np.ix_(ind['V'][3],
                                                             ind['V'][3])] / \
                                                   (cond[3] + cond[4]) #N44
        dads4[np.ix_(ind['p'][3],ind['p'][2])] = - hm[np.ix_(ind['p'][3],
                                                              ind['p'][2])] \
                                                   / cond[3] #S43
        dads4[np.ix_(ind['p'][2],ind['p'][3])] = - hm[np.ix_(ind['p'][2],
                                                              ind['p'][3])] \
                                                   / cond[3] #S34
        dads4[np.ix_(ind['p'][3],ind['p'][3])] = - pow(cond[3], -2) * \
                                                   hm[np.ix_(ind['p'][3],
                                                             ind['p'][3])] / \
                                                   (1.0/cond[3] + 1.0/cond[4])
                                                   #S44
    return dads4





######### 2nd Frechet derivatives ############

def dAds1ds1(cond, ind, dads1):
    da2ds1ds1 = np.zeros((dads1.shape))
    da2ds1ds1[np.ix_(ind['p'][0],ind['p'][0])] = -2 / cond[0] * \
        dads1[np.ix_(ind['p'][0],ind['p'][0])] # S11
    return da2ds1ds1

def dAds2ds2(cond, ind, dads2):
    num_meshes = len(cond)
    da2ds2ds2 = np.zeros((dads2.shape))
    da2ds2ds2[np.ix_(ind['p'][0],ind['p'][0])] = -2 / cond[1] * \
        dads2[np.ix_(ind['p'][0],ind['p'][0])] # S11
    if num_meshes > 2:
        da2ds2ds2[np.ix_(ind['p'][0],ind['p'][1])] = -2 / cond[1] * \
            dads2[np.ix_(ind['p'][0],ind['p'][1])] # S12
        da2ds2ds2[np.ix_(ind['p'][1],ind['p'][0])] = -2 / cond[1] * \
            dads2[np.ix_(ind['p'][1],ind['p'][0])] # S21
        da2ds2ds2[np.ix_(ind['p'][1],ind['p'][1])] = -2 / cond[1] * \
            dads2[np.ix_(ind['p'][1],ind['p'][1])] # S22
    return da2ds2ds2

def dAds3ds3(cond, ind, dads3):
    num_meshes = len(cond)
    da2ds3ds3 = np.zeros((dads3.shape))
    da2ds3ds3[np.ix_(ind['p'][1],ind['p'][1])] = -2 / cond[2] * \
        dads3[np.ix_(ind['p'][1],ind['p'][1])] # S22
    if num_meshes > 3:
        da2ds3ds3[np.ix_(ind['p'][1],ind['p'][2])] = -2 / cond[2] * \
            dads3[np.ix_(ind['p'][1],ind['p'][2])] # S23
        da2ds3ds3[np.ix_(ind['p'][2],ind['p'][1])] = -2 / cond[2] * \
            dads3[np.ix_(ind['p'][2],ind['p'][1])] # S32
        da2ds3ds3[np.ix_(ind['p'][2],ind['p'][2])] = -2 / cond[2] * \
            dads3[np.ix_(ind['p'][2],ind['p'][2])] # S33
    return da2ds3ds3

def dAds4ds4(cond, ind, dads4):
    num_meshes = len(cond)
    da2ds4ds4 = np.zeros((dads4.shape))
    da2ds4ds4[np.ix_(ind['p'][2],ind['p'][2])] = - 2 / cond[3] * \
        dads4[np.ix_(ind['p'][2],ind['p'][2])] # S33
    if num_meshes > 4:
        da2ds4ds4[np.ix_(ind['p'][2],ind['p'][3])] = -2 / cond[3] * \
            dads4[np.ix_(ind['p'][2],ind['p'][3])] # S34
        da2ds4ds4[np.ix_(ind['p'][3],ind['p'][2])] = -2 / cond[3] * \
            dads4[np.ix_(ind['p'][3],ind['p'][2])] # S43
        da2ds4ds4[np.ix_(ind['p'][3],ind['p'][3])] = -2 / cond[3] * \
            dads4[np.ix_(ind['p'][3],ind['p'][3])] # S44
    return da2ds4ds4

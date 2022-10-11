import os, pytest
import numpy as np
import openmeeg as om
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pyreite.OpenMEEGHead import OpenMEEGHead, om2np
import sys
sys.path.append('./tests')
from data_for_testing import simple_test_shapes, find_center_of_triangle
from collections import OrderedDict


def test_OpenMEEGHead():
    # tbd
    tmp ='tmp_tri%d' 
    num = np.random.randint(2, 4)
    num = 2
    bnds = simple_test_shapes(num_nested_meshes=num)
    geom, cond = OrderedDict(), {}
    for i, bnd in enumerate(bnds):
        geom[tmp % (i+1)] = bnd
        cond[tmp % (i+1)] = np.random.rand()
    electrodes = find_center_of_triangle(bnds[-1][0], bnds[-1][1])
    head = OpenMEEGHead(cond, geom, electrodes)
    nb_entries, nb_elements = 0, 0
    for i, bnd in enumerate(bnds):
        pos, tri = bnd
        nb_elements += pos.shape[0] + tri.shape[0] 
        nb_entries += len(head.ind['V'][i]) + len(head.ind['p'][i])
        #if i != len(bnds)-1:
        #    nb_entries += len(head.ind['p'][i])
        if i == len(bnds)-1:
            nb_elements -= tri.shape[0]
    #assert nb_elements == nb_entries == head.A.nlin()
    assert nb_entries == head.A.nlin()
    #assert head.ind['p'][0][-1]+1 == head.A.nlin() # if inside out and openmeeg works outside in
    assert head.mesh_names[0] == list(geom.keys())[0] # right order
    assert head.Ainv != head.A  # the order is important here!
    eitsm = om2np(head.eitsm)
    assert np.sum(eitsm) != 0
    assert head.eitsm.nlin() == head.A.nlin() 
    assert head.eitsm.ncol() == electrodes.shape[0] == head.n_electrodes
    assert head.gain.nlin() == head.gain.ncol() == electrodes.shape[0]
    h2em = om2np(head.h2em)
    assert (h2em >= 0.0).all() and (h2em <= 1.0).all()   
    assert int(np.sum(h2em)) == electrodes.shape[0]
    head.set_cond(cond)
    assert head._A == None
    assert head._Ainv == None
    gain = om2np(head.gain)
    hminv = om2np(head.Ainv)
    #assert_array_almost_equal(h2em.dot(hminv.dot(eitsm)), gain)

if __name__ == '__main__':
    test_OpenMEEGHead()

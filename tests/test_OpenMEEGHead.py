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
    assert nb_entries == head.A.shape[0]
    #assert head.ind['p'][0][-1]+1 == head.A.nlin() # if inside out and openmeeg works outside in
    assert head.mesh_names[0] == list(geom.keys())[0] # right order
    assert (head.Ainv != head.A).all()  # the order is important here!
    assert np.sum(head.eitsm) != 0
    assert head.eitsm.shape[0] == head.A.shape[0] 
    assert head.eitsm.shape[1] == electrodes.shape[0] == head.n_electrodes
    assert head.gain.shape[0] == head.gain.shape[1] == electrodes.shape[0]
    #assert (head.h2em >= 0.0).all() and (head.h2em <= 1.0).all()   
    assert int(np.round(np.sum(head.h2em),0)) == electrodes.shape[0]
    head.set_cond(cond)
    assert head._A == None
    assert head._Ainv == None

    new_head = OpenMEEGHead(cond, geom, electrodes)
    gain = new_head.gain
    Ainv = new_head.Ainv
    assert_array_almost_equal(new_head.gain, gain)

if __name__ == '__main__':
    test_OpenMEEGHead()

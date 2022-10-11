import os, pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import openmeeg as om
from data_for_testing import simple_test_shapes, find_center_of_triangle
from pyreite.data_io import *

def test_write_geom_file():
    tmp ='tmp_tri%d' 
    num = np.random.randint(2, 6)
    bnds = simple_test_shapes(num_nested_meshes=num)
    geom, names = {}, []
    for i, bnd in enumerate(bnds):
        geom[tmp % (i+1)] = bnd
        names.append(str(i+1))
    geom_file = './tmp_test.geom'
    write_geom_file(geom, geom_file)
    geometry = om.Geometry(geom_file)
    for filename in [geom_file] + [tmp % (i+1)+'.tri' for i in range(len(bnds))]:
        os.remove(filename)
    assert not geometry.has_cond()
    assert geometry.nb_meshes() == len(bnds)
    assert geometry.size() == np.sum([bnd[0].shape[0]+bnd[1].shape[0] for bnd in bnds])
    for i, mesh in enumerate(geometry.meshes()):
        assert str(mesh) == names[i]

def test_write_cond_file():
    tmp ='tmp_tri%d' 
    num = np.random.randint(2, 6)
    bnds = simple_test_shapes(num_nested_meshes=num)
    cond, geom = {}, {}
    for i, bnd in enumerate(bnds):
        name = i+1 #len(bnds)-i
        geom[tmp % (i+1)] = bnd
        cond[tmp % (i+1)] = np.random.rand()
    cond_file, geom_file = './tmp_test.cond', './tmp_test.geom'
    write_geom_file(geom, geom_file)
    write_cond_file(cond, cond_file)
    geometry = om.Geometry(geom_file, cond_file)
    for filename in [geom_file, cond_file] + [tmp % (i+1)+'.tri' for i in range(len(bnds))]:
        os.remove(filename)
    assert geometry.has_cond()
    for i in range(1, len(bnds)+1):
        assert pytest.approx(cond[tmp % i], 5) == geometry.sigma(tmp % i)


def test_write_elec_file():
    tmp ='tmp_tri%d' 
    num = np.random.randint(2, 6)
    bnds = simple_test_shapes(num_nested_meshes=num)
    geom = {}
    for i, bnd in enumerate(bnds):
        geom[tmp % (i+1)] = bnd

    geom_file, elec_file = './tmp_test.geom', './tmp_test.elec'
    write_geom_file(geom, geom_file)
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])
    elecs = elecs[::2,:]
    write_elec_file(elecs, elec_file)

    geometry = om.Geometry(geom_file)
    sensors = om.Sensors(elec_file, geometry) 
    for filename in [elec_file, geom_file] + [tmp % (i+1) +'.tri' for i in range(len(bnds))]:
        os.remove(filename)
    assert_array_almost_equal(om.asarray(sensors.getPositions()), elecs)


def test_write_load_tri():
    num = np.random.randint(3, 50)
    pos, tri = [], []
    pos =  np.random.rand(num*3).reshape((num, 3))
    elems = [i for i in np.arange(num) for j in range(4)]
    while len(elems) % 3 != 0:
        elems.append(np.random.randint(0, num))
    while elems:
        face = []
        for _ in range(3):
            t = np.random.choice(elems)
            face.append(t)
            first_t = [i for i in range(len(elems)) if elems[i] == t][0]
            del elems[first_t]
        tri.append(face)
           
        """
        tri.append(np.random.choice(elems,3))
        for t in tri[len(tri)-1]:
            first_t = [i for i in range(len(elems)) if elems[i] == t]
            try:
                del elems[first_t[0]]
            except:
                import code
                code.interact(local=locals())
        """
    tri = np.array(tri)
    write_tri(pos, tri, './tmp.tri')
    bnd  = load_tri('./tmp.tri')
    os.remove('./tmp.tri')
    new_pos, new_tri = bnd
    assert_array_equal(pos, new_pos)
    assert_array_equal(tri, new_tri)



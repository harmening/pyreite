import os, pytest, tempfile
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import openmeeg as om
from data_for_testing import simple_test_shapes, find_center_of_triangle
from pyreite.data_io import *


def test_load_elecs_dips_txt():
    dirpath = tempfile.mkdtemp()
    tmp_fn = os.path.join(dirpath, 'tmp.txt')
    elecs = []
    with open(tmp_fn, 'w') as f:
        for i in range(np.random.randint(2, 6)):
            elec = [np.random.rand() for _ in range(3)]
            elecs.append(elec)
            f.write('%f, %f, %f\n' % (elec[0], elec[1], elec[2]))
    assert_array_almost_equal(elecs, load_elecs_dips_txt(tmp_fn))
    
def test_write_dip_file():
    dirpath = tempfile.mkdtemp()
    tmp_fn = os.path.join(dirpath, 'tmp.txt')
    dips = [[np.random.rand() for _ in range(6)] for i in \
            range(np.random.randint(2, 6))]
    write_dip_file(dips, tmp_fn)
    assert_array_almost_equal(dips, load_elecs_dips_txt(tmp_fn))
    dips = [[np.random.rand() for _ in range(3)] for i in \
            range(np.random.randint(2, 6))]
    write_dip_file(dips, tmp_fn)
    dips_with_vecs = []
    for dip in dips:
        dips_with_vecs.append(dip+[1,0,0])
        dips_with_vecs.append(dip+[0,1,0])
        dips_with_vecs.append(dip+[0,0,1])
    write_dip_file(dips, tmp_fn)
    assert_array_almost_equal(dips_with_vecs, load_elecs_dips_txt(tmp_fn))

def test_write_geom_file():
    dirpath = tempfile.mkdtemp()
    tmp = os.path.join(dirpath, 'tmp_tri%d')
    num = np.random.randint(2, 6)
    bnds = simple_test_shapes(num_nested_meshes=num)
    geom, names = {}, []
    for i, bnd in enumerate(bnds):
        geom[tmp % (i+1)] = bnd
        names.append(str(i+1))
    geom_file = os.path.join(dirpath, 'tmp_test.geom')
    write_geom_file(geom, geom_file)
    geometry = om.Geometry(geom_file)
    for fn in [geom_file] + [tmp % (i+1)+'.tri' for i in range(len(bnds))]:
        os.remove(fn)
    assert not geometry.has_conductivities()
    assert geometry.nb_parameters() == np.sum([bnd[0].shape[0]+bnd[1].shape[0] \
                                               for bnd in bnds])
    for i in range(len(bnds)):
        mesh = geometry.mesh(str(i+1))
        assert str(mesh) == str(i+1)

def test_write_cond_file():
    dirpath = tempfile.mkdtemp()
    tmp = os.path.join(dirpath, 'tmp_tri%d')
    num = np.random.randint(2, 6)
    bnds = simple_test_shapes(num_nested_meshes=num)
    cond, geom = {}, {}
    for i, bnd in enumerate(bnds):
        geom[tmp % (i+1)] = bnd
        cond[tmp % (i+1)] = np.random.rand()
    cond_file = os.path.join(dirpath, 'tmp_test.cond')
    geom_file = os.path.join(dirpath, 'tmp_test.geom')
    write_geom_file(geom, geom_file)
    write_cond_file(cond, cond_file)
    geometry = om.Geometry(geom_file, cond_file)
    for fn in [geom_file, cond_file] + [tmp % (i+1)+'.tri' for i in \
               range(len(bnds))]:
        os.remove(fn)
    assert geometry.has_conductivities()
    for i in range(1, len(bnds)):
        assert pytest.approx(cond[tmp % i], 5) == geometry.sigma(geometry.mesh(str(i)), geometry.mesh(str(i+1)))


def test_write_elec_file():
    dirpath = tempfile.mkdtemp()
    tmp = os.path.join(dirpath, 'tmp_tri%d')
    num = np.random.randint(2, 6)
    bnds = simple_test_shapes(num_nested_meshes=num)
    geom = {}
    for i, bnd in enumerate(bnds):
        geom[tmp % (i+1)] = bnd

    geom_file = os.path.join(dirpath, 'tmp_test.geom')
    elec_file = os.path.join(dirpath, 'tmp_test.elec')
    write_geom_file(geom, geom_file)
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])
    elecs = elecs[::2,:]
    write_elec_file(elecs, elec_file)

    geometry = om.Geometry(geom_file)
    sensors = om.Sensors(elec_file, geometry) 
    for fn in [elec_file, geom_file] + [tmp % (i+1) +'.tri' for i in \
               range(len(bnds))]:
        os.remove(fn)
    assert_array_almost_equal(sensors.getPositions().array(), elecs)


def test_write_load_tri():
    dirpath = tempfile.mkdtemp()
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
    tri = np.array(tri)
    write_tri(pos, tri, os.path.join(dirpath, 'tmp.tri'))
    bnd = load_tri(os.path.join(dirpath, 'tmp.tri'))
    os.remove(os.path.join(dirpath, 'tmp.tri'))
    new_pos, new_tri = bnd
    assert_array_equal(pos, new_pos)
    assert_array_equal(tri, new_tri)


def test_write_bnd():
    # not in use anymore?
    assert True

if __name__ == '__main__':
    test_write_cond_file()

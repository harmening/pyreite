import os, pytest
import numpy as np
import openmeeg as om
import random
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pyreite.geometry import create_geometry, mesh2bnd, align_electrodes
from pyreite.data_io import write_cond_file, write_geom_file, write_elec_file
from tests.data_for_testing import simple_test_shapes, find_center_of_triangle, colin
from collections import OrderedDict


def test_create_geometry():
    tmp ='tmp_tri%d' 
    num = random.randint(2, 6)
    bnds = simple_test_shapes(num_nested_meshes=num)
    cond = {}
    geom = OrderedDict()
    for i, bnd in enumerate(bnds):
        geom[tmp % (i+1)] = bnd
        cond[tmp % (i+1)] = random.random()

    cond_file, geom_file, elec_file = './tmp_test.cond', './tmp_test.geom', './tmp_test.elec'
    write_cond_file(cond, cond_file)
    write_geom_file(geom, geom_file)
    elecs = find_center_of_triangle(bnds[-1][0], bnds[-1][1])
    elecs = elecs[::2,:]
    write_elec_file(elecs, elec_file)

    geometry, sensors = create_geometry(geom_file, cond_file, elec_file)
    for i in range(1, len(bnds)+1):
        os.remove(tmp % i + '.tri')
    for filename in [cond_file, geom_file, elec_file]:
        os.remove(filename)
    n_elecs, dim = elecs.shape
    assert sensors.getNumberOfSensors() == n_elecs
    assert_array_almost_equal(om.asarray(sensors.getPositions()), elecs)
    assert geometry.nb_meshes() == len(bnds)
    assert geometry.has_cond()
    assert geometry.is_nested()
    assert geometry.nb_vertices() == np.sum([bnd[0].shape[0] for bnd in bnds])
    assert geometry.nb_triangles() == np.sum([bnd[1].shape[0] for bnd in bnds])
    assert geometry.size() == np.sum([bnd[0].shape[0]+bnd[1].shape[0] for bnd in bnds])
    for i in range(1, len(bnds)+1):
        assert pytest.approx(cond[tmp % i], 5) == geometry.sigma(tmp % i)
    mesh = random.choice(geometry.meshes())
    print(mesh)
    bnd = [bnd for i, bnd in enumerate(geom.values()) if i+1 == int(str(mesh))][0]
    min_idx = np.min([v.getindex() for v in mesh.vertices()])
    for vertex in mesh.vertices():
        assert_array_almost_equal(np.array([vertex(x) for x in range(dim)]),
                                  bnd[0][vertex.getindex()-min_idx])
    #assert mesh.get_triangles_for_vertex(vertex) == np.where(bnd[1] == vertex.getindex())


def test_mesh2bnd():
    bnd = simple_test_shapes(num_nested_meshes=1)[0]
    geom = {'tmp_tri1': bnd}
    cond = {'tmp_tri1': random.random()}
    cond_file, geom_file = './tmp_test.cond', './tmp_test.geom'
    write_cond_file(cond, cond_file)
    write_geom_file(geom, geom_file)
    geometry = om.Geometry(geom_file, cond_file) 
    os.remove('tmp_tri1.tri')
    mesh = geometry.meshes()[0]
    new_pos, new_tri = mesh2bnd(mesh)
    pos, tri = bnd
    assert_array_equal(new_pos, pos)
    assert_array_equal(new_tri, tri)


def test_getVolumeIndices():
    # maybe not necessary any more! (use geom.meshes() built in functions)
    assert True


def test_align_electrodes():
    bnd_colin = colin()
    elec_aligned = align_electrodes(bnd_colin['electrodes_not_aligned'], bnd_colin['scalp'])
    assert_array_almost_equal(elec_aligned, bnd_colin['electrodes_aligned'], 2)
    dist = np.linalg.norm(elec_aligned-bnd_colin['electrodes_aligned'], axis=1)
    assert_array_almost_equal(dist, np.zeros(len(elec_aligned)), 2)

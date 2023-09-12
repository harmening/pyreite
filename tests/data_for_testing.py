from math import sqrt
from pyreite.data_io import load_tri
import numpy as np
import os
BASEDIR = os.path.dirname(os.path.realpath(__file__))


def fast_simple_test_shapes(num_nested_meshes):
    bnds = []
    scale = 1.1
    verts, faces = form_base_icosahedron(scale)
    verts, faces, middle_point_cache = subdivision(verts, faces, {}, scale)
    for i in range(num_nested_meshes):
        scale *= 1.1
        verts, faces = form_base_icosahedron(scale)
        verts, faces, middle_point_cache = subdivision(verts, faces, {}, scale)
        verts, faces, middle_point_cache = subdivision(verts, faces, \
                                                       middle_point_cache, \
                                                       scale)
        verts, faces, middle_point_cache = subdivision(verts, faces, \
                                                       middle_point_cache, \
                                                       scale)
        pos, tri = np.array(verts), np.array(faces)
        pos = (pos/100) * (i+1)
        bnds.append((pos, tri))
    return bnds 

def simple_test_shapes(num_nested_meshes):
    bnds = []
    scale = 1.3
    verts, faces = form_base_icosahedron(scale)
    middle_point_cache = {}
    for i in range(num_nested_meshes):
        pos, tri = np.array(verts), np.array(faces)
        pos = (pos/100) * (i+1)
        bnds.append((pos, tri))
        verts, faces, middle_point_cache = subdivision(verts, faces, \
                                                       middle_point_cache, \
                                                       scale)
        #write_tri(pos, tri, './icosphere'+str(i)+'.tri')
    return bnds 

def subdivision(verts, faces, middle_point_cache, scale):
    faces_subdiv = []
    for tri in faces:
        v1, verts, middle_point_cache = middle_point(tri[0], tri[1], verts, \
                                                     middle_point_cache, scale)
        v2, verts, middle_point_cache = middle_point(tri[1], tri[2], verts, \
                                                     middle_point_cache, scale)
        v3, verts, middle_point_cache = middle_point(tri[2], tri[0], verts, \
                                                     middle_point_cache, scale)
        faces_subdiv.append([tri[0], v1, v3])
        faces_subdiv.append([tri[1], v2, v1])
        faces_subdiv.append([tri[2], v3, v2])
        faces_subdiv.append([v1, v2, v3])
    faces = faces_subdiv
    return verts, faces, middle_point_cache

def middle_point(point_1, point_2, verts, middle_point_cache, scale):
    """ Find a middle point and project to the unit sphere """
    # Check if we have already cut this edge first # to avoid duplicated verts
    smaller_index = min(point_1, point_2)
    greater_index = max(point_1, point_2)
    key = '{0}-{1}'.format(smaller_index, greater_index)
    if key in middle_point_cache:
        return middle_point_cache[key], verts, middle_point_cache
    # If it's not in cache, then we can cut it
    vert_1 = verts[point_1]
    vert_2 = verts[point_2]
    middle = [sum(i)/2 for i in zip(vert_1, vert_2)]
    verts.append(vertex(*middle, scale))
    index = len(verts) - 1
    middle_point_cache[key] = index
    return index, verts, middle_point_cache

def vertex(x, y, z, scale):
    """ Return vertex
    coordinates fixed to the unit sphere """
    length = sqrt(x**2 + y**2 + z**2)
    return [(i * scale) / length for i in (x,y,z)]

def form_base_icosahedron(scale):
    # Golden ratio
    PHI = (1 + sqrt(5)) / 2
    verts = [
            vertex(-1, PHI, 0, scale),
            vertex( 1, PHI, 0, scale),
            vertex(-1, -PHI, 0, scale),
            vertex( 1, -PHI, 0, scale),
            vertex(0, -1, PHI, scale),
            vertex(0, 1, PHI, scale),
            vertex(0, -1, -PHI, scale),
            vertex(0, 1, -PHI, scale),
            vertex( PHI, 0, -1, scale),
            vertex( PHI, 0, 1, scale),
            vertex(-PHI, 0, -1, scale),
            vertex(-PHI, 0, 1, scale),]
             
    faces = [
             # 5 faces around point 0
             [0, 11, 5],
             [0, 5, 1],
             [0, 1, 7],
             [0, 7, 10],
             [0, 10, 11],
             # Adjacent faces
             [1, 5, 9],
             [5, 11, 4],
             [11, 10, 2],
             [10, 7, 6],
             [7, 1, 8],
             # 5 faces around 3
             [3, 9, 4],
             [3, 4, 2],
             [3, 2, 6],
             [3, 6, 8],
             [3, 8, 9],
             # Adjacent faces
             [4, 9, 5],
             [2, 4, 11],
             [6, 2, 10],
             [8, 6, 7],
             [9, 8, 1],
            ]
    return verts, faces



def find_center_of_triangle(vertices, faces):
    centerpoints = np.zeros(faces.shape)
    #faces = correct_face_numbering(faces)
    for i, tri in enumerate(faces):
        p1 = vertices[tri[0]]
        p2 = vertices[tri[1]]
        p3 = vertices[tri[2]]
        m12 = (p1+p2)/2.0
        c = (m12+p3)/2.0
        centerpoints[i,:] = c 
    return centerpoints


def colin():
    bnd = {}
    for shell in ['scalp', 'skull', 'csf', 'cortex']:
        bnd[shell] = load_tri(os.path.join(BASEDIR, 'test_data', shell+'.tri'))
    with open(os.path.join(BASEDIR, 'test_data', 'electrodes_not_aligned.txt'), 
              'r') as f:
        bnd['electrodes_not_aligned'] = [[float(x) for x in line.split()] for 
                                         line in f.readlines()]
    with open(os.path.join(BASEDIR, 'test_data', 'electrodes_aligned.txt'),
              'r') as f:
        bnd['electrodes_aligned'] = [[float(x) for x in line.split()] for line 
                                     in f.readlines()]
    return bnd


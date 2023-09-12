#!/usr/bin/env python
import numpy as np
import openmeeg as om
om.__version__ = 2.4


def create_geometry(geom_file, cond_file, elec_file):
    geometry = om.Geometry(geom_file, cond_file)
    assert geometry.is_nested()
    #assert geometry.selfCheck()
    sensors = om.Sensors(elec_file, geometry)
    return geometry, sensors


def mesh2bnd(mesh):
    min_idx = min([vert.getindex() for vert in mesh.vertices()])
    verts = np.zeros((mesh.nb_vertices(), 3))
    for v in mesh.vertices():
        verts[v.getindex()-min_idx,:] = [v(xx) for xx in range(3)]
    # improve security here? (-> if vertices are not steadily ongoing numbers)
    tris = [[v.getindex()-min_idx for v in [tri.s1(), tri.s2(), tri.s3()]]
                                  for tri in mesh.iterator()]
    return np.array(verts), np.array(tris)


# The following code is adapted from fieldtrip:
# https://github.com/fieldtrip/fieldtrip/blob/master/private/project_elec.m
def align_electrodes(elc, scalp):
    pnt, tri = scalp
    pnt, tri, elc = np.array(pnt), np.array(tri), np.array(elc)
    Nelc = len(elc)
    el   = np.zeros((Nelc, 4))
    for i in range(Nelc):
        proj, dist = ptriprojn(pnt[tri[:,0],:], pnt[tri[:,1],:], \
                               pnt[tri[:,2],:], np.array([elc[i,:]]), 1)
        minindx = np.argmin(abs(dist))
        mindist = np.min(abs(dist))
        la, mu, _, _ = lmoutrn(pnt[np.newaxis, tri[minindx,0],:], \
                               pnt[np.newaxis, tri[minindx,1],:], \
                               pnt[np.newaxis, tri[minindx,2],:], \
                               proj[np.newaxis, minindx,:])
        smallest_dist = dist[minindx]
        smallest_tri  = minindx
        smallest_la   = la[0]
        smallest_mu   = mu[0]
        # store the projection for this electrode
        el_i = [smallest_tri, smallest_la, smallest_mu, smallest_dist]
        el[i,:] = np.array(el_i)

    prj = np.zeros((elc.shape))
    for i in range(Nelc):
        v1 = pnt[tri[int(el[i,0]),0],:]
        v2 = pnt[tri[int(el[i,0]),1],:]
        v3 = pnt[tri[int(el[i,0]),2],:]
        la = el[i,1]
        mu = el[i,2]
        prj[i,:] = routlm(v1, v2, v3, la, mu)
    return prj


def ptriprojn(v1, v2, v3, r, flag=0):
    # the optional flag can be:
    #   0 (default)  project the point anywhere on the complete plane
    #   1            project the point within or on the edge of the triangle
    la, mu, dist, proj = lmoutrn(v1, v2, v3, r)
    if flag == 1:
        # for each of the six regions outside the triangle the projection
        # point is on the edge, or on one of the corners
        sel= la<0
        proj[sel,:], dist[sel] = plinprojn(v1[sel,:], v3[sel,:], r, 1)
    
        sel = (mu<0) & (la>=0)
        proj[sel,:], dist[sel] = plinprojn(v1[sel,:], v2[sel,:], r, 1)
    
        # la+mu>1 & mu>0 & la>0 -> project onto vec2
        sel = ((la+mu)>1) & (mu>=0) & (la>=0)
        proj[sel,:], dist[sel] = plinprojn(v2[sel,:], v3[sel,:], r, 1)
    return proj, dist


def lmoutrn(v1, v2, v3, r):
    # LMOUTRN computes the la/mu parameters of a point projected to triangles
    if len(r) == 1 and len(v1) > 1:
        r = np.repeat(r, len(v1)).T.reshape(v1.T.shape).T   

    # compute la/mu parameters
    vec0 = r  - v1
    vec1 = v2 - v1
    #vec2 = v3 - v2
    vec3 = v3 - v1
    origin = np.repeat(np.mean(np.vstack((v1, v2, v3)), axis=0), \
                       len(v1)).T.reshape(v1.T.shape).T     
    tmp = np.empty((3, 2, len(v1)))
    tmp[:,0,:] = vec1.T
    tmp[:,1,:] = vec3.T
    for i in range(len(v1)):
        tmp[:,:,i] = np.linalg.pinv(tmp[:,:,i]).T

    la = np.sum(np.multiply(vec0.T, tmp[:,0,:]), axis=0)
    mu = np.sum(np.multiply(vec0.T, tmp[:,1,:]), axis=0)

            
    # determine the projection onto the plane of the triangle
    proj  = v1 + np.multiply(np.vstack((la, la, la)).T, vec1) + \
            np.multiply(np.vstack((mu, mu, mu)).T, vec3)

    # determine the signed distance from the original point to its projection
    # where the sign is negative if the original point is closer to the origin 
    origin_r    = np.sum(pow((r    - origin), 2), axis=1)
    origin_proj = np.sum(pow((proj - origin), 2), axis=1)

    dist = np.sqrt(np.sum(pow((r - proj), 2), axis=1)) * \
           np.sign(origin_r-origin_proj)

    return la, mu, dist, proj


def plinprojn(l1, l2, r, flag=False):
    # PLINPROJN projects a point onto a line or linepiece
    # where l1 and l2 are Nx3 matrices with the begin and endpoints of the 
    # linepieces, and r is the point that is projected onto the lines
    # This is a vectorized version of Robert's ptriproj function and is
    # generally faster than a for-loop around the mex-file.
    #
    # the optional flag can be:
    #   0 (default)  project the point anywhere on the complete line
    #   1            project the point within or on the edge of the linepiece

    v  = l2-l1                   # vector from l1 to l2
    dp = r -l1 #bsxfun(@minus, r, l1);  # vector from l1 to r
    t  = np.sum(np.multiply(dp, v), 1) / np.sum(pow(v, 2),1)

    if flag:
        for i in range(len(t)):
            if t[i] < 0:
                t[i] = 0
            if t[i] > 1:
                t[i] = 1

    proj = l1 + np.vstack((np.multiply(t, v[:,0]), np.multiply(t, v[:,1]), \
                           np.multiply(t, v[:,2]))).T
    dist = np.sqrt(pow((r[:,0]-proj[:,0]), 2) + pow((r[:,1]-proj[:,1]), 2) + \
                   pow((r[:,2]-proj[:,2]), 2))
    return proj, dist


def routlm(v1, v2, v3, la, mu):
    r = np.zeros(3)
    r[0] = (1-la-mu)*v1[0]+la*v2[0]+mu*v3[0];
    r[1] = (1-la-mu)*v1[1]+la*v2[1]+mu*v3[1];
    r[2] = (1-la-mu)*v1[2]+la*v2[2]+mu*v3[2];
    return r


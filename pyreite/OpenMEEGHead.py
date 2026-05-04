#!/usr/bin/env python
from __future__ import print_function
import os, itertools, tempfile, time
from shutil import copyfile
import numpy as np, openmeeg as om
from pyreite.data_io import write_cond_file, write_elec_file, write_geom_file
from pyreite.geometry import create_geometry
from collections import OrderedDict


def nandot(v1, v2):
    return np.where(np.isnan(v1),0,v1).dot(np.where(np.isnan(v2),0,v2))


class OpenMEEGHead(object):
    def __init__(self, conductivity, geometry, elec_positions, sigma=None,
                 omega=None):
        if not isinstance(geometry, dict):
            raise ValueError
        if not isinstance(elec_positions, (list, np.ndarray, str)):
            raise ValueError

        geom_out2inside = OrderedDict([(tissue, bnd) for tissue, bnd in
                                      reversed(geometry.items())])
        self.mesh_names = list(geom_out2inside.keys()) # out to inside
        self.geometry = geom_out2inside # outside to inside
        self.cond = conductivity
        self.elec_positions = elec_positions

        with tempfile.TemporaryDirectory() as tmp:
            fn_geom = os.path.join(tmp, 'head.geom')
            fn_cond = os.path.join(tmp, 'head.cond')
            fn_elec = os.path.join(tmp, 'head.elec')
            write_geom_file(geom_out2inside, fn_geom)
            write_cond_file(self.cond, fn_cond)
            if isinstance(elec_positions, str):
                copyfile(elec_positions, fn_elec)
            else:
                write_elec_file(elec_positions, fn_elec)
            self.geom, self.sens = create_geometry(fn_geom, fn_cond, fn_elec)

        self.GAUSS_ORDER = 3
        # derivatives are implemented inside out
        self.ind = self._get_indices_inside_out()
        self._A = None
        self._Ainv = None
        self._h2em = None
        self._eitsm = None
        self._gain = None
        self._C = None
        self._V = None
        self._condition_nb = None
        self.sigma = sigma
        self.omega = omega
        self.first_derivatives = None

    def _get_indices_inside_out(self):
        ind = {tissue: i for i, tissue in enumerate(reversed(self.mesh_names))}
        num_meshes = len(ind)
        ind['V'] = []
        ind['p'] = []
        # Openmeeg works outside in
        meshes = [self.geom.mesh(str(i)) for i in range(num_meshes, 0, -1)]
        for m, mesh in enumerate(meshes):
            ind['V'].append([i.index() for i in mesh.vertices()])
            ind['p'].append([t.index() for t in mesh.triangles()])
            if m == (num_meshes-1):
                ind['p'][m] = []
        return ind

    @property
    def A(self):
        """Compute/return the attribute system matrix A"""
        if not isinstance(self._A, np.ndarray):
            A = om.HeadMat(self.geom, self.GAUSS_ORDER)
            self._A = self.asnp(om.Matrix(A))
            #self._condition_nb = self.condition_nb
        return self._A
    @property
    def Ainv(self):
        """Compute/return the attribute system matrix A inverse"""
        if not isinstance(self._Ainv, np.ndarray):
            print("Warning: inverting A explicitly...")
            a = time.time()
            self._Ainv = np.linalg.pinv(self.A)
            print('Inverting took %.2fmin.' % ((time.time()-a)/60))
        return self._Ainv
    @property
    def eitsm(self):
        """Compute/return the attribute righthandside for EIT"""
        if not isinstance(self._eitsm, np.ndarray):
            eitsm = om.EITSourceMat(self.geom, self.sens, self.GAUSS_ORDER)
            self._eitsm = self.asnp(eitsm)
        return self._eitsm
    @property
    def h2em(self):
        """Compute/return the mapping from outer boundary to electrodes"""
        if not isinstance(self._h2em, np.ndarray):
            h2em = om.Head2EEGMat(self.geom, self.sens)
            self._h2em = self.asnp(h2em)
        return self._h2em
    @property
    def gain(self):
        """Compute/return the EIT gain/leadfield matrix"""
        if not isinstance(self._gain, np.ndarray):
            #self._gain = om.GainEEG(om.Matrix(np.fortranarray(self.Ainv)),
            #                        om.Matrix(np.fortranarray(self.eitsm)),
            #                        om.Matrix(np.fortranarray(self.h2em)))
            if isinstance(self._Ainv, np.ndarray):
                # use precomputed Ainv
                L = self.h2em.dot(np.dot(self.Ainv, self.eitsm))
            else:
                # faster
                C = np.linalg.solve(self.A, self.eitsm)
                L = np.dot(self.h2em, C)
            self._gain = L
        return self._gain
    @property
    def n_electrodes(self):
        return self.sens.getNumberOfSensors()

    @property
    def C(self):
        if not isinstance(self._C, np.ndarray):
            _, self._C = self._EIT_data(self.gain, self.sens)
        return self._C
    @property
    def V(self):
        if not isinstance(self._V, np.ndarray):
            self._V, _ = self._EIT_data(self.gain, self.sens, \
                                        freqs=[10**7], Iamp=[133.0e-3], \
                                        ref='no_ref', excluded_chan=[], \
                                        nonans=True)
        return self._V
    def Vsetter(self, freqs=[10**7], Iamp=[133.0e-3], ref='no_ref', \
          excluded_chan=[], nonans=True):
        if not isinstance(self._V, np.ndarray):
            self._V, _ = self._EIT_data(self.gain, self.sens, freqs=freqs, \
                                        Iamp=Iamp, ref=ref, \
                                        excluded_chan=excluded_chan, \
                                        nonans=nonans)
        return self._V

    def _EIT_data(self, G_eit, sens, freqs=[10**7], Iamp=[133.0e-3], \
                  ref='no_ref', excluded_chan=[], nonans=True):
        n_elec = sens.getNumberOfSensors()
        sel_chan = range(1, n_elec+1)
        ei = lambda idx: np.array([0]*(idx)+[1]+[0]*(n_elec-idx-1))
        V = np.zeros((len(freqs), n_elec, n_elec, G_eit.shape[0]))
        C = np.zeros((len(freqs), n_elec, G_eit.shape[1], G_eit.shape[1]))
        for i, freq in enumerate(freqs):
            for Source, Sink in itertools.product(sel_chan, sel_chan):
                C[i, Source-1,Sink-1,:] = np.multiply(np.array(ei(Source-1) - \
                                                      ei(Sink-1)),
                                                      Iamp[i]) #* freq (later?)
            this_V = np.einsum('ijl,kl', C[i], G_eit)
            V[i, :, :, :] = this_V
        for i, this_V in enumerate(V):
            if not nonans:
                for Source, Sink in itertools.product(sel_chan, sel_chan):
                    this_V[Source-1,Sink-1,Source-1]=np.nan
                    this_V[Source-1,Sink-1,Sink-1]=np.nan
                for e in excluded_chan:
                    this_V[e-1,:,:] = np.nan
                    this_V[:,e-1,:] = np.nan
                    this_V[:,:,e-1] = np.nan
            if ref == 'CAR':
                #this_V = np.subtract(this_V, np.nanmean(this_V, axis=2))
                car = np.nanmean(this_V, axis=2)
                for t in range(G_eit.shape[0]):
                    this_V[:,:,t] = np.subtract(this_V[:,:,t], car)
                #    import code
                #    code.interact(local=locals())
            elif ref == 'no_ref': # for testing only
                pass
            else:
                chans = [chan for chan in range(1,  n_elec+1) \
                         if chan not in excluded_chan]
                if ref in chans:
                    this_V = np.subtract(this_V, this_V[:,:,ref-1])
                    #this_V = np.subtract(this_V, np.nanmean(this_V, axis=2)) #+CAR
                else:
                    raise NotImplementedError
            V[i,:,:,:] = this_V
        return V, C

    def asnp(self, openmeeg_matrix):
        """Convert openmeeg_matrix to numpy array"""
        return om2np(openmeeg_matrix)

    def set_cond(self, conductivity):
        if isinstance(conductivity, list) or isinstance(conductivity, \
                                                        np.ndarray):
            conductivity = {t: c for t, c in zip(reversed(self.mesh_names), \
                                                 conductivity)}
        self.cond = conductivity
        geom_out2inside = OrderedDict([(tissue, bnd) for tissue, bnd in
                                       self.geometry.items()])
        with tempfile.TemporaryDirectory() as tmp:
            fn_geom = os.path.join(tmp, 'head.geom')
            fn_cond = os.path.join(tmp, 'head.cond')
            fn_elec = os.path.join(tmp, 'head.elec')
            write_geom_file(geom_out2inside, fn_geom)
            write_cond_file(self.cond, fn_cond)
            write_elec_file(self.elec_positions, fn_elec)
            self.geom, self.sens = create_geometry(fn_geom, fn_cond, fn_elec)
        self._A = None
        self._Ainv = None
        self._eitsm = None
        self._gain = None
        self._C = None
        self._V = None
        self._condition_nb = None
        self.first_derivatives = None
        if isinstance(self.sigma, np.ndarray) or isinstance(self.sigma, list) \
                or isinstance(self.sigma, dict):
            self.sigma = conductivity

    def dirichlet(self, neumann_data, exp_V_sens):
        gain_all = self.Ainv.dot(self.eitsm)
        u_minus_f = neumann_data - exp_V_sens
        ret = nandot(u_minus_f, gain_all.T)
        return ret


def om2np(om_matrix_tmp):
    np_matrix = np.zeros((om_matrix_tmp.nlin(), om_matrix_tmp.ncol()))
    for i in range(om_matrix_tmp.nlin()):
        for j in range(om_matrix_tmp.ncol()):
            np_matrix[i,j] = om_matrix_tmp(i,j)
    return np_matrix

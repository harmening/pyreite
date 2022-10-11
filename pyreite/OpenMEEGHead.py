#!/usr/bin/env python
from __future__ import print_function
import os, itertools, tempfile
from random import random 
from shutil import copyfile
import numpy as np, openmeeg as om
from pyreite.data_io import *
from pyreite.geometry import create_geometry
from collections import OrderedDict


class OpenMEEGHead(object):
    def __init__(self, conductivity, geometry, elec_positions, omega=None):      
        tmp = tempfile.mkdtemp()
        if isinstance(geometry, dict):
            geom_out2inside = OrderedDict([(tissue, bnd) for tissue, bnd in
                                           geometry.items()])
            #                              reversed(geometry.items())])
            fn_geom = os.path.join(tmp, 'geom_'+str(random()))
            write_geom_file(geom_out2inside, fn_geom)
            self.mesh_names = list(geom_out2inside.keys()) # out to inside
        else:
            raise ValueError
        self.geometry = geometry # outside to inside
        self.cond = conductivity
        fn_cond = os.path.join(tmp, 'cond_'+str(random()))
        write_cond_file(self.cond, fn_cond)
        self.elec_positions = elec_positions
        fn_elec = os.path.join(tmp, 'elec_'+str(random()))
        if isinstance(elec_positions, list) or isinstance(elec_positions, np.ndarray):
            write_elec_file(elec_positions, fn_elec)
        elif isinstance(elec_positions, str):
            copyfile(elec_positions, fn_elec)
        else:
            raise ValueError
        self.geom, self.sens = create_geometry(fn_geom, fn_cond, fn_elec)     
        for fn in [fn_geom, fn_cond, fn_elec]:
            os.remove(fn)    
        if isinstance(geometry, dict):
            for tissue in geometry.keys(): 
                os.remove(os.path.join(tmp, tissue+'.tri'))
        self.GAUSS_ORDER = 3
        #self.ind = self._get_indices_inside_out()
        self.ind = self._get_indices_outside_in()
        self._A = None
        self._Ainv = None
        self._h2em = None
        self._eitsm = None
        self._gain = None
        self._C = None
        self._V = None
        self._V_dip = None
        self._condition_nb = None
        self.omega = omega

    def _get_indices_outside_in(self):
        #idx = [i for i in range(self.geom.nb_meshes())]
        ind = {tissue: i for i, tissue in enumerate(self.mesh_names)}
        ind['V'] = []
        ind['p'] = []
        for m, mesh in enumerate(self.geom.meshes()):
            ind['V'].append([i.getindex() for i in mesh.vertices()])
            ind['p'].append([t.getindex() for t in mesh.iterator()])
            if m == 0:
                ind['p'][m] = []
        return ind
    
    @property
    def A(self):
        """Compute/return the attribute system matrix A"""
        if not self._A:
            self._A = om.HeadMat(self.geom, self.GAUSS_ORDER)
            #self._condition_nb = self.condition_nb
        return self._A
    @property
    def Ainv(self):
        """Compute/return the attribute system matrix A inverse"""
        if not self._Ainv:
            # deepcopy
            fn_A = os.path.join(tempfile.mkdtemp(), 'A_'+str(random()))
            self.A.save(fn_A)
            self._Ainv = self._A
            self._Ainv.invert()
            self._A = om.SymMatrix(fn_A)
            os.remove(fn_A)
        return self._Ainv
    @property
    def eitsm(self):
        """Compute/return the attribute righthandside for EIT"""
        if not self._eitsm:
            self._eitsm = om.EITSourceMat(self.geom, self.sens, self.GAUSS_ORDER)
        return self._eitsm
    @property
    def h2em(self):
        """Compute/return the mapping from outer boundary to electrodes"""
        if not self._h2em:
            self._h2em = om.Head2EEGMat(self.geom, self.sens)
        return self._h2em
    @property
    def gain(self):
        """Compute/return the EIT gain/leadfield matrix"""
        if not self._gain:
            self._gain = om.GainEEG(self.Ainv, self.eitsm, self.h2em)
        return self._gain
    @property
    def n_electrodes(self):
        return self.sens.getNumberOfSensors()

    @property
    def C(self):
        if not isinstance(self._C, np.ndarray):
            self._V, self._C = self._EIT_data(self.asnp(self.gain), self.sens)
        return self._C
    @property
    def V(self):
        if not isinstance(self._V, np.ndarray):
            self._V, self._C = self._EIT_data(self.asnp(self.gain), self.sens)
        return self._V

    def _EIT_data(self, G_eit, sens, freqs=[0], Iamp=[1], freq=0, ref='CAR',
                  excluded_chan=[], nonans=False): # set nonans=False
        n_elec = sens.getNumberOfSensors()
        sel_chan = range(1, n_elec+1)
        ei = lambda idx: np.array([0]*(idx)+[1]+[0]*(n_elec-idx-1))
        V = np.zeros((len(freqs), n_elec, n_elec, n_elec))
        C = np.zeros((len(freqs), G_eit.shape[0], G_eit.shape[1], G_eit.shape[0]))
        for i, freq in enumerate(freqs):
            for Source, Sink in itertools.product(sel_chan, sel_chan):
                C[i, Source-1,Sink-1,:] = np.multiply(np.array(ei(Source-1) - ei(Sink-1)),
                                                      Iamp[i]) # * freq (later?)
            this_V = np.einsum('ijl,kl', C[i], G_eit)
            V[i, :, :, :] = this_V
        if not nonans:
            for i, this_V in enumerate(V):
                for Source, Sink in itertools.product(sel_chan, sel_chan):
                    this_V[Source-1,Sink-1,Source-1]=np.nan
                    this_V[Source-1,Sink-1,Sink-1]=np.nan
                for e in excluded_chan:
                    this_V[e-1,:,:] = np.nan
                    this_V[:,e-1,:] = np.nan
                    this_V[:,:,e-1] = np.nan
                if ref == 'CAR':
                    this_V = np.subtract(this_V, np.nanmean(this_V, axis=2))
                else:
                    chans = [chan for chan in range(1,  n_elec+1) if chan not in excluded_chan]
                    if ref in chans:
                        this_V = np.subtract(this_V, this_V[:,:,ref-1])
                    else:
                        raise NotImplementedError
                    V[i,:,:,:] = this_V
        return V, C

    def asnp(self, openmeeg_matrix):
        """Convert openmeeg_matrix to numpy array"""
        return om2np(openmeeg_matrix)

    def set_cond(self, conductivity):
        self.cond = conductivity # inside out
        geom_out2inside = OrderedDict([(tissue, bnd) for tissue, bnd in
                                       reversed(self.geometry.items())])
        tmp = tempfile.mkdtemp()
        fn_geom = os.path.join(tmp, 'geom_'+str(random()))
        fn_cond = os.path.join(tmp, 'cond_'+str(random()))
        fn_elec = os.path.join(tmp, 'elec_'+str(random()))
        write_geom_file(geom_out2inside, fn_geom)
        write_cond_file(self.cond, fn_cond)
        write_elec_file(self.elec_positions, fn_elec)
        self.geom, self.sens = create_geometry(fn_geom, fn_cond, fn_elec)     
        for fn in [fn_geom, fn_cond, fn_elec]:
            os.remove(fn)    
        if isinstance(self.geometry, dict):
            for tissue in self.geometry.keys(): 
                os.remove(os.path.join(tmp, tissue+'.tri'))
        self._A = None
        self._Ainv = None
        self._eitsm = None
        self._gain = None
        self._C = None
        self._V = None
        self._condition_nb = None

    @property
    def condition_nb(self):
        if not self._condition_nb:
            self._condition_nb = np.linalg.cond(om2np(self.A))
            print('Condition number of system matrix: %f' % self._condition_nb)
        return self._condition_nb



def om2np(om_matrix_tmp):
    np_matrix = np.zeros((om_matrix_tmp.nlin(), om_matrix_tmp.ncol()))
    for i in range(om_matrix_tmp.nlin()):
        for j in range(om_matrix_tmp.ncol()):
            np_matrix[i,j] = om_matrix_tmp(i,j)
    return np_matrix

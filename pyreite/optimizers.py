import numpy as np
from collections import OrderedDict   
import pyreite
from pyreite.material_derivative import EIT_protocol, jacobian, hessian


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def printred(string):
	print(f"{bcolors.FAIL}%s{bcolors.ENDC}" % string)
def printyellow(string):
	print(f"{bcolors.WARNING}%s{bcolors.ENDC}" % string)
def printgreen(string):
	print(f"{bcolors.OKGREEN}%s{bcolors.ENDC}" % string)
def printblue(string):
	print(f"{bcolors.OKBLUE}%s{bcolors.ENDC}" % string)




def loss_residuals(cond, model, V_experiment, fixed=[], scale=False):
    if any([model.cond[tissue] != cond[tissue] for tissue in model.mesh_names]):
        model.set_cond(cond)
        printred("loss: SETTING NEW CONDUCTIVITY VALUES: "+str(cond))
    ND2V = EIT_protocol(num_elec=model.n_electrodes, n_freq=1, \
                        protocol='all_realistic')
    V = model.V
    V = V.flatten()[ND2V]
    if scale:
        V_experiment *= np.max(np.abs(V))
    dV = -(V_experiment-V)
    Error=0.5*np.nansum(pow(dV, 2));
    printgreen("Error: %f\n" % Error)
    return dV#, Error

def jac(cond, model, V_experiment, fixed=[]):
    if any([model.cond[tissue] != cond[tissue] for tissue in model.mesh_names]):
        model.set_cond(cond)
        printred("jac: SETTING NEW CONDUCTIVITY VALUES: "+str(cond))
    J = jacobian(cond, model)
    return J

def hess(cond, model, V_experiment, fixed=[]):
    if any([model.cond[tissue] != cond[tissue] for tissue in model.mesh_names]):
        model.set_cond(cond)
        printred("hess: SETTING NEW CONDUCTIVITY VALUES: "+str(cond))
    H = hessian(cond, model)
    return H

def jac_hess(cond, model, V_experiment, fixed=[]):
    if any([model.cond[tissue] != cond[tissue] for tissue in model.mesh_names]):
        model.set_cond(cond)
        printred("jac_hess: SETTING NEW CONDUCTIVITY VALUES: "+str(cond))
    J, model = jacobian(cond, model, return_model=True)
    H = hessian(cond, model)
    return J, H



def tikhonov(A, b, lamb, Lpr0):
	Lpr = pow(lamb,2) * Lpr0;
	# Moore-Penrose generalized inverse with Tikhonov regularization
	x = np.dot(np.linalg.pinv(A.conj().T.dot(A) + Lpr), A.conj().T.dot(b))
	return x

def levenberg_marquardt_hessian(A, dA, b, lamb, Lpr0):
    Lpr = pow(lamb,2) * Lpr0;
    # Moore-Penrose generalized inverse with Tikhonov regularization + Hessian
    Ainv = np.linalg.pinv(A.conj().T.dot(A) + Lpr)
    theta1 = np.dot(Ainv, A.conj().T.dot(b))
    b2 = theta1.dot(dA).T.dot(theta1) 
    theta2 = 0.5 * np.dot(Ainv, A.conj().T.dot(b2)) 
    x = theta1 + theta2
    return x

def is_posdef(M):
    w, _ = np.linalg.eigh(M) 
    return True if (w > 0).all() else False 

def levenberg_marquardt_hessiancheck(A, dA, b, lamb, Lpr0):
    Lpr = pow(lamb,2) * Lpr0;
    # Moore-Penrose generalized inverse with Tikhonov regularization + Hessian
    Ainv = np.linalg.pinv(A.conj().T.dot(A) + Lpr)
    theta1 = np.dot(Ainv, A.conj().T.dot(b))
    posdef_h = [is_posdef(dA[:,:,i]) for i in range(dA.shape[2])]
    dAsmall = dA[:,:,posdef_h]
    Ainvsmall = np.linalg.pinv(A[posdef_h].conj().T.dot(A[posdef_h]) + Lpr)    
    theta1small = np.dot(Ainvsmall, A[posdef_h].conj().T.dot(b[posdef_h]))
    b2small = theta1small.dot(dAsmall).T.dot(theta1small)
    theta2small = 0.5 * np.dot(Ainvsmall, A[posdef_h].conj().T.dot(b2small)) 
    x = theta1 + theta2small
    return x


def levenberg_marquardt_hessian_noser(A, dA, b, lamb, P=None, Q=None):
    tik_reg_param = pow(lamb,2)
    if not (isinstance(P, np.ndarray) or isinstance(P, list)):
        # NOSER (weigths Tikhonov regularization by jacobian sensitivity) 
        P = tik_reg_param*np.diag(np.diag(A.conj().T.dot(A)))
    if not (isinstance(Q, np.ndarray) or isinstance(Q, list)):
        Q = np.identity(A.shape[0])
    x_0 = np.zeros(A.shape[1]) # regularize ||x|| not ||x-x_0|| 
    x = np.dot(np.linalg.pinv(A.conj().T.dot(Q.dot(A)) + dA.dot(b) + P), \
               A.conj().T.dot(Q.dot(b-A.dot(x_0))))
    return x


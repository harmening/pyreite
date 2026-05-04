import numpy as np
from collections import OrderedDict
import pyreite
from pyreite.colors import bcolors, printred, printyellow, printgreen, printblue
from pyreite.EIThelpers import EIT_protocol, apply_protocol
from pyreite.material_derivative import jacobian, hessian




def loss_residuals(cond, model, V_experiment, fixed=[], scale=False, ND2V=None,
                   protocol=None):
    if any([model.cond[t] != cond[t] for t in model.mesh_names]):
        model.set_cond(cond)
        printred("loss: SETTING NEW CONDUCTIVITY VALUES: "+str(cond))
    V = model.V
    if protocol is not None:
        V = apply_protocol(V, protocol)
    else:
        if ND2V is None:
            ND2V = EIT_protocol(num_elec=model.n_electrodes, n_freq=1,
                                protocol='all_realistic')
        V = V.flatten()[ND2V]
    if scale:
        V_experiment = V_experiment * np.max(np.abs(V))
    dV = -(V_experiment-V)
    Error=0.5*np.nansum(dV**2);
    printgreen("Error: %f\n" % Error)
    return dV#, Error

def jac(cond, model, V_experiment, fixed=[], ND2V=None, protocol=None):
    if any([model.cond[tissue] != cond[tissue] for tissue in model.mesh_names]):
        model.set_cond(cond)
        printred("jac: SETTING NEW CONDUCTIVITY VALUES: "+str(cond))
    J = jacobian(cond, model, return_model=False, ND2V=ND2V, protocol=protocol)
    removed_cols = [idx for idx, tiss in enumerate(reversed(model.mesh_names)) if tiss in fixed]
    for idx in reversed(removed_cols):
        J = np.delete(J, idx, 1)  # delete idx column of J
    return J

def hess(cond, model, V_experiment, fixed=[], ND2V=None, protocol=None):
    if any([model.cond[tissue] != cond[tissue] for tissue in model.mesh_names]):
        model.set_cond(cond)
        printred("hess: SETTING NEW CONDUCTIVITY VALUES: "+str(cond))
    H = hessian(cond, model, ND2V=ND2V, protocol=protocol)
    return H

def jac_hess(cond, model, V_experiment, fixed=[], ND2V=None, protocol=None):
    if any([model.cond[tissue] != cond[tissue] for tissue in model.mesh_names]):
        model.set_cond(cond)
        printred("jac_hess: SETTING NEW CONDUCTIVITY VALUES: "+str(cond))
    J, model = jacobian(cond, model, return_model=True, ND2V=ND2V,
                        protocol=protocol)
    H = hessian(cond, model, ND2V=ND2V, protocol=protocol)
    return J, H


def tikhonov(A, b, lamb, Lpr0):
	Lpr = lamb**2 * Lpr0;
	# Moore-Penrose generalized inverse with Tikhonov regularization
	x = np.dot(np.linalg.inv(A.conj().T.dot(A) + Lpr), A.conj().T.dot(b))
	return x

def levenberg_marquardt_hessian(A, dA, b, lamb, Lpr0):
    Lpr = lamb**2 * Lpr0;
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
    Lpr = lamb**2 * Lpr0;
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


def _as_reg_matrix(Lpr0):
    """Accept vector or matrix; return a square np.array."""
    Lpr0 = np.asarray(Lpr0)
    if Lpr0.ndim == 1:
        return np.diag(Lpr0) # vector of precisions -> diagonal matrix
    return Lpr0


def levenberg_marquardt_hessiancheck_prior_centered(
        A, dA, b, lamb, Lpr0, x, mu):
    """
    Prior-centred LM with your 'x_new = x - x_step' convention.

    Solves: (A^H A + λ^2 L0) x_step = A^H b + λ^2 L0 (x - mu)
    then you should do: x_new = x - x_step.
    """
    L0 = np.asarray(Lpr0)
    if L0.ndim == 1:
        L0 = np.diag(L0)
    L = (lamb**2) * L0

    At = A.conj().T
    H  = At @ A + L

    # <-- this is the key difference vs your original LM:
    rhs = At @ b + L @ (x - mu)

    # 1st-order step
    theta1 = np.linalg.solve(H, rhs)

    # Hessian-based 2nd-order correction (your original structure)
    posdef_h = [is_posdef(dA[:, :, i]) for i in range(dA.shape[2])]
    if np.any(posdef_h):
        A_small = A[posdef_h]
        b_small = b[posdef_h]
        dA_small = dA[:, :, posdef_h]

        At_small = A_small.conj().T
        H_small  = At_small @ A_small + L
        rhs_small = At_small @ b_small + L @ (x - mu)

        theta1_small = np.linalg.solve(H_small, rhs_small)

        # second-order term: same pattern you had
        idxs = np.where(posdef_h)[0]
        b2_small = np.array([
            theta1_small @ (dA[:, :, k] @ theta1_small)
            for k in idxs
        ])

        theta2_small = 0.5 * np.linalg.solve(H_small, At_small @ b2_small)

        x_step = theta1 + theta2_small
    else:
        x_step = theta1

    return x_step






def build_Lpr0_from_J(J, prev_diag=None, floor=1e-6):
    """Return Lpr0 following Transtrum & Sethna damping scaling."""
    diagJ = np.einsum("ij,ij->j", J, J)  # diag(J^T J)
    if prev_diag is not None:
        diagJ = np.maximum(diagJ, prev_diag)  # 'largest so far'
    diagJ = np.maximum(diagJ, floor**2)
    Lpr0 = 1.0 / np.sqrt(diagJ)
    return Lpr0, diagJ

def build_Lpr0_combined(J, sigma_x, prev_diag=None, floor=1e-6):
    """
    Combine sensitivity-based scaling (Λ_sens) with prior precision (Λ_prior).
    Returns Lambda_total (vector) and the 'largest so far' diag(J^T J) for
    Minpack-style scaling à la Transtrum & Sethna.
    """
    # diag(J^T J)
    diagJ = np.einsum("ij,ij->j", J, J)

    # Minpack-style: keep the largest diag entries seen so far
    if prev_diag is not None:
        diagJ = np.maximum(diagJ, prev_diag)

    # Floor to avoid evaporation / zero curvature
    diagJ = np.maximum(diagJ, floor**2)

    # Sensitivity-based precision
    Lambda_sens = 1.0 / diagJ

    # Prior precision (sigma_x are prior stds in parameter space)
    sigma_x = np.asarray(sigma_x)
    Lambda_prior = 1.0 / (sigma_x**2)

    # Combined precision
    Lambda_total = Lambda_sens + Lambda_prior #Lpr0

    return Lambda_total, diagJ


def levenberg_marquardt_hessian_noser(A, dA, b, lamb, P=None, Q=None):
    tik_reg_param = lamb**2
    if not (isinstance(P, np.ndarray) or isinstance(P, list)):
        # NOSER (weigths Tikhonov regularization by jacobian sensitivity)
        P = tik_reg_param*np.diag(np.diag(A.conj().T.dot(A)))
    if not (isinstance(Q, np.ndarray) or isinstance(Q, list)):
        Q = np.identity(A.shape[0])
    x_0 = np.zeros(A.shape[1]) # regularize ||x|| not ||x-x_0||
    return np.dot(np.linalg.inv(A.conj().T.dot(Q.dot(A)) + dA.dot(b) + P),
                  A.conj().T.dot(Q.dot(b-A.dot(x_0))))


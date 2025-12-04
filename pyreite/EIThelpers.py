import itertools
import numpy as np



def scale_experiment_to_simulation(V_experiment: np.ndarray,
                                   V_sim: np.ndarray) -> np.ndarray:
    """Match experiment data scale to simulation data scale."""
    V_sim_mean = np.mean(V_sim)
    V_sim_std = np.std(V_sim)

    V_experiment_mean = np.mean(V_experiment)
    V_experiment_std = np.std(V_experiment)

    V_experiment_scaled = (
        (V_experiment - V_experiment_mean) / V_experiment_std
        * V_sim_std
        + V_sim_mean
    )
    return V_experiment_scaled




###### Protocol helpers ######

def apply_protocol(V4, protocol):
    if V4.ndim == 3:
        V4 = V4[np.newaxis, ...]
    assert V4.ndim == 4, "apply_protocol: V4 must be 4-dimensional " + \
                         "(n_freq, n_elec, n_elec, n_elec)"
    return (  V4[:, protocol[:,0], protocol[:,1], protocol[:,2]]
            - V4[:, protocol[:,0], protocol[:,1], protocol[:,3]]).reshape(-1)

def EIT_protocol(num_elec, n_freq=1, protocol='all'):
    if isinstance(protocol, list):
        ND2V = [False for _ in range(n_freq*num_elec*num_elec*num_elec)]
        for p in protocol:
            Source, Sink, measure = p[0], p[1], p[2]
            ND2V[Source*num_elec*num_elec + Sink*num_elec + measure] = True
        assert len(protocol) == np.sum(ND2V), "Invalid protocol: Protocol " + \
                "length does not match number of True values in ND2V. " + \
                "Check for duplicates or out-of-bound electrode indices."
        return ND2V
    if protocol == 'all':
        return [True for _ in range(n_freq*num_elec*num_elec*num_elec)]
    if protocol == 'all_realistic':
        ND2V = []
        for Source, Sink in itertools.product(range(num_elec), range(num_elec)):
            if Source < Sink:
                for measure in range(num_elec):
                    if measure not in [Source, Sink]:
                        ND2V.append(True)
                    else:
                        ND2V.append(False)
            else:
                for measure in range(num_elec):
                    ND2V.append(False)
        assert np.sum(ND2V) == num_elec*(num_elec-1)/2 * (num_elec-2)
        assert len(ND2V) == num_elec*num_elec*num_elec
        return ND2V

def build_full_protocol(n_elecs: int) -> np.ndarray:
    """Build the 4-electrode EIT protocol for n_elecs electrodes."""
    protocol = []
    for source, sink in itertools.product(range(n_elecs), range(n_elecs)):
        if source < sink:
            for m1, m2 in itertools.product(range(n_elecs), range(n_elecs)):
                # m1 and m2 are measuring electrodes, different from source/sink
                if (
                    m1 not in [source, sink, m2]
                    and m2 not in [source, sink, m1]
                    and m1 < m2
                ):
                    protocol.append((source, sink, m1, m2))

    protocol = np.array(protocol)
    return protocol






####### Fisher Information Matrix helpers ######

def compute_fim_from_J(J_log, noise_var=1.0):
    """
    Compute Fisher Information Matrix F_p for log-conductivities
    given a full stacked Jacobian J_log.

    Parameters
    ----------
    J_log : array, shape (M, P)
        Jacobian of residuals w.r.t. log-σ parameters.
    noise_var : float
        Assumed variance of measurement noise (σ_n^2).
        If residuals are whitened, you can set this to 1.0.

    Returns
    -------
    F_p : array, shape (P, P)
        Fisher Information Matrix in parameter space.
    """
    J_log = np.asarray(J_log)
    F_p = J_log.T @ J_log / noise_var
    return F_p

def compute_fim_from_blocks(J_blocks_log, noise_var=1.0):
    """
    Compute Fisher Information Matrix F_p by summing contributions
    from individual blocks J_k (one per drive pattern).

    Parameters
    ----------
    J_blocks_log : sequence of arrays
        J_blocks_log[k] has shape (M_k, P)
    noise_var : float
        Assumed variance of measurement noise.

    Returns
    -------
    F_p : array, shape (P, P)
        Fisher Information Matrix.
    """
    J_blocks_log = list(J_blocks_log)
    if not J_blocks_log:
        raise ValueError("J_blocks_log must be non-empty")

    P = J_blocks_log[0].shape[1]
    F_p = np.zeros((P, P), dtype=float)

    for Jk in J_blocks_log:
        Jk = np.asarray(Jk)
        assert Jk.shape[1] == P, "All blocks must have same #params"
        F_p += Jk.T @ Jk

    F_p /= noise_var
    return F_p


def d_optimality(F, param_indices=None, reg=1e-12):
    """
    D-optimality criterion: log(det(F_sub + reg*I)).
    """
    F = np.asarray(F)
    if param_indices is not None:
        idx = np.asarray(param_indices)
        F_sub = F[np.ix_(idx, idx)]
    else:
        F_sub = F

    Psub = F_sub.shape[0]
    F_sub_reg = F_sub + reg * np.eye(Psub)
    sign, logdet = np.linalg.slogdet(F_sub_reg)
    if sign <= 0:
        # Degenerate / singular; treat as very bad
        return -np.inf
    return logdet


def greedy_select_blocks_doptimal(J_blocks_log,
                                  num_select,
                                  noise_var=1.0,
                                  param_indices=None,
                                  reg=1e-12):
    """
    Greedy D-optimal subset selection of blocks based on FIM.

    Parameters
    ----------
    J_blocks_log : list of arrays
        J_blocks_log[k]: (M_k, P) Jacobian block for pattern k w.r.t. log-σ.
    num_select : int
        Number of blocks (patterns) to select.
    noise_var : float
        Measurement noise variance (σ_n^2).
    param_indices : list or array of int, optional
        Indices of parameters of interest for the design criterion.
        If None, all parameters are used.
    reg : float
        Small diagonal ridge for numerical stability in logdet.

    Returns
    -------
    selected_indices : list of int
        Indices (into J_blocks_log) of the selected patterns.
    F_p : array, shape (P, P)
        Final Fisher information matrix after selecting all patterns.
    history : list of dict
        Per-step info: [{'k': idx, 'criterion': value, 'F_p': ...}, ...]
        (F_p in history are copies at each step; can omit for memory).
    """
    J_blocks_log = list(J_blocks_log)
    K = len(J_blocks_log)
    if num_select > K:
        raise ValueError("num_select cannot exceed number of blocks")

    P = J_blocks_log[0].shape[1]
    F_p = np.zeros((P, P), dtype=float)
    selected = []
    available = set(range(K))
    history = []

    for t in range(num_select):
        best_gain = -np.inf
        best_k = None
        best_F_p_new = None

        for k in available:
            Jk = np.asarray(J_blocks_log[k])
            F_p_candidate = F_p + (Jk.T @ Jk) / noise_var

            crit = d_optimality(F_p_candidate,
                                param_indices=param_indices,
                                reg=reg)

            if crit > best_gain:
                best_gain = crit
                best_k = k
                best_F_p_new = F_p_candidate

        if best_k is None:
            # No improvement possible (degenerate case)
            break

        selected.append(best_k)
        available.remove(best_k)
        F_p = best_F_p_new

        history.append({
            "step": t,
            "k": best_k,
            "criterion": best_gain,
            # "F_p": F_p.copy(),  # uncomment if you want snapshots
        })

    return selected, F_p, history


def info_metric(F, metric="min_eig", param_indices=None, reg=1e-12):
    F = np.asarray(F)
    if param_indices is not None:
        idx = np.asarray(param_indices)
        F = F[np.ix_(idx, idx)]

    if metric == "min_eig":
        # smallest eigenvalue (after small ridge)
        F_reg = F + reg * np.eye(F.shape[0])
        w = np.linalg.eigvalsh(F_reg)
        return float(w[0])

    elif metric == "logdet":
        # D-optimality: log det(F + reg*I)
        F_reg = F + reg * np.eye(F.shape[0])
        sign, logdet = np.linalg.slogdet(F_reg)
        if sign <= 0:
            return -np.inf
        return float(logdet)

    else:
        raise ValueError(f"Unknown metric '{metric}'")


def greedy_select_until_full(
    J_blocks_log,
    noise_var=1.0,
    param_indices=None,
    metric="min_eig",
    target_frac=0.9,
    reg=1e-12,
    max_blocks=None,
):
    """
    Greedy selection of blocks until the info metric reaches
    a target fraction of the full FIM.

    Parameters
    ----------
    J_blocks_log : list of arrays
        J_blocks_log[k]: (M_k, P) Jacobian blocks (log-space).
    noise_var : float
        Measurement noise variance (σ_n^2).
    param_indices : list of int or None
        Parameter indices of interest (for metric). If None, use all params.
    metric : {"min_eig","logdet"}
        Information metric.
    target_frac : float
        Stop when metric(F_sub) >= target_frac * metric(F_all).
    reg : float
        Small ridge for stability in metric.
    max_blocks : int or None
        Optional safety cap on number of selected blocks.

    Returns
    -------
    selected : list of int
        Indices of chosen blocks.
    F_sub : array, shape (P,P)
        FIM from selected blocks.
    history : list of dict
        Info per iteration: metric values, chosen k, etc.
    """
    # --- full FIM and target metric ---
    F_all = compute_fim_from_blocks(J_blocks_log, noise_var=noise_var)
    target_val = info_metric(F_all, metric=metric,
                             param_indices=param_indices, reg=reg)

    J_blocks_log = list(J_blocks_log)
    K = len(J_blocks_log)
    P = J_blocks_log[0].shape[1]

    F_sub = np.zeros((P, P), dtype=float)
    selected = []
    available = set(range(K))
    history = []

    if max_blocks is None:
        max_blocks = K

    current_val = info_metric(F_sub, metric=metric,
                              param_indices=param_indices, reg=reg)

    for t in range(max_blocks):
        best_gain = -np.inf
        best_k = None
        best_F_new = None
        best_val = None

        for k in available:
            Jk = np.asarray(J_blocks_log[k])
            F_candidate = F_sub + (Jk.T @ Jk) / noise_var
            val = info_metric(F_candidate, metric=metric,
                              param_indices=param_indices, reg=reg)

            if val > best_gain:
                best_gain = val
                best_k = k
                best_F_new = F_candidate
                best_val = val

        if best_k is None:
            break

        # update
        selected.append(best_k)
        available.remove(best_k)
        F_sub = best_F_new
        current_val = best_val

        frac = current_val / target_val if target_val > 0 else 0.0
        history.append({
            "step": t,
            "k": best_k,
            "metric_value": current_val,
            "fraction_of_full": frac,
        })

        # stopping condition
        if frac >= target_frac:
            break

    return selected, F_sub, history, F_all, target_val

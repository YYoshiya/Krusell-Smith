import numpy as np
from scipy.linalg import inv
from scipy import interpolate
import time
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import time 
import multiprocessing as multi
from dataclasses import dataclass, field

@dataclass
class KSSolution:
    k_opt: np.ndarray  # 3D array of float64
    value: np.ndarray  # 3D array of float64
    B: np.ndarray      # 1D array of float64 (vector)
    R2: np.ndarray     # 1D array of float64 (vector)

    def __init__(self, k_opt_shape, value_shape, B_length, R2_length):
        self.k_opt = np.zeros(k_opt_shape, dtype=np.float64)
        self.value = np.zeros(value_shape, dtype=np.float64)
        self.B = np.zeros(B_length, dtype=np.float64)
        self.R2 = np.zeros(R2_length, dtype=np.float64)
@dataclass
class TransitionMatrix:
    P: np.ndarray       # 4x4
    Pz: np.ndarray      # 2x2 aggregate shock
    Peps_gg: np.ndarray # 2x2 idiosyncratic shock conditional on good to good
    Peps_bb: np.ndarray # 2x2 idiosyncratic shock conditional on bad to bad
    Peps_gb: np.ndarray # 2x2 idiosyncratic shock conditional on good to bad
    Peps_bg: np.ndarray # 2x2 idiosyncratic shock conditional on bad to good

def create_transition_matrix(ug, ub, zg_ave_dur, zb_ave_dur, ug_ave_dur, ub_ave_dur, puu_rel_gb2bb, puu_rel_bg2gg):
    # Probability of remaining in good state
    pgg = 1 - 1 / zg_ave_dur
    # Probability of remaining in bad state
    pbb = 1 - 1 / zb_ave_dur
    # Probability of changing from g to b
    pgb = 1 - pgg
    # Probability of changing from b to g
    pbg = 1 - pbb  
    
    # Probability of 0 to 0 cond. on g to g
    p00_gg = 1 - 1 / ug_ave_dur
    # Probability of 0 to 0 cond. on b to b
    p00_bb = 1 - 1 / ub_ave_dur
    # Probability of 0 to 1 cond. on g to g
    p01_gg = 1 - p00_gg
    # Probability of 0 to 1 cond. on b to b
    p01_bb = 1 - p00_bb
    
    # Probability of 0 to 0 cond. on g to b
    p00_gb = puu_rel_gb2bb * p00_bb
    # Probability of 0 to 0 cond. on b to g
    p00_bg = puu_rel_bg2gg * p00_gg
    # Probability of 0 to 1 cond. on g to b
    p01_gb = 1 - p00_gb
    # Probability of 0 to 1 cond. on b to g
    p01_bg = 1 - p00_bg
    
    # Probability of 1 to 0 cond. on g to g
    p10_gg = (ug - ug * p00_gg) / (1 - ug)
    # Probability of 1 to 0 cond. on b to b
    p10_bb = (ub - ub * p00_bb) / (1 - ub)
    # Probability of 1 to 0 cond. on g to b
    p10_gb = (ub - ug * p00_gb) / (1 - ug)
    # Probability of 1 to 0 cond. on b to g
    p10_bg = (ug - ub * p00_bg) / (1 - ub)
    # Probability of 1 to 1 cond. on g to g
    p11_gg = 1 - p10_gg
    # Probability of 1 to 1 cond. on b to b
    p11_bb = 1 - p10_bb
    # Probability of 1 to 1 cond. on g to b
    p11_gb = 1 - p10_gb
    # Probability of 1 to 1 cond. on b to g
    p11_bg = 1 - p10_bg
    
    # Constructing the transition matrices
    P = np.array([[pgg * p11_gg, pgb * p11_gb, pgg * p10_gg, pgb * p10_gb],
                  [pbg * p11_bg, pbb * p11_bb, pbg * p10_bg, pbb * p10_bb],
                  [pgg * p01_gg, pgb * p01_gb, pgg * p00_gg, pgb * p00_gb],
                  [pbg * p01_bg, pbb * p01_bb, pbg * p00_bg, pbb * p00_bb]])
    
    Pz = np.array([[pgg, pgb],
                   [pbg, pbb]])
    
    Peps_gg = np.array([[p11_gg, p10_gg],
                        [p01_gg, p00_gg]])
    
    Peps_bb = np.array([[p11_bb, p10_bb],
                        [p01_bb, p00_bb]])
    
    Peps_gb = np.array([[p11_gb, p10_gb],
                        [p01_gb, p00_gb]])
    
    Peps_bg = np.array([[p11_bg, p10_bg],
                        [p01_bg, p00_bg]])

    transmat = TransitionMatrix(P, Pz, Peps_gg, Peps_bb, Peps_gb, Peps_bg)
    return transmat

# wage function
def w(z, K, L):
    return (1-alpha)*z*K**(alpha)*L**(-alpha)

# interest rate function
def r(z, K, L):
    return alpha*z*K**(alpha-1)*L**(1-alpha)


# utility function 
def utility(x): 
    if sigma==1:
        return  np.log(x) 
    else:
        return x**(1-sigma) / (1-sigma)
    
def rhs_bellman():
    z, eps = s_grid[s_i, 0], s_grid[s_i, 2]
    Kp, L = compute_Kp_L()
    c = (r(alpha,z,K,L)+1-delta)*k+w(alpha,z,K,L)*(eps*l_bar+(1.0-eps)*mu)-kp 
    expec = compute_expectation(kp,Kp,value,s_i,ksp)
    return u(c)+beta*expec

def compute_expectation(kp, Kp, value, s_i):
    expec = 0
    for s_n_i in range(4):
        value_itp = interpolate.interp1d(k, value[:,:,s_n_i], kind='linear')
        expec += transmat.P[s_i, s_n_i] * value_itp(kp, Kp)
    return expec

def maximize_rhs(k_i, K_i, s_i):
    k_min, k_max = ksp.k_grid[0], ksp.k_grid[-1]
    k=ksp.k_grid[k_i]
    K=ksp.K_grid[K_i]
    z, eps = s_grid[s_i, 0], s_grid[s_i, 1]
    Kp, L = compute_Kp_L(K,s_i)
    k_c_pos = (r(z,K,L)+1-delta)*k+w(z,K,L)*(eps*l_bar+(1.0-eps)*mu)
    obj = -rhs_bellman(kp, k, K, s_i)
    res = minimize_scalar(obj, bounds=(k_min, min(k_c_pos, k_max)), method='bounded')
    
    # 最適化結果の取得
    kss.k_opt[k_i, K_i, s_i] = res.x
    kss.value[k_i, K_i, s_i] = -res.fun  # 最大化された値（マイナス符号を戻す）



def solve_ump():
    counter_VFI = 0
    while True:
        counter_VFI += 1
        value_old = np.copy(kss.value)
        for k_i in range(ksp['k_size']):
            for K_i in range(ksp['K_size']):
                for s_i in range(ksp['s_size']):
                    maximize_rhs(k_i, K_i, s_i, ksp, kss)
        iterate_policy()
        dif = maximum(abs, value_old - kss.value)
        if dif < tol:
            break


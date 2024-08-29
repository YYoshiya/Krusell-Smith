import numpy as np
from scipy.linalg import inv
from scipy import interpolate
import time
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from scipy.interpolate import RegularGridInterpolator
import time 
import multiprocessing as multi
from dataclasses import dataclass, field
import quantecon as qe
from quantecon import MarkovChain
from tqdm import tqdm  # For progress bar
import statsmodels.api as sm

class KSSolution:
    def __init__(self, k_opt, value, B, R2):
        self.k_opt = k_opt
        self.value = value
        self.B = B
        self.R2 = R2

def KSSolution_initializer(ksp, filename="result.h5"):
    # Initialize k_opt
    k_opt = ksp['beta'] * np.repeat(ksp['k_grid'][:, np.newaxis, np.newaxis], 
                                    repeats=[ksp['K_size'], ksp['s_size']], axis=1)
    k_opt = 0.9 * np.repeat(ksp['k_grid'][:, np.newaxis, np.newaxis], 
                            repeats=[ksp['K_size'], ksp['s_size']], axis=1)
    k_opt = np.clip(k_opt, ksp['k_min'], ksp['k_max'])

    # Initialize value function
    value = ksp['u'](0.1 / 0.9 * k_opt) / (1 - ksp['beta'])

    # Initialize B
    B = np.array([0.0, 1.0, 0.0, 1.0])
    
    kss = KSSolution(k_opt, value, B, [0.0, 0.0])
    return kss
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


def generate_shocks(z_shock_size, population):
    mc = MarkovChain(transmat.Pz)
    zi_shock = mc.simulate(ts_length=z_shock_size)
    # idiosyncratic shocks
    epsi_shock = np.empty((z_shock_size, population), dtype=int)
    rand_draw = np.random.rand(population)
    if zi_shock[0] == 1:  # if good
        epsi_shock[0, :] = (rand_draw < ksp.ug).astype(int) + 1
    elif zi_shock[0] == 2:  # if bad
        epsi_shock[0, :] = (rand_draw < ksp.ub).astype(int) + 1
    else:
        raise ValueError(f"the value of zi_shock[0] ({zi_shock[0]}) is strange")
    
    for t in range(1, z_shock_size):
        draw_eps_shock_wrapper(zi_shock[t], zi_shock[t-1], epsi_shock[t, :], epsi_shock[t-1, :], transmat)

    for t in range(z_shock_size):
        n_e = np.sum(epsi_shock[t, :] == 1)  # Count number of employed
        empl_rate_ideal = 1.0 - ksp['ug'] if zi_shock[t] == 1 else 1.0 - ksp['ub']
        gap = round(empl_rate_ideal * population) - n_e
        
        if gap > 0:
            # Select unemployed individuals to become employed
            unemployed_indices = np.where(epsi_shock[t, :] == 2)[0]
            if len(unemployed_indices) > 0:
                become_employed_i = np.random.choice(unemployed_indices, gap, replace=False)
                epsi_shock[t, become_employed_i] = 1
        elif gap < 0:
            # Select employed individuals to become unemployed
            employed_indices = np.where(epsi_shock[t, :] == 1)[0]
            if len(employed_indices) > 0:
                become_unemployed_i = np.random.choice(employed_indices, -gap, replace=False)
                epsi_shock[t, become_unemployed_i] = 2
    
    return zi_shock, epsi_shock

#Define the main function that does the work
def draw_eps_shock(epsi_shocks, epsi_shock_before, Peps):
    # loop over entire population
    for i in range(len(epsi_shocks)):
        rand_draw = np.random.rand()
        if epsi_shock_before[i] == 1:
            epsi_shocks[i] = 1 if Peps[0, 0] >= rand_draw else 2  # Employed before
        else:
            epsi_shocks[i] = 1 if Peps[1, 0] >= rand_draw else 2  # Unemployed before

# Wrapper function that selects the correct transition matrix
def draw_eps_shock_wrapper(zi, zi_lag, epsi_shocks, epsi_shock_before, transmat):
    if zi == 1 and zi_lag == 1:
        Peps = transmat['Peps_gg']
    elif zi == 1 and zi_lag == 2:
        Peps = transmat['Peps_bg']
    elif zi == 2 and zi_lag == 1:
        Peps = transmat['Peps_gb']
    elif zi == 2 and zi_lag == 2:
        Peps = transmat['Peps_bb']
    else:
        raise ValueError("Invalid zi or zi_lag value")
    
class Stochastic:
    def __init__(self, epsi_shocks, k_population):
        self.epsi_shocks = epsi_shocks  # Should be a 2D list or a NumPy array of integers
        self.k_population = k_population 


def simulate_aggregate_path(ksp, kss, zi_shocks, K_ts, sm):
    epsi_shocks = sm.epsi_shocks
    k_population = sm.k_population

    T = len(zi_shocks)  # simulated duration
    N = epsi_shocks.shape[1]  # number of agents

    # Loop over T periods with progress bar
    for t, z_i in enumerate(tqdm(zi_shocks, desc="Simulating aggregate path", mininterval=0.5)):
        K_ts[t] = np.mean(k_population)  # current aggregate capital
        
        # Loop over individuals
        for i, k in enumerate(k_population):
            eps_i = epsi_shocks[t, i]  # idiosyncratic shock
            s_i = epsi_zi_to_si(eps_i, z_i, ksp['z_size'])  # transform (z_i, eps_i) to s_i
            
            # Obtain next capital holding by interpolation
            itp_pol = RegularGridInterpolator((ksp['k_grid'], ksp['K_grid']), kss['k_opt'][:, :, s_i])
            k_population[i] = itp_pol((k, K_ts[t]))

    return None

def epsi_zi_to_si(eps_i, z_i, z_size):
    return z_i + z_size * (eps_i - 1)

def regress_ALM(ksp, kss, zi_shocks, K_ts, T_discard=100):
    n_g = np.sum(zi_shocks[T_discard:-1] == 1)
    n_b = np.sum(zi_shocks[T_discard:-1] == 2)
    B_n = np.empty(4)
    x_g = np.empty(n_g)
    y_g = np.empty(n_g)
    x_b = np.empty(n_b)
    y_b = np.empty(n_b)
    
    i_g, i_b = 0, 0
    
    for t in range(T_discard, len(zi_shocks) - 1):
        if zi_shocks[t] == 1:
            x_g[i_g] = np.log(K_ts[t])
            y_g[i_g] = np.log(K_ts[t + 1])
            i_g += 1
        else:
            x_b[i_b] = np.log(K_ts[t])
            y_b[i_b] = np.log(K_ts[t + 1])
            i_b += 1
    
    X_g = sm.add_constant(x_g)
    X_b = sm.add_constant(x_b)
    
    resg = sm.OLS(y_g, X_g).fit()
    resb = sm.OLS(y_b, X_b).fit()
    
    kss.R2 = [resg.rsquared, resb.rsquared]
    B_n[0], B_n[1] = resg.params
    B_n[2], B_n[3] = resb.params
    
    dif_B = np.max(np.abs(B_n - kss.B))
    print(f"difference of ALM coefficient is {B_n}")
    
    return B_n, dif_B

def find_ALM_coef(umpsm, sm, ksp, kss, zi_shocks, tol_ump=1e-8, max_iter_ump=100,
                  tol_B=1e-8, max_iter_B=20, update_B=0.3, T_discard=100):
    
    K_ts = np.empty(len(zi_shocks))
    counter_B = 0
    
    while True:
        counter_B += 1
        print(f" --- Iteration over ALM coefficient: {counter_B} ---")
        
        # Solve individual problem
        solve_ump(umpsm, ksp, kss, max_iter=max_iter_ump, tol=tol_ump)
        
        # Compute aggregate path of capital
        simulate_aggregate_path(ksp, kss, zi_shocks, K_ts, sm)
        
        # Obtain new ALM coefficient by regression
        B_n, dif_B = regress_ALM(ksp, kss, zi_shocks, K_ts, T_discard=T_discard)
        
        # Check convergence
        if dif_B < tol_B:
            print("-----------------------------------------------------")
            print(f"ALM coefficient successfully converged : dif = {dif_B}")
            print("-----------------------------------------------------")
            break
        elif counter_B == max_iter_B:
            print("----------------------------------------------------------------")
            print(f"Iteration over ALM coefficient reached its maximum ({max_iter_B})")
            print("----------------------------------------------------------------")
            break
        
        # Update B
        kss.B = update_B * B_n + (1 - update_B) * kss.B
    
    return K_ts
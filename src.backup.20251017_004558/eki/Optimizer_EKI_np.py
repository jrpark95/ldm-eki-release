"""
Ensemble Kalman Inversion (EKI) Optimizer with NumPy

This module implements various EKI methods including:
- Standard EnKF
- Adaptive EnKF
- EnKF with Localization
- EnKF with Barrier Method
- EnRML, EnKF-MDA, and Regularized EnKF (REnKF)
"""

import os
import numpy as np
import Model_Connection_np_Ensemble as Model_np

def Run(input_config={}, input_data={}):
    """
    Main EKI optimization loop.

    Args:
        input_config: Configuration dictionary containing algorithm parameters
        input_data: Input data dictionary

    Returns:
        Tuple containing optimization results and convergence metrics
    """
    np.random.seed(0)
    inverse = Inverse(input_config)
    model = Model_np.Model(input_config, input_data)

    time = np.arange(input_config['time'], dtype=int)
    iteration = np.arange(input_config['iteration'], dtype=int)

    state_update_list = []
    state_update_iteration_list = []

    misfits_list = []
    misfits_list_tmp = []  # temporary list for convergence check
    noises_list = []
    residuals_list = []
    discrepancy_bools_list = []
    residual_bools_list = []

    misfit_list = []
    misfit_list_tmp = []  # temporary list for convergence check
    noise_list = []
    residual_list = []
    discrepancy_bool_list = []
    residual_bool_list = []
    ensXiter_list = []
    diff_list = []
    
    info_list = []

    for t in time:
        if t == 0:
            state_predict = model.make_ensemble()
        else:
            state_predict = model.predict(state_update, t)
        ob, ob_err = model.get_ob(t)
        obs = np.tile(ob, (input_config['sample'], 1)).T
        if input_config['perturb_option'] == 'On_time':
            obs = _perturb(ob, ob_err, input_config['sample'])

        alpha_inv_history = []
        T_n = 0.0

        for i in iteration:
            print(f"Iteration {i+1}/{len(iteration)}")
            if input_config['perturb_option'] == 'On_iter':
                obs = _perturb(ob, ob_err, input_config['sample'])

            # Forward model: state → observation
            state_in_ob = model.state_to_ob(state_predict)

            # Alpha calculation for Adaptive EKI
            if input_config['Adaptive_EKI'] == 'On':
                Phi_n = compute_Phi_n(obs, state_in_ob, ob_err)
                alpha_inv = compute_alpha_inv(len(state_in_ob), Phi_n, alpha_inv_history, i)
                alpha_inv_history.append(alpha_inv)
                if len(alpha_inv_history) > 1 and (alpha_inv >= 1.0 - T_n or alpha_inv < 0.0):
                    print(f"Adaptive EKI converged at iteration {i+1}")
                    break
                T_n += alpha_inv

            # Inverse methods (choose one): EnKF, EnRML, EnKF_MDA, REnKF, etc.
            if input_config.get('Localized_EKI') == 'On':
                if input_config.get('Adaptive_EKI') == 'On':
                    state_update = inverse.Adaptive_EnKF_with_Localizer(i, state_predict, state_in_ob, obs, ob_err, ob, alpha_inv)
                else:
                    state_update = inverse.EnKF_with_Localizer(i, state_predict, state_in_ob, obs, ob_err, ob)
            elif input_config.get('Adaptive_EKI') == 'On':
                state_update = inverse.Adaptive_EnKF(i, state_predict, state_in_ob, obs, ob_err, ob, alpha_inv)
            elif input_config.get('Regularization') == 'On':
                state_update = inverse.REnKF(i, state_predict, state_in_ob, obs, ob_err, ob)
            else:
                state_update = inverse.EnKF(i, state_predict, state_in_ob, obs, ob_err, ob)
            
            misfits = np.mean(state_update,1) - np.mean(state_predict, 1)
            state_predict = state_update.copy()
            state_update_iteration_list.append(state_update.copy())

            # Save iteration data for visualization
            iteration_dir = 'logs/eki_iterations'
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir, exist_ok=True)
            np.save(f'{iteration_dir}/iteration_{i+1:03d}.npy', state_update)

            # Convergence check
            misfits_list_tmp.append(misfits)
            misfits_list.append(misfits.copy())
            misfits_err = np.zeros(misfits.shape)            
            discrepancy_bools, residual_bools, residuals, noises = _convergence(misfits_list_tmp, misfits_err)
            discrepancy_bools_list.append(discrepancy_bools)
            residual_bools_list.append(residual_bools)
            residuals_list.append(np.asarray(residuals).copy())
            noises_list.append(noises.copy())

            misfit = np.linalg.norm(np.mean(obs - state_in_ob, axis=1))
            misfit_list_tmp.append(misfit)
            misfit_list.append(misfit.copy())
            discrepancy_bool, residual_bool, residual, noise = _convergence(misfit_list_tmp, ob_err)
            discrepancy_bool_list.append(discrepancy_bool)
            residual_bool_list.append(residual_bool)
            residual_list.append(np.asarray(residual).copy())
            noise_list.append(noise.copy())
            
            ensXiter = 0
            if residual_bool:
                ensXiter = input_config['sample'] * (i+1)
                ensXiter_list.append(ensXiter)
            else:
                ensXiter_list.append(ensXiter)        

        # Error check (skip if shapes don't match - happens when using more obs timesteps than state timesteps)
        if np.mean(state_update.copy(), 1).shape == model.real_state_init.shape:
            diff = np.abs(np.mean(state_update.copy(), 1) - model.real_state_init)/model.real_state_init
            diff_list.append(diff)
        else:
            diff_list.append(np.zeros_like(model.real_state_init))
        
        state_update_list.append(state_update.copy())

    if input_config['time'] == 1:
        state_update_list = state_update_list[0]
        diff_list = diff_list[0]

    return (state_update_list, state_update_iteration_list, info_list, misfit_list,
            discrepancy_bool_list, residual_bool_list, residual_list, noise_list,
            ensXiter_list, diff_list, misfits_list, discrepancy_bools_list,
            residual_bools_list, residuals_list)


class Inverse(object):
    """
    Inverse modeling methods for EKI.

    Supports multiple variants:
    - EnKF: Standard Ensemble Kalman Filter
    - Adaptive_EnKF: Adaptive step size EKI
    - EnKF_with_Localizer: Localized covariance
    - EnRML: Ensemble Randomized Maximum Likelihood
    - EnKF_MDA: Multiple Data Assimilation
    - REnKF: Regularized EnKF with constraints
    """
    def __init__(self, input_config):
        self.sample = input_config['sample']
        self.input_config = input_config    
        self.time = 0
        self.alpha = input_config['EnKF_MDA_steps']
        self.beta = input_config['EnRML_step_length']
        self.lambda_value= input_config['REnKF_lambda']
        self.weighting_factor = input_config['Localization_weighting_factor']

    def EnKF_with_barrier(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        EnKF with Barrier Method for box constraints.

        Applies logarithmic barrier functions to enforce bounds during updates.
        Note: This method is kept for reference but not used by default.
        """
        tau = 0.5
        inflation = 0.9
        lower = 1.0e-6
        upper = 1.0e+15

        if iteration == 0:
            x0 = _ave_substracted(state_predict)  # (d x N)
            self.C0 = (1.0 / (self.sample - 1.0)) * np.dot(x0, x0.T)

        x = _ave_substracted(state_predict)      # (d x N)
        hx = _ave_substracted(state_in_ob)         # (d x N)
        pxz = 1.0 / (self.sample - 1.0) * np.dot(x, hx.T)
        pzz = 1.0 / (self.sample - 1.0) * np.dot(hx, hx.T)
        k = np.dot(pxz, np.linalg.pinv(pzz + ob_err))
        dx = np.dot(k, obs - state_in_ob)
        state_update = state_predict + dx

        C_hat = (1.0 / (self.sample - 1.0)) * np.dot(x, x.T)

        d = state_predict.shape[0]
        lower_bounds = np.full(d, lower)
        upper_bounds = np.full(d, upper)
        state_mean = np.mean(state_predict, axis=1)
        eps = 1e-10
        barrier_grad = (1.0/(upper_bounds - state_mean + eps) - 1.0/(state_mean - lower_bounds + eps))
        barrier_term = (1.0 / tau) * np.dot(C_hat, barrier_grad)
        barrier_term_expanded = np.tile(barrier_term[:, np.newaxis], (1, self.sample))
        state_update = state_update + barrier_term_expanded

        if inflation != 1.0:
            mean_state = np.mean(state_update, axis=1, keepdims=True)
            state_update = mean_state + inflation * (state_update - mean_state)

        return state_update

    def EnKF(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        Standard Ensemble Kalman Filter update.

        Computes Kalman gain and updates ensemble members based on
        innovation (observation - predicted observation).
        """
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)
        k = np.dot(pxz, np.linalg.pinv(pzz+ob_err))
        dx = np.dot(k, obs-state_in_ob)
        state_update = np.array(state_predict) + dx
        return state_update

    def Adaptive_EnKF(self, iteration, state_predict, state_in_ob, obs, ob_err, ob, alpha_inv):
        """
        Adaptive EnKF with automatic step size control.

        Uses adaptive alpha parameter to adjust step size based on
        ensemble spread and data misfit.
        """
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)
        alpha = 1.0/alpha_inv
        k_modified = np.dot(pxz, np.linalg.pinv(pzz + alpha * ob_err))
        xi = _perturb(np.zeros(ob.shape), ob_err, self.sample)
        perturbed_diff = obs + np.sqrt(alpha) * xi - state_in_ob
        dx = np.dot(k_modified, perturbed_diff)
        state_update = np.array(state_predict) + dx
        return state_update

    @staticmethod
    def centralized_localizer(matrix, L):
        """
        Apply Gaspari-Cohn localization to covariance matrix.

        Args:
            matrix: Covariance matrix to localize
            L: Localization length scale

        Returns:
            Localized covariance matrix
        """
        distances1 = compute_distances(matrix.shape[0])
        distances2 = compute_distances(matrix.shape[1])

        # Compute the localization matrices using Gaussian taper
        Psi1 = np.vectorize(lambda d: np.exp(-d**2 / (2*L**2)))(distances1)
        Psi2 = np.vectorize(lambda d: np.exp(-d**2 / (2*L**2)))(distances2)

        # Apply element-wise localization
        localized_matrix = matrix * Psi1 * Psi2
        return localized_matrix

    def EnKF_with_Localizer(self, iteration, state_predict, state_in_ob, obs, ob_err, ob,
                           localizer_func=None):
        """
        EnKF with covariance localization.

        Reduces spurious correlations in covariance estimation by applying
        localization functions to the covariance matrices.
        """
        if localizer_func is None:
            localizer_func = Inverse.centralized_localizer
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)

        # Apply localization to covariances
        pxz = localizer_func(pxz, self.weighting_factor)
        pzz = localizer_func(pzz, self.weighting_factor)

        k_modified = np.dot(pxz, np.linalg.pinv(pzz + ob_err))
        dx = np.dot(k_modified, obs-state_in_ob)
        state_update = state_predict + dx
        return state_update

    def Adaptive_EnKF_with_Localizer(self, iteration, state_predict, state_in_ob, obs, ob_err, ob,
                                     alpha_inv, localizer_func=None):
        """
        Adaptive EnKF with covariance localization.

        Combines adaptive step size control with localized covariances.
        """
        if localizer_func is None:
            localizer_func = Inverse.centralized_localizer
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)

        # Apply localization to covariances
        pxz = localizer_func(pxz, self.weighting_factor)
        pzz = localizer_func(pzz, self.weighting_factor)

        alpha = 1.0/alpha_inv
        k_modified = np.dot(pxz, np.linalg.pinv(pzz + alpha * ob_err))

        xi = _perturb(np.zeros(ob.shape), ob_err, self.sample)
        perturbed_diff = obs + np.sqrt(alpha) * xi - state_in_ob

        dx = np.dot(k_modified, perturbed_diff)
        state_update = state_predict + dx
        return state_update

    def EnRML(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        Ensemble Randomized Maximum Likelihood method.

        Iterative ensemble smoother based on Gauss-Newton optimization.
        """
        if iteration == 0:
            self.state0 = state_predict
        x0 = _ave_substracted(self.state0)
        p0 = 1.0/(self.sample-1.0) * np.dot(x0, x0.T)
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        sen = np.dot(hx, np.linalg.pinv(x))
        p0_sen = np.dot(p0, sen.T)
        gn_sub = np.dot(np.dot(sen, p0), sen.T) + ob_err
        gn = np.dot(p0_sen, np.linalg.inv(gn_sub))
        dx = np.dot(gn, obs-state_in_ob) + \
             np.dot(gn, np.dot(sen, state_predict - self.state0))
        state_update = self.beta*self.state0 + (1.0-self.beta)*state_predict + self.beta*dx
        return state_update

    def EnKF_MDA(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        Ensemble Kalman Filter with Multiple Data Assimilation.

        Assimilates the same data multiple times with inflated observation error
        to improve convergence.
        """
        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)
        k_modified = np.dot(pxz, np.linalg.inv(pzz + self.alpha*ob_err))
        obs_origin = np.tile(ob, (obs.shape[1], 1)).T
        ob_perturb = obs - obs_origin
        obs_mda = obs_origin + np.sqrt(self.alpha)*ob_perturb
        dx = np.dot(k_modified, obs_mda - state_in_ob)
        state_update = state_predict + dx
        return state_update

    def REnKF(self, iteration, state_predict, state_in_ob, obs, ob_err, ob):
        """
        Regularized Ensemble Kalman Filter.

        Applies penalty-based constraints to enforce physical bounds
        and regularization during the update step.
        """
        def tanh_penalty(x):
            return np.tanh(x)

        def tanh_penalty_derivative(x):
            return 1 - np.tanh(x) ** 2

        def constraint_func(x):
            penalty = np.zeros_like(x)
            penalty[x < 0.0] = tanh_penalty(x[x < 0.0])
            penalty[x > 1.0e+15] = tanh_penalty(x[x > 1.0e+15] - 1.0e+15)
            return penalty

        def constraint_derivative(x):
            derivative = np.zeros_like(x)
            derivative[x < 0] = tanh_penalty_derivative(x[x < 0])
            derivative[x > 1e+15] = tanh_penalty_derivative(x[x > 1e+15] - 1e+15)
            return derivative

        def lambda_function(iteration):
            lambda_value = self.lambda_value
            return lambda_value

        x = _ave_substracted(state_predict)
        hx = _ave_substracted(state_in_ob)
        pxz = 1.0/(self.sample-1.0) * np.dot(x, hx.T)
        pzz = 1.0/(self.sample-1.0) * np.dot(hx, hx.T)
        k = np.dot(pxz, np.linalg.pinv(pzz+ob_err))
        dx = np.dot(k, obs-state_in_ob)
        pxx = 1.0/(self.sample-1.0) * np.dot(x, x.T)
        pzx = 1.0/(self.sample-1.0) * np.dot(hx, x.T)
        k_constraints = -pxx + np.dot(k, pzx)

        constraint_mat = np.zeros([len(state_predict), self.sample])
        w = np.identity(state_predict.shape[0])
        lamda = lambda_function(iteration)

        for j in range(self.sample):
            jstate = state_predict[:, j]
            gw = np.dot(constraint_derivative(jstate).T, w)
            gwg = np.dot(gw, constraint_func(jstate))
            constraint_mat[:, j] += lamda * gwg

        dx_constraints = np.dot(k_constraints, constraint_mat)
        state_update = state_predict + dx + dx_constraints
        return state_update


# Utility Functions

def _perturb(ave, cov, samp, perturb=1e-30):
    """
    Generate perturbed ensemble from mean and covariance.

    Args:
        ave: Mean vector
        cov: Covariance matrix
        samp: Number of samples
        perturb: Small value added to diagonal for numerical stability

    Returns:
        Perturbed ensemble matrix (dim x samp)
    """
    dim = len(ave)
    cov += np.eye(dim) * perturb
    chol_decomp = np.linalg.cholesky(cov)
    corr_perturb = np.random.normal(loc=0.0, scale=1.0, size=(dim, samp))
    perturbation = np.matmul(chol_decomp, corr_perturb)
    get_perturb = np.tile(ave, (samp, 1)).T + perturbation
    return get_perturb


def _ave_substracted(m):
    """
    Subtract ensemble mean from each member.

    Args:
        m: Ensemble matrix (dim x samp)

    Returns:
        Mean-subtracted ensemble (dim x samp)
    """
    samp = m.shape[1]
    ave = np.array([np.mean(m, 1)])
    ave = np.tile(ave.T, (1, samp))
    ave_substracted = np.array(m) - ave
    return ave_substracted


def _convergence(misfit_list, ob_err, noise_factor=1.0, min_residual=1e-6):
    """
    Check convergence based on discrepancy principle and residual change.

    Args:
        misfit_list: List of misfit values over iterations
        ob_err: Observation error covariance
        noise_factor: Multiplier for noise level threshold
        min_residual: Minimum relative residual for convergence

    Returns:
        Tuple: (discrepancy_bool, residual_bool, residual, noise)
    """
    # Discrepancy principle
    if np.any(ob_err) == True:
        noise_level = np.sqrt(np.trace(ob_err))
        noise = noise_factor * noise_level
    else:
        noise = noise_factor * ob_err
    discrepancy_bool = misfit_list[-1] < noise

    # Residual of misfit
    mIter = len(misfit_list) -1
    if mIter == 0:
        residual = np.nan
        residual_bool = False
    else:
        residual = abs(misfit_list[mIter]-misfit_list[mIter-1])
        residual /= abs(misfit_list[0])
        residual_bool = residual < min_residual
    return discrepancy_bool, residual_bool, residual, noise


def sec_fisher(cov, N_ens):
    """
    Secondary Fisher information matrix with ensemble correction.

    Applies statistical correction to covariance matrix based on ensemble size.

    Args:
        cov: Covariance matrix
        N_ens: Ensemble size

    Returns:
        Corrected covariance matrix
    """
    rows, cols = cov.shape

    # If the matrix is square, process normally
    if rows == cols:
        v = np.sqrt(np.diag(cov))
        V = np.diag(v)
        V_inv = np.linalg.inv(V)
        R = V_inv @ cov @ V_inv
    else:
        # For non-square matrices, just compute the normalized matrix
        row_scales = np.sqrt(np.diag(np.dot(cov, cov.T)))
        col_scales = np.sqrt(np.diag(np.dot(cov.T, cov)))
        R = (cov / row_scales[:, None]) / col_scales[None, :]
    
    R_sec = np.zeros_like(R)
    for i in range(rows):
        for j in range(min(i + 1, cols)):
            r = R[i, j]
            if (i == j) | (r >= 1):
                R_sec[i, j] = 1
            else:
                s = np.arctanh(r)
                σ_s = 1 / np.sqrt(N_ens - 3)
                σ_r = (np.tanh(s + σ_s) - np.tanh(s - σ_s)) / 2
                Q = r / σ_r
                alpha = (Q**2) / (1 + Q**2)
                R_sec[i, j] = alpha * r

    if rows == cols:
        return V @ R_sec @ V
    else:
        return (R_sec * row_scales[:, None]) * col_scales[None, :]


def compute_Phi_n(y, state_in_ob, Gamma):
    """
    Compute normalized data misfit for adaptive EKI.

    Args:
        y: Observations
        state_in_ob: Predicted observations
        Gamma: Observation error covariance

    Returns:
        Vector of normalized misfits for each ensemble member
    """
    results_matrix = np.dot(np.linalg.pinv(Gamma)**0.5, (y - state_in_ob))
    phi_n = np.linalg.norm(results_matrix, ord=2, axis=0)
    return phi_n


def compute_alpha_inv(M, Phi_n, alpha_inv_history, n):
    """
    Compute adaptive step size parameter for EKI.

    Args:
        M: Number of observations
        Phi_n: Normalized data misfit vector
        alpha_inv_history: List of previous alpha_inv values
        n: Current iteration number

    Returns:
        Adaptive step size parameter alpha_inv
    """
    # Empirical mean and variance of Phi_n
    Phi_mean = np.mean(Phi_n)
    Phi_var = np.var(Phi_n)

    # Compute cumulative step size
    t_n = sum(alpha_inv_history[:n]) if n != 1 else 0

    # Compute alpha_inv with bounds
    alpha_inv = min(max(M/(2.0 * Phi_mean), np.sqrt(M/(2.0 * Phi_var))), 1.0 - t_n)

    return alpha_inv


def compute_distances(size):
    """
    Compute pairwise distance matrix for localization.

    Args:
        size: Dimension of the state space

    Returns:
        Distance matrix (size x size)
    """
    return np.abs(np.arange(size) - np.arange(size)[:, np.newaxis])
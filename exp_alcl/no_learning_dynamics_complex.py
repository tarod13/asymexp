import numpy as np
import scipy.linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_stochastic_matrix(n):
    P = np.random.rand(n, n)**10
    P -= np.min(P)
    P /= np.sum(P, axis=1, keepdims=True)
    return P

def generate_graph_laplacian(n, delta):
    P = generate_stochastic_matrix(n)
    L = np.eye(n)*(1+delta) - P 
    return L

def compute_eigen_decomposition(L):
    eigvals, eigvecs_left, eigvecs_right = scipy.linalg.eig(L, left=True, right=True)
    eigval = eigvals[0].real
    u = eigvecs_left[:, 0].real
    v = eigvecs_right[:, 0].real
    uTv = u.T @ v
    u /= uTv
    return eigval, u, v

def generate_initial_conditions(n):
    xr0 = np.random.randn(n, 1)
    xi0 = np.random.randn(n, 1)
    yr0 = np.random.randn(n, 1)
    yi0 = np.random.randn(n, 1)
    return xr0, xi0, yr0, yi0

def compute_min_norm_control(grad_r, grad_i, drift_val, epsilon=1e-8):
    """
    Solves the QP: min ||u||^2 s.t. u.grad + drift <= 0
    Solution: u = - max(0, drift) * grad / ||grad||^2
    """
    grad_sq_norm = np.sum(grad_r**2) + np.sum(grad_i**2)
    
    # If gradient is zero (at optimum), no control needed
    if grad_sq_norm < epsilon:
        return np.zeros_like(grad_r), np.zeros_like(grad_i), 0.0

    # The constraint violation (if positive, we need control)
    psi = drift_val
    
    if psi <= 0:
        return np.zeros_like(grad_r), np.zeros_like(grad_i), 0.0
    
    # Exact projection scalar
    gamma = psi / grad_sq_norm
    
    u_r = -gamma * grad_r
    u_i = -gamma * grad_i
    
    return u_r, u_i, gamma

def control_norm_exact(xr, xi, fx_bar, fi_bar, lambda_reg):
    """
    V = 0.5 * ( ||x||^2 - 1 )^2
    grad V = e_x * 2x
    Lf V = grad V . f = 2 * e_x * (x.f)
    """
    # 1. Compute Error and V
    norm_sq = np.sum(xr**2) + np.sum(xi**2)
    e_x = norm_sq - 1.0
    V = 0.5 * e_x**2

    # 2. Compute Gradient of V w.r.t x (real and imag)
    # grad_xr = 2 * e_x * xr
    # grad_xi = 2 * e_x * xi
    grad_xr = 2 * e_x * xr
    grad_xi = 2 * e_x * xi
    
    # 3. Compute Lie Derivative Lf V (Drift)
    # dot(grad, f)
    dot_prod = np.sum(grad_xr * fx_bar) + np.sum(grad_xi * fi_bar)
    Lf_V = dot_prod
    
    # 4. Total Drift required to be cancelled
    # We want dot(u, grad) + Lf_V + lambda*V <= 0
    drift_total = Lf_V + lambda_reg * V
    
    # 5. Solve QP
    ux_r, ux_i, gamma = compute_min_norm_control(grad_xr, grad_xi, drift_total)
    
    return ux_r, ux_i, e_x, gamma

def control_phase_exact(xr, xi, yr, yi, fxr, fxi, fyr, fyi, lambda_phi):
    """
    V = 0.5 * ( Im(y^H x) )^2 = 0.5 * e_phi^2
    e_phi = yr.xi - yi.xr
    """
    # 1. Compute Error and V
    term1 = np.sum(yr * xi)
    term2 = np.sum(yi * xr)
    e_phi = term1 - term2
    V = 0.5 * e_phi**2
    
    # 2. Compute Gradients of V w.r.t all state variables
    # grad V = e_phi * grad(e_phi)
    # grad_xr(e_phi) = -yi  => grad_xr(V) = -e_phi * yi
    # grad_xi(e_phi) = +yr  => grad_xi(V) = +e_phi * yr
    # grad_yr(e_phi) = +xi  => grad_yr(V) = +e_phi * xi
    # grad_yi(e_phi) = -xr  => grad_yi(V) = -e_phi * xr
    
    g_xr = -e_phi * yi
    g_xi = +e_phi * yr
    g_yr = +e_phi * xi
    g_yi = -e_phi * xr
    
    # 3. Compute Lie Derivative Lf V
    # sum(grad . f) over all 4 vectors
    Lf_V = (np.sum(g_xr * fxr) + np.sum(g_xi * fxi) + 
            np.sum(g_yr * fyr) + np.sum(g_yi * fyi))
            
    # 4. Total Drift
    drift_total = Lf_V + lambda_phi * V
    
    # 5. Solve QP (Combined state vector effectively)
    # We need to treat the 4 vectors as one giant vector for the projection
    grad_sq_norm = (np.sum(g_xr**2) + np.sum(g_xi**2) + 
                    np.sum(g_yr**2) + np.sum(g_yi**2))
    
    if grad_sq_norm < 1e-8 or drift_total <= 0:
        return np.zeros_like(xr), np.zeros_like(xr), np.zeros_like(xr), np.zeros_like(xr), e_phi, 0.0
        
    gamma = drift_total / grad_sq_norm
    
    ux_r = -gamma * g_xr
    ux_i = -gamma * g_xi
    uy_r = -gamma * g_yr
    uy_i = -gamma * g_yi
    
    return ux_r, ux_i, uy_r, uy_i, e_phi, gamma

def dynamics(L, xr, xi, yr, yi, lambda_x, lambda_y, lambda_phi):
    xr, xi = xr.reshape(-1,1), xi.reshape(-1,1)
    yr, yi = yr.reshape(-1,1), yi.reshape(-1,1)

    # --- 1. Natural Flow (Eigenvalue Search) ---
    Lxr = L @ xr; Lxi = L @ xi
    LTyr = L.T @ yr; LTyi = L.T @ yi

    # Estimates
    denom_x = np.sum(xr**2) + np.sum(xi**2) + 1e-8 # Safety for alpha calc
    alpha = (np.sum(xr * Lxr) + np.sum(xi * Lxi)) / denom_x
    beta  = (np.sum(xr * Lxi) - np.sum(xi * Lxr)) / denom_x
    
    # Base Vector Fields f(x)
    fx_r_bar = -Lxr + alpha * xr - beta * xi
    fx_i_bar = -Lxi + beta * xr  + alpha * xi
    
    # Left vector uses conjugate eigenvalue (alpha - i*beta)
    fy_r_bar = -LTyr + alpha * yr + beta * yi
    fy_i_bar = -LTyi - beta * yr  + alpha * yi

    # --- 2. Exact Control Inputs ---
    
    # A. Norm Control (Calculated independently for X and Y)
    ux_r_norm, ux_i_norm, e_x, b_x = control_norm_exact(xr, xi, fx_r_bar, fx_i_bar, lambda_x)
    uy_r_norm, uy_i_norm, e_y, b_y = control_norm_exact(yr, yi, fy_r_bar, fy_i_bar, lambda_y)

    # Apply norm controls to get "Stabilized" flow for Phase calculation?
    # Strictly speaking, Lf V includes ALL f. 
    # To avoid solving a coupled system of equations, we can assume the controls act 
    # on the base field f_bar. This is valid if constraints are orthogonal-ish, 
    # or we can just sum them up (superposition of controls). 
    # Given phase control is orthogonal to norm control (rotation vs scaling), 
    # calculating them based on f_bar is theoretically sound.
    
    # B. Phase Control
    ux_r_phi, ux_i_phi, uy_r_phi, uy_i_phi, e_phi, b_phi = control_phase_exact(
        xr, xi, yr, yi, 
        fx_r_bar, fx_i_bar, fy_r_bar, fy_i_bar, 
        lambda_phi
    )

    # --- 3. Total Dynamics ---
    fx_r = fx_r_bar + ux_r_norm + ux_r_phi
    fx_i = fx_i_bar + ux_i_norm + ux_i_phi
    fy_r = fy_r_bar + uy_r_norm + uy_r_phi
    fy_i = fy_i_bar + uy_i_norm + uy_i_phi

    aux = {
        'alpha': alpha, 'beta': beta,
        'e_x': e_x, 'e_y': e_y, 'e_phi': e_phi,
        'b_x': b_x, 'b_y': b_y, 'b_phi': b_phi
    }
    
    return (fx_r, fx_i, fy_r, fy_i), aux

def step(L, xr, xi, yr, yi, lr, lambda_x, lambda_y, lambda_phi):
    (fxr, fxi, fyr, fyi), aux = dynamics(L, xr, xi, yr, yi, lambda_x, lambda_y, lambda_phi)

    # Standard Euler step
    xr_new = xr + lr * fxr
    xi_new = xi + lr * fxi
    yr_new = yr + lr * fyr
    yi_new = yi + lr * fyi

    # Log gradient norm for sanity check
    gn_x = np.sqrt(np.sum(fxr**2) + np.sum(fxi**2))
    
    # We do NOT normalize gradients here anymore. 
    # The exact QP formulation scales the control forces precisely to match the drift.
    # Normalizing them would break the physical guarantee of the Lyapunov derivative.

    return (xr_new, xi_new, yr_new, yi_new), aux, {'gn': gn_x}

def simulate_dynamics(L, xr0, xi0, yr0, yi0, lr, lambda_x, lambda_y, lambda_phi, n_steps):
    xr, xi = xr0, xi0
    yr, yi = yr0, yi0
    
    traj_x, traj_y, aux_list = [], [], []
    
    for _ in tqdm(range(n_steps), desc="Simulating"):
        (xr, xi, yr, yi), aux, _ = step(L, xr, xi, yr, yi, lr, lambda_x, lambda_y, lambda_phi)
        traj_x.append(xr + 1j*xi)
        traj_y.append(yr + 1j*yi)
        aux_list.append(aux)
    
    x_t = np.concatenate([v.T for v in traj_x], axis=0)
    y_t = np.concatenate([v.T for v in traj_y], axis=0)
    t = np.arange(n_steps)

    aux_arrays = {}
    for key in aux_list[0]:
        val_list = [a[key] for a in aux_list]
        aux_arrays[key] = np.array(val_list).flatten() if np.ndim(val_list[0])==0 else np.array(val_list).squeeze()

    return t, x_t, y_t, aux_arrays

def visualize_metrics(t, x_t, y_t, true_u, true_v, true_eigval, aux_arrays):    
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.plot(t, np.abs(aux_arrays['e_x']), label=r'$e_x$')
    plt.plot(t, np.abs(aux_arrays['e_y']), label=r'$e_y$')
    plt.title('Norm Errors (Exact QP)')
    plt.yscale('log')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(t, aux_arrays['e_phi'], label=r'$e_{\phi}$', color='purple')
    plt.title('Phase Locking Error')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(t, aux_arrays['alpha'], label=r'$\hat{\alpha}$')
    plt.plot(t, aux_arrays['beta'], label=r'$\hat{\beta}$')
    plt.axhline(true_eigval.real, color='k', linestyle='--')
    plt.axhline(true_eigval.imag, color='r', linestyle='--')
    plt.title('Eigenvalues')
    plt.legend()

    plt.subplot(2, 3, 4)
    sim_v = np.abs(np.matmul(x_t.conj(), true_v)) / (np.linalg.norm(x_t, axis=1) * np.linalg.norm(true_v))
    sim_u = np.abs(np.matmul(y_t.conj(), true_u)) / (np.linalg.norm(y_t, axis=1) * np.linalg.norm(true_u))
    plt.plot(t, sim_v, label='Right Sim')
    plt.plot(t, sim_u, label='Left Sim')
    plt.title('Ground Truth Similarity')
    plt.ylim([0, 1.05])
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(t, aux_arrays['b_x'], label='$b_x$ (Lagrange Mult)')
    plt.plot(t, aux_arrays['b_phi'], label='$b_{\phi}$')
    plt.title('Control Effort (Gamma)')
    plt.legend()
    plt.yscale('log')

    plt.subplot(2, 3, 6)
    dot_prods = np.sum(y_t.conj() * x_t, axis=1)
    plt.plot(t, dot_prods.real, label='Real')
    plt.plot(t, dot_prods.imag, label='Imag')
    plt.title('Inner Product y^H x')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(n, delta, lr, lambda_x, lambda_y, lambda_phi, n_steps):
    L = generate_graph_laplacian(n, delta)
    eigval, u, v = compute_eigen_decomposition(L)
    xr, xi, yr, yi = generate_initial_conditions(n)

    t, x_t, y_t, aux_arrays = simulate_dynamics(
        L, xr, xi, yr, yi, lr, lambda_x, lambda_y, lambda_phi, n_steps
    )
    
    visualize_metrics(t, x_t, y_t, u, v, eigval, aux_arrays)

if __name__ == "__main__":
    n = 20
    delta = 0.1
    lr = 0.002
    # Lambdas are now strictly decay rates for the Lyapunov function (time constants)
    lambda_x = 5.0 
    lambda_y = 5.0
    lambda_phi = 10.0
    n_steps = 10000
    
    main(n, delta, lr, lambda_x, lambda_y, lambda_phi, n_steps)
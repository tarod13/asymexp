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

def compute_eigen_decomposition(L, k):
    # Get all eigenvalues
    eigvals, eigvecs_left, eigvecs_right = scipy.linalg.eig(L, left=True, right=True)
    
    # Sort by real part (smallest first for Laplacian)
    idx = np.argsort(eigvals.real)
    
    # Select bottom k
    vals = eigvals[idx[:k]]
    U = eigvecs_left[:, idx[:k]]
    V = eigvecs_right[:, idx[:k]]
    
    # Normalize pairs individually for metric tracking
    for i in range(k):
        dot = np.vdot(U[:, i], V[:, i])
        U[:, i] /= np.conj(dot) # Normalize so u^H v = 1
        
    return vals, U, V

def generate_initial_conditions(n, k):
    xr0 = np.random.randn(n, k)
    xi0 = np.random.randn(n, k)
    yr0 = np.random.randn(n, k)
    yi0 = np.random.randn(n, k)
    return xr0, xi0, yr0, yi0

def compute_min_norm_control(grad_r, grad_i, drift_val, epsilon=1e-8):
    grad_sq_norm = np.sum(grad_r**2) + np.sum(grad_i**2)
    
    if grad_sq_norm < epsilon:
        return np.zeros_like(grad_r), np.zeros_like(grad_i), 0.0

    psi = drift_val
    if psi <= 0:
        return np.zeros_like(grad_r), np.zeros_like(grad_i), 0.0
    
    gamma = psi / grad_sq_norm
    u_r = -gamma * grad_r
    u_i = -gamma * grad_i
    return u_r, u_i, gamma

def control_norm_exact(xr, xi, fx_bar, fi_bar, lambda_reg):
    # xr, xi are vectors (n, 1)
    norm_sq = np.sum(xr**2) + np.sum(xi**2)
    e_x = norm_sq - 1.0
    V = 0.5 * e_x**2
    
    grad_xr = 2 * e_x * xr
    grad_xi = 2 * e_x * xi
    
    Lf_V = np.sum(grad_xr * fx_bar) + np.sum(grad_xi * fi_bar)
    drift_total = Lf_V + lambda_reg * V
    
    # Fix: Unpack and return e_x
    ux_r, ux_i, gamma = compute_min_norm_control(grad_xr, grad_xi, drift_total)
    return ux_r, ux_i, e_x, gamma

def control_phase_exact(xr, xi, yr, yi, fxr, fxi, fyr, fyi, lambda_phi):
    # Phase locking between current pair x_k, y_k
    term1 = np.sum(yr * xi)
    term2 = np.sum(yi * xr)
    e_phi = term1 - term2
    V = 0.5 * e_phi**2
    
    g_xr = -e_phi * yi
    g_xi = +e_phi * yr
    g_yr = +e_phi * xi
    g_yi = -e_phi * xr
    
    Lf_V = (np.sum(g_xr * fxr) + np.sum(g_xi * fxi) + 
            np.sum(g_yr * fyr) + np.sum(g_yi * fyi))
            
    drift_total = Lf_V + lambda_phi * V
    
    grad_sq_norm = (np.sum(g_xr**2) + np.sum(g_xi**2) + 
                    np.sum(g_yr**2) + np.sum(g_yi**2))
    
    if grad_sq_norm < 1e-8 or drift_total <= 0:
        return np.zeros_like(xr), np.zeros_like(xr), np.zeros_like(xr), np.zeros_like(xr), e_phi, 0.0
        
    gamma = drift_total / grad_sq_norm
    return -gamma*g_xr, -gamma*g_xi, -gamma*g_yr, -gamma*g_yi, e_phi, gamma

def control_orth_exact(target_r, target_i, others_r, others_i, f_target_r, f_target_i, lambda_orth):
    """
    Ensures 'target' vector is orthogonal to all 'others'.
    Constraint: target \perp others_j  =>  <others_j, target> = 0 (Complex inner product)
    """
    n_others = others_r.shape[1]
    if n_others == 0:
        return np.zeros_like(target_r), np.zeros_like(target_i), 0.0, 0.0

    # 1. Compute Inner Products (Complex)
    # Broadcast computation: (n, 1) vs (n, k_prev)
    dot_r = np.sum(others_r * target_r, axis=0) + np.sum(others_i * target_i, axis=0) 
    dot_i = np.sum(others_r * target_i, axis=0) - np.sum(others_i * target_r, axis=0) 
    
    # 2. Total Lyapunov Energy
    V = 0.5 * (np.sum(dot_r**2) + np.sum(dot_i**2))
    
    # 3. Gradients w.r.t target vector (xr, xi)
    grad_xr = (others_r @ dot_r) - (others_i @ dot_i) 
    grad_xr = grad_xr.reshape(-1, 1)

    grad_xi = (others_i @ dot_r) + (others_r @ dot_i)
    grad_xi = grad_xi.reshape(-1, 1)
    
    # 4. Drift
    Lf_V = np.sum(grad_xr * f_target_r) + np.sum(grad_xi * f_target_i)
    drift_total = Lf_V + lambda_orth * V
    
    # 5. Solve QP
    u_r, u_i, gamma = compute_min_norm_control(grad_xr, grad_xi, drift_total)
    
    return u_r, u_i, V, gamma

def dynamics(L, XR, XI, YR, YI, lambda_x, lambda_y, lambda_phi, lambda_orth):
    n, k = XR.shape
    
    FX_R, FX_I = np.zeros_like(XR), np.zeros_like(XI)
    FY_R, FY_I = np.zeros_like(YR), np.zeros_like(YI)
    
    aux_k = [] 

    for i in range(k):
        xr, xi = XR[:, i:i+1], XI[:, i:i+1]
        yr, yi = YR[:, i:i+1], YI[:, i:i+1]
        
        # 1. Natural Flow
        Lxr = L @ xr; Lxi = L @ xi
        LTyr = L.T @ yr; LTyi = L.T @ yi

        denom_x = np.sum(xr**2) + np.sum(xi**2) + 1e-8
        alpha = (np.sum(xr * Lxr) + np.sum(xi * Lxi)) / denom_x
        beta  = (np.sum(xr * Lxi) - np.sum(xi * Lxr)) / denom_x
        
        fx_r_bar = -Lxr + alpha * xr - beta * xi
        fx_i_bar = -Lxi + beta * xr  + alpha * xi
        fy_r_bar = -LTyr + alpha * yr + beta * yi
        fy_i_bar = -LTyi - beta * yr  + alpha * yi
        
        # 2. Controls
        
        # A. Norm Control (FIXED unpacking)
        ux_r_n, ux_i_n, e_x, b_x = control_norm_exact(xr, xi, fx_r_bar, fx_i_bar, lambda_x)
        uy_r_n, uy_i_n, e_y, b_y = control_norm_exact(yr, yi, fy_r_bar, fy_i_bar, lambda_y)
        
        # B. Orthogonality Control
        if i > 0:
            prev_Y_r = YR[:, :i]
            prev_Y_i = YI[:, :i]
            prev_X_r = XR[:, :i]
            prev_X_i = XI[:, :i]
            
            ux_r_o, ux_i_o, V_orth_x, b_orth_x = control_orth_exact(
                xr, xi, prev_Y_r, prev_Y_i, fx_r_bar, fx_i_bar, lambda_orth
            )
            uy_r_o, uy_i_o, V_orth_y, b_orth_y = control_orth_exact(
                yr, yi, prev_X_r, prev_X_i, fy_r_bar, fy_i_bar, lambda_orth
            )
        else:
            ux_r_o, ux_i_o, V_orth_x, b_orth_x = 0, 0, 0, 0
            uy_r_o, uy_i_o, V_orth_y, b_orth_y = 0, 0, 0, 0

        # C. Phase Locking
        f_total_x_r = fx_r_bar + ux_r_n + ux_r_o
        f_total_x_i = fx_i_bar + ux_i_n + ux_i_o
        f_total_y_r = fy_r_bar + uy_r_n + uy_r_o
        f_total_y_i = fy_i_bar + uy_i_n + uy_i_o

        ux_r_p, ux_i_p, uy_r_p, uy_i_p, e_phi, b_phi = control_phase_exact(
            xr, xi, yr, yi, f_total_x_r, f_total_x_i, f_total_y_r, f_total_y_i, lambda_phi
        )

        # 3. Summation
        FX_R[:, i:i+1] = f_total_x_r + ux_r_p
        FX_I[:, i:i+1] = f_total_x_i + ux_i_p
        FY_R[:, i:i+1] = f_total_y_r + uy_r_p
        FY_I[:, i:i+1] = f_total_y_i + uy_i_p
        
        aux_k.append({
            'alpha': alpha, 'beta': beta,
            'e_x': e_x, 'e_y': e_y, 
            'e_phi': e_phi, 'V_orth_x': V_orth_x,
            'b_x': b_x, 'b_phi': b_phi, 'b_orth_x': b_orth_x
        })

    return (FX_R, FX_I, FY_R, FY_I), aux_k

def step(L, xr, xi, yr, yi, lr, lambda_x, lambda_y, lambda_phi, lambda_orth):
    (fxr, fxi, fyr, fyi), aux_k = dynamics(L, xr, xi, yr, yi, lambda_x, lambda_y, lambda_phi, lambda_orth)

    xr_new = xr + lr * fxr
    xi_new = xi + lr * fxi
    yr_new = yr + lr * fyr
    yi_new = yi + lr * fyi
    
    return (xr_new, xi_new, yr_new, yi_new), aux_k

def simulate_dynamics(L, XR0, XI0, YR0, YI0, lr, lambda_x, lambda_y, lambda_phi, lambda_orth, n_steps):
    XR, XI = XR0, XI0
    YR, YI = YR0, YI0
    
    traj_x = []
    traj_y = []
    aux_history = []
    
    for step_i in tqdm(range(n_steps), desc="Simulating"):
        (XR, XI, YR, YI), aux_k = step(L, XR, XI, YR, YI, lr, lambda_x, lambda_y, lambda_phi, lambda_orth)
        
        if step_i % 10 == 0:
            traj_x.append(XR + 1j*XI)
            traj_y.append(YR + 1j*YI)
            aux_history.append(aux_k)
    
    X_T = np.stack(traj_x, axis=0)
    Y_T = np.stack(traj_y, axis=0)
    t = np.arange(0, n_steps, 10)

    return t, X_T, Y_T, aux_history

def visualize_metrics(t, X_T, Y_T, true_U, true_V, true_vals, aux_history):    
    k = X_T.shape[2]
    
    plt.figure(figsize=(18, 12))

    # 1. Eigenvalues
    plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    for i in range(k):
        alpha_t = [aux[i]['alpha'] for aux in aux_history]
        plt.plot(t, alpha_t, color=colors[i], label=f'Est Re[{i}]' if i==0 else "")
        plt.axhline(true_vals[i].real, color=colors[i], linestyle='--', alpha=0.5)
    plt.title('Real Part of Eigenvalues')
    plt.xlabel('Time')

    plt.subplot(2, 3, 2)
    for i in range(k):
        beta_t = [aux[i]['beta'] for aux in aux_history]
        plt.plot(t, beta_t, color=colors[i], linestyle=':', label=f'Est Im[{i}]' if i==0 else "")
        plt.axhline(true_vals[i].imag, color=colors[i], linestyle='--', alpha=0.5)
    plt.title('Imag Part of Eigenvalues')
    
    # 2. Similarity to Ground Truth
    plt.subplot(2, 3, 3)
    for i in range(k):
        xt = X_T[:, :, i]
        v = true_V[:, i]
        sim = np.abs(np.matmul(xt.conj(), v)) / (np.linalg.norm(xt, axis=1) * np.linalg.norm(v))
        plt.plot(t, sim, color=colors[i], label=f'Pair {i}')
    plt.title('Convergence to True Eigenvectors')
    plt.ylim([0, 1.05])
    plt.legend()
    
    # 3. Orthogonality Check
    plt.subplot(2, 3, 4)
    final_X = X_T[-1, :, :]
    final_Y = Y_T[-1, :, :]
    orth_matrix = np.abs(final_Y.T.conj() @ final_X)
    plt.imshow(orth_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Magnitude |y_i^H x_j|')
    plt.title(f'Final Biorthogonality (T={t[-1]})')
    plt.xlabel('Right Vector Index')
    plt.ylabel('Left Vector Index')

    # 4. Orthogonality Control Effort
    plt.subplot(2, 3, 5)
    for i in range(1, k): 
        b_orth = [aux[i]['b_orth_x'] for aux in aux_history]
        plt.plot(t, b_orth, color=colors[i], label=f'k={i}')
    plt.title('Orthogonality Control Effort (Gamma)')
    plt.yscale('log')
    plt.legend()
    
    # 5. Phase Error
    plt.subplot(2, 3, 6)
    for i in range(k):
        e_phi = [aux[i]['e_phi'] for aux in aux_history]
        plt.plot(t, e_phi, color=colors[i])
    plt.title('Phase Locking Errors')
    
    plt.tight_layout()
    plt.show()

def main(n, k, delta, lr, lambda_x, lambda_y, lambda_phi, lambda_orth, n_steps):
    # Setup
    L = generate_graph_laplacian(n, delta)

    # Ground Truth
    vals, U, V = compute_eigen_decomposition(L, k)
    print("Target Eigenvalues (sorted by Real part):")
    print(vals)

    # Init
    XR, XI, YR, YI = generate_initial_conditions(n, k)

    # Sim
    t, X_T, Y_T, aux_history = simulate_dynamics(
        L, XR, XI, YR, YI, lr, lambda_x, lambda_y, lambda_phi, lambda_orth, n_steps
    )
    
    visualize_metrics(t, X_T, Y_T, U, V, vals, aux_history)

if __name__ == "__main__":
    n = 30
    k = 4
    delta = 0.1
    lr = 0.001
    lambda_x = 5.0
    lambda_y = 5.0
    lambda_phi = 10.0
    lambda_orth = 20.0
    n_steps = 100000
    
    main(n, k, delta, lr, lambda_x, lambda_y, lambda_phi, lambda_orth, n_steps)
import numpy as np
import scipy.linalg
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_graph_laplacian(n, delta):
    P = np.random.rand(n, n)**10
    P -= np.min(P)
    P /= np.sum(P, axis=1, keepdims=True)
    # rot = np.roll(np.eye(n), 1, axis=1) - np.eye(n)
    rot = np.zeros((n, n))
    L = np.eye(n)*(1+delta) - P + 0.5*rot
    return L

def compute_unique_targets(L, k_max):
    """
    Returns the first k_max UNIQUE eigenpairs (sorted by real part).
    If a complex conjugate pair exists, only the one with +imag part is kept.
    """
    vals, left, right = scipy.linalg.eig(L, left=True, right=True)
    
    # Sort by real part
    idx = np.argsort(vals.real)
    vals = vals[idx]
    right = right[:, idx]
    
    unique_vals = []
    unique_vecs = []
    
    seen_indices = set()
    
    for i in range(len(vals)):
        if i in seen_indices: continue
            
        v = vals[i]
        
        # Check for conjugate pair further down the list
        has_conjugate = False
        for j in range(i+1, len(vals)):
            if j in seen_indices: continue
            
            # If real parts match and imag parts are opposite
            if np.isclose(v.real, vals[j].real) and np.isclose(v.imag, -vals[j].imag) and abs(v.imag) > 1e-5:
                # Found a conjugate pair. 
                # We add the current one (i) to our targets.
                # We mark (j) as seen so we don't add it as a separate target.
                seen_indices.add(j)
                has_conjugate = True
                break
        
        unique_vals.append(v)
        unique_vecs.append(right[:, i])
        
        if len(unique_vals) == k_max:
            break
            
    return np.array(unique_vals), np.stack(unique_vecs, axis=1)

def init_dynamics(n, k):
    xr = np.random.randn(n, k)
    xi = np.random.randn(n, k)
    yr = np.random.randn(n, k)
    yi = np.random.randn(n, k)
    return xr, xi, yr, yi

# ... [Control/Dynamics functions remain identical to previous working version] ...
# (solve_min_norm, control_norm_exact, control_phase_exact, control_orth_double)
# Re-pasting them for a complete runnable script

def solve_min_norm(grad_r, grad_i, drift, epsilon=1e-8):
    grad_norm = np.sum(grad_r**2) + np.sum(grad_i**2)
    if grad_norm < epsilon or drift <= 0:
        return np.zeros_like(grad_r), np.zeros_like(grad_i), 0.0
    gamma = drift / grad_norm
    u_r = -gamma * grad_r
    u_i = -gamma * grad_i
    return u_r, u_i, gamma

def control_norm_exact(xr, xi, fx, fi, lambda_reg):
    n_sq = np.sum(xr**2) + np.sum(xi**2)
    e = n_sq - 1.0
    V = 0.5 * e**2
    gr = 2 * e * xr
    gi = 2 * e * xi
    drift = np.sum(gr * fx) + np.sum(gi * fi)
    total_drift = drift + lambda_reg * V
    ur, ui, gamma = solve_min_norm(gr, gi, total_drift)
    return ur, ui, e, gamma

def control_phase_exact(xr, xi, yr, yi, fxr, fxi, fyr, fyi, lambda_phi):
    e = np.sum(yr * xi) - np.sum(yi * xr)
    V = 0.5 * e**2
    g_xr = -e * yi
    g_xi = +e * yr
    g_yr = +e * xi
    g_yi = -e * xr
    Lf = (np.sum(g_xr * fxr) + np.sum(g_xi * fxi) + 
          np.sum(g_yr * fyr) + np.sum(g_yi * fyi))
    drift = Lf + lambda_phi * V
    norm_sq = np.sum(g_xr**2) + np.sum(g_xi**2) + np.sum(g_yr**2) + np.sum(g_yi**2)
    if norm_sq < 1e-8 or drift <= 0:
        return (np.zeros_like(xr), np.zeros_like(xr), 
                np.zeros_like(xr), np.zeros_like(xr), e, 0.0)
    gamma = drift / norm_sq
    return (-gamma*g_xr, -gamma*g_xi, -gamma*g_yr, -gamma*g_yi, e, gamma)

def control_orth_double(target_r, target_i, prev_r, prev_i, fr, fi, lambda_orth):
    n, k_prev = prev_r.shape
    if k_prev == 0:
        return np.zeros_like(target_r), np.zeros_like(target_i), 0.0, 0.0
    B = np.concatenate([prev_r, prev_i], axis=1)
    E_r = B.T @ target_r
    E_i = B.T @ target_i
    V = 0.5 * (np.sum(E_r**2) + np.sum(E_i**2))
    g_r = B @ E_r
    g_i = B @ E_i
    drift = np.sum(g_r * fr) + np.sum(g_i * fi)
    total_drift = drift + lambda_orth * V
    ur, ui, gamma = solve_min_norm(g_r, g_i, total_drift)
    return ur, ui, V, gamma

def dynamics(L, XR, XI, YR, YI, lambda_x, lambda_y, lambda_phi, lambda_orth):
    n, k = XR.shape
    FXR, FXI = np.zeros_like(XR), np.zeros_like(XI)
    FYR, FYI = np.zeros_like(YR), np.zeros_like(YI)
    aux = []
    
    for i in range(k):
        xr, xi = XR[:, i:i+1], XI[:, i:i+1]
        yr, yi = YR[:, i:i+1], YI[:, i:i+1]
        Lxr = L @ xr; Lxi = L @ xi
        LTyr = L.T @ yr; LTyi = L.T @ yi
        denom = np.sum(xr**2) + np.sum(xi**2) + 1e-8
        alpha = (np.sum(xr * Lxr) + np.sum(xi * Lxi)) / denom
        beta  = (np.sum(xr * Lxi) - np.sum(xi * Lxr)) / denom
        fx_bar = -Lxr + alpha * xr - beta * xi
        fi_bar = -Lxi + beta * xr + alpha * xi
        gy_bar = -LTyr + alpha * yr + beta * yi
        gi_bar = -LTyi - beta * yr + alpha * yi
        ux_r_n, ux_i_n, ex, bx = control_norm_exact(xr, xi, fx_bar, fi_bar, lambda_x)
        uy_r_n, uy_i_n, ey, by = control_norm_exact(yr, yi, gy_bar, gi_bar, lambda_y)
        if i > 0:
            pr_y, pi_y = YR[:, :i], YI[:, :i]
            pr_x, pi_x = XR[:, :i], XI[:, :i]
            ux_r_o, ux_i_o, Vx, box = control_orth_double(xr, xi, pr_y, pi_y, fx_bar, fi_bar, lambda_orth)
            uy_r_o, uy_i_o, Vy, boy = control_orth_double(yr, yi, pr_x, pi_x, gy_bar, gi_bar, lambda_orth)
        else:
            ux_r_o, ux_i_o, Vx, box = 0, 0, 0, 0
            uy_r_o, uy_i_o, Vy, boy = 0, 0, 0, 0
        ft_xr = fx_bar + ux_r_n + ux_r_o
        ft_xi = fi_bar + ux_i_n + ux_i_o
        ft_yr = gy_bar + uy_r_n + uy_r_o
        ft_yi = gi_bar + uy_i_n + uy_i_o
        ux_r_p, ux_i_p, uy_r_p, uy_i_p, ep, bp = control_phase_exact(
            xr, xi, yr, yi, ft_xr, ft_xi, ft_yr, ft_yi, lambda_phi
        )
        FXR[:, i:i+1] = ft_xr + ux_r_p
        FXI[:, i:i+1] = ft_xi + ux_i_p
        FYR[:, i:i+1] = ft_yr + uy_r_p
        FYI[:, i:i+1] = ft_yi + uy_i_p
        aux.append({'alpha': alpha, 'beta': beta, 'err_orth': Vx, 'err_phase': ep, 'b_norm': bx})
        
    return (FXR, FXI, FYR, FYI), aux

def step(L, XR, XI, YR, YI, lr, lx, ly, lp, lo):
    (fx, fi, fy, fyi), aux = dynamics(L, XR, XI, YR, YI, lx, ly, lp, lo)
    XR += lr * fx; XI += lr * fi
    YR += lr * fy; YI += lr * fyi
    return (XR, XI, YR, YI), aux

def simulate_dynamics(L, XR, XI, YR, YI, n_steps, lr, lx, ly, lp, lo):
    t_axis, traj_X, hist = [], [], []
    for i in tqdm(range(n_steps)):
        (XR, XI, YR, YI), aux = step(L, XR, XI, YR, YI, lr, lx, ly, lp, lo)
        if i % 10 == 0:
            t_axis.append(i)
            traj_X.append(XR + 1j*XI)
            step_data = {}
            for k in range(len(aux)):
                for key, val in aux[k].items():
                    if key not in step_data: step_data[key] = []
                    step_data[key].append(val)
            hist.append(step_data)
    return t_axis, np.stack(traj_X, axis=0), hist

def visualize_metrics(t, X_T, hist, unique_vals, unique_vecs):
    k_est = X_T.shape[2]
    colors = plt.cm.tab10(np.linspace(0, 1, k_est))
    
    plt.figure(figsize=(18, 12))
    
    # 1. Real Eigenvalues
    plt.subplot(2, 3, 1)
    for i in range(k_est):
        if i < len(unique_vals):
            plt.axhline(unique_vals[i].real, color=colors[i], linestyle='--', alpha=0.5)
        alpha = [h['alpha'][i] for h in hist]
        plt.plot(t, alpha, color=colors[i], label=f'Est {i}')
    plt.title('Eigenvalues (Real)')
    plt.legend()
    
    # 2. Imag Eigenvalues
    plt.subplot(2, 3, 2)
    for i in range(k_est):
        if i < len(unique_vals):
            # Plot both +beta and -beta as valid targets
            v_imag = unique_vals[i].imag
            plt.axhline(v_imag, color=colors[i], linestyle='--', alpha=0.3)
            if abs(v_imag) > 1e-5:
                plt.axhline(-v_imag, color=colors[i], linestyle=':', alpha=0.3)
        beta = [h['beta'][i] for h in hist]
        plt.plot(t, beta, color=colors[i])
    plt.title('Eigenvalues (Imaginary)')
    
    # 3. Corrected "Max" Cosine Similarity
    plt.subplot(2, 3, 3)
    
    for i in range(k_est):
        if i < unique_vecs.shape[1]:
            # Target is the i-th unique vector
            v_target = unique_vecs[:, i]
            
            # Pre-calculate norm of target
            norm_target = np.linalg.norm(v_target)
            
            sim_curve = []
            
            for step_idx in range(X_T.shape[0]):
                x_est = X_T[step_idx, :, i]
                norm_est = np.linalg.norm(x_est)
                
                # Check similarity against v
                dot_v = np.vdot(x_est, v_target)
                sim_v = np.abs(dot_v) / (norm_est * norm_target)
                
                # Check similarity against conj(v)
                dot_conj = np.vdot(x_est, v_target.conj())
                sim_conj = np.abs(dot_conj) / (norm_est * norm_target)
                
                # Take the MAX
                sim_curve.append(max(sim_v, sim_conj))
                
            plt.plot(t, sim_curve, color=colors[i], label=f'Pair {i}')
            
    plt.title('Similarity to Target (Max of v / conj(v))')
    plt.ylim([0, 1.05])
    plt.legend()

    # 4. Orthogonality Error
    plt.subplot(2, 3, 4)
    for i in range(1, k_est):
        orth = [h['err_orth'][i] for h in hist]
        plt.plot(t, orth, color=colors[i], label=f'Pair {i}')
    plt.title('Double Deflation Error')
    plt.yscale('log')
    plt.legend()

    # 5. Phase Error
    plt.subplot(2, 3, 5)
    for i in range(k_est):
        phi = [h['err_phase'][i] for h in hist]
        plt.plot(t, np.abs(phi), color=colors[i])
    plt.title('Phase Locking Error')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def main(n, k, n_steps):
    delta = 0.1
    lr = 0.001
    lx, ly = 5.0, 5.0
    lp = 10.0
    lo = 20.0 
    
    L = generate_graph_laplacian(n, delta)
    
    # Calculate unique targets
    unique_vals, unique_vecs = compute_unique_targets(L, k_max=k)
    print("Unique Target Eigenvalues:")
    print(unique_vals)
    
    XR, XI, YR, YI = init_dynamics(n, k)
    
    t, X_T, hist = simulate_dynamics(L, XR, XI, YR, YI, n_steps, lr, lx, ly, lp, lo)
    
    visualize_metrics(t, X_T, hist, unique_vals, unique_vecs)

if __name__ == "__main__":
    main(n=30, k=4, n_steps=200000)
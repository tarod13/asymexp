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
    print("Eigenvalues of the graph Laplacian L:")
    print(eigvals)
    print("Norm of the right dominant eigenvector v:", np.linalg.norm(v))
    print("Norm of the left dominant eigenvector u:", np.linalg.norm(u))
    print("Dominant eigenvalue:", eigval)
    print("Left eigenvector u^T * right eigenvector v:", u.T @ v)
    return eigval, u, v

def compute_jacobian(L, x, y, beta, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    n = L.shape[0]
    I = np.eye(n)
    LT = L.T
    xxT = np.outer(x, x)
    yyT = np.outer(y, y)
    xyT = np.outer(x, y)
    y2 = y.T @ y
    Ly = L @ y
    LTx = LT @ x
    yTLx = y.T @ (L @ x)
    s = np.sign(yTLx)[0,0]

    adversarial_sign = -1 if adversarial else 1

    dfx_dx = lr_x * b * (-2*xxT - yyT)
    dfx_dy = lr_x * (-s*LT + beta*I - b*xyT.T)
    dfx_dbx = lr_x * y
    dfx_dby = np.zeros((n,1))
    dfx_da = -lr_x * x

    dfy_dx = lr_y * (adversarial_sign * (-s*L + beta*I) - b*xyT)
    dfy_dy = -lr_y * b * xxT
    dfy_dbx = np.zeros((n,1))
    dfy_dby = adversarial_sign * lr_y * x
    dfy_da = np.zeros((n,1))

    dfbx_dx = np.zeros((1, n))
    dfbx_dy = -lr_bx * (-Ly.T + beta*y.T)
    dfbx_bx = -lr_bx * y2
    dfbx_dby = np.zeros((1,1))
    dfbx_da = np.zeros((1,1))

    dfby_dx = -lr_by * (-LTx.T + beta*x.T)
    dfby_dy = np.zeros((1, n))
    dfby_dbx = np.zeros((1,1))
    dfby_dby = -lr_by * np.ones((1,1))
    dfby_da = np.zeros((1,1))

    dfa_dx = lr_a * 2 * x.T
    dfa_dy = np.zeros((1, n))
    dfa_dbx = np.zeros((1,1))
    dfa_dby = np.zeros((1,1))
    dfa_da = np.zeros((1,1))

    J_x = np.concatenate([dfx_dx, dfx_dy, dfx_dbx, dfx_dby, dfx_da], axis=1)
    J_y = np.concatenate([dfy_dx, dfy_dy, dfy_dbx, dfy_dby, dfy_da], axis=1)
    J_bx = np.concatenate([dfbx_dx, dfbx_dy, dfbx_bx, dfbx_dby, dfbx_da], axis=1)
    J_by = np.concatenate([dfby_dx, dfby_dy, dfby_dbx, dfby_dby, dfby_da], axis=1)
    J_a = np.concatenate([dfa_dx, dfa_dy, dfa_dbx, dfa_dby, dfa_da], axis=1)
    J = np.concatenate([J_x, J_y, J_bx, J_by, J_a], axis=0)

    return J

def analyze_stability(J):
    jac_eigvals = np.linalg.eigvals(J)

    # Order eigenvalues by real component
    jac_eigvals = jac_eigvals[np.argsort(jac_eigvals.real)[::-1]]
    
    print("Eigenvalues of the Jacobian at the dominant eigenvector:")
    print(jac_eigvals)
    if np.any(jac_eigvals.real > 0):
        print("The system is locally unstable around the dominant eigenvector.")
    else:
        print("The system is locally stable around the dominant eigenvector.")

def generate_initial_conditions(n):
    x0 = np.random.randn(n, 1)
    y0 = np.random.randn(n, 1)
    b_x0 = 1.0 * np.ones((1, 1))
    b_y0 = 1.0 * np.ones((1, 1))
    a0 = 1.0 * np.ones((1, 1))
    return x0, y0, b_x0, b_y0, a0

def dynamics(L, x, y, beta_x, beta_y, a, b, adversarial=False):
    adversarial_sign = -1 if adversarial else 1

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    yTx = np.einsum('ij,ij->j', y, x).reshape(1,1)
    x2 = np.einsum('ij,ij->j', x, x).reshape(1,1)
    Lx = np.einsum('ij,jk->ik', L, x)
    LTy = np.einsum('ji,jk->ik', L, y)
    
    e_x = x2 - 1
    e_xy = yTx - 1
    yTLx = np.einsum('ij,ij->j', y, Lx).reshape(1,1)
    s = np.sign(yTLx)

    fx = (-s*LTy + beta_x[0,0] * y) - (a[0,0] * x) - (b * e_x[0,0] * x) - (b * e_xy[0,0] * y)
    fy = adversarial_sign * (-s*Lx + beta_y[0,0] * x) - (b * e_xy[0,0] * x)
    f_bx = -np.einsum('ij,ij->j', (-s*LTy + beta_x[0,0] * y), y).reshape(1,1)
    f_by = -np.einsum('ij,ij->j', (-s*Lx + beta_y[0,0] * x), x).reshape(1,1)
    f_a = e_x
    
    return fx, fy, f_bx, f_by, f_a

def step(L, x, y, beta_x, beta_y, a, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial):
    
    # Compute gradients
    fx, fy, f_bx, f_by, f_a = dynamics(
        L, x, y, beta_x, beta_y, a, b, adversarial=adversarial
    )

    # Update variables
    x_new = x + lr_x * fx
    y_new = y + lr_y * fy
    beta_x_new = beta_x + lr_bx * f_bx
    beta_y_new = beta_y + lr_by * f_by
    a_new = a + lr_a * f_a

    # Calculate gradient norms for monitoring
    grad_norms = {
        '||grad_x||': np.linalg.norm(fx),
        '||grad_y||': np.linalg.norm(fy),
        '||grad_beta_x||': np.linalg.norm(f_bx),
        '||grad_beta_y||': np.linalg.norm(f_by),
        '||grad_a||': np.linalg.norm(f_a)
    }
    return x_new, y_new, beta_x_new, beta_y_new, a_new, grad_norms

def objective_functions(L, x, y, beta_x, beta_y, a, b, adversarial=False):
    adversarial_sign = -1 if adversarial else 1

    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    yTx = np.einsum('ij,ij->j', y, x).reshape(1,1)
    x2 = np.einsum('ij,ij->j', x, x).reshape(1,1)
    Lx = np.einsum('ij,jk->ik', L, x)
    LTy = np.einsum('ji,jk->ik', L, y)
    
    e_x = x2 - 1
    e_xy = yTx - 1
    graph_loss = np.einsum('ij,ij->j', y, Lx).reshape(1,1)

    L_x = graph_loss - beta_x * e_xy + (a/2) * e_x + (b/2) * (e_x**2 + e_xy**2)
    L_y = adversarial_sign * (graph_loss - beta_y * e_xy) + (b/2) * e_xy**2
    L_b_x = ((-LTy + beta_x * y)**2).sum(keepdims=True) / 2
    L_b_y = ((-Lx + beta_y * x)**2).sum(keepdims=True) / 2
    L_a = (e_x**2) / 2

    return L_x, L_y, L_b_x, L_b_y, L_a

def simulate_dynamics(L, x0, y0, beta_x0, beta_y0, a0, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial, n_steps):
    # Initialize variables
    x = x0
    y = y0
    beta_x = beta_x0
    beta_y = beta_y0
    a = a0

    # Record trajectory
    trajectory = []
    trajectory.append(
        (x.copy(), y.copy(), beta_x.copy(), beta_y.copy(), a.copy())
    )
    objectives = []
    grad_norms_list = []

    for _ in tqdm(range(n_steps), desc="Simulating dynamics", leave=True):
        x, y, beta_x, beta_y, a, grad_norms = step(
            L, x, y, beta_x, beta_y, a, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial
        )
        trajectory.append(
            (x.copy(), y.copy(), beta_x.copy(), beta_y.copy(), a.copy())
        )
        grad_norms_list.append(grad_norms)

        L_x, L_y, L_bx, L_by, L_a = objective_functions(
            L, x, y, beta_x, beta_y, a, b, adversarial
        )
        objectives.append((L_x, L_y, L_bx, L_by, L_a))
    
    # Separate recorded trajectory into arrays
    t = np.arange(n_steps+1)
    x_t = np.concatenate([state[0] for state in trajectory], axis=1).T
    y_t = np.concatenate([state[1] for state in trajectory], axis=1).T
    beta_x_t = np.concatenate([state[2] for state in trajectory], axis=1).T
    beta_y_t = np.concatenate([state[3] for state in trajectory], axis=1).T
    a_t = np.concatenate([state[4] for state in trajectory], axis=1).T

    print("Shapes of the recorded trajectory:")
    print(x_t.shape, y_t.shape, beta_x_t.shape, beta_y_t.shape, a_t.shape)

    # Convert gradient norms to array
    grad_norm_arrays = {key: np.array([gn[key] for gn in grad_norms_list]) for key in grad_norms_list[0]
    }

    # Convert objectives to array
    L_x_t = np.concatenate([obj[0] for obj in objectives], axis=1).T
    L_y_t = np.concatenate([obj[1] for obj in objectives], axis=1).T
    L_bx_t = np.concatenate([obj[2] for obj in objectives], axis=1).T
    L_by_t = np.concatenate([obj[3] for obj in objectives], axis=1).T
    L_a_t = np.concatenate([obj[4] for obj in objectives], axis=1).T

    return t, x_t, y_t, beta_x_t, beta_y_t, a_t, L_x_t, L_y_t, L_bx_t, L_by_t, L_a_t, grad_norm_arrays

def compute_metrics(x_t, y_t, u, v):
    ny_t = np.linalg.norm(y_t, axis=1)
    e_x_t = np.linalg.norm(x_t, axis=1)**2 - 1
    e_xy_t = np.einsum('ij,ij->i', y_t, x_t) - 1
    cs_v = np.abs(np.einsum('ij,j->i', x_t, v)) / (np.linalg.norm(x_t, axis=1) * np.linalg.norm(v))
    cs_u = np.abs(np.einsum('ij,j->i', y_t, u)) / (np.linalg.norm(y_t, axis=1) * np.linalg.norm(u))
    return ny_t, e_x_t, e_xy_t, cs_v, cs_u

def visualize_metrics(
        t, ny_t, e_x_t, e_xy_t, cs_v, cs_u, beta_x_t, beta_y_t, eigval, a_t,
        L_x_t, L_y_t, L_bx_t, L_by_t, L_a_t, grad_norm_arrays
    ):
    

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 4, 1)
    plt.plot(t, np.abs(e_x_t), label=r'$e_x$')
    plt.plot(t, np.abs(e_xy_t), label=r'$e_{xy}$')
    plt.plot(t, ny_t, label=r'$||y||_2$')
    plt.title('Errors and y Norm over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Error Magnitude')
    plt.legend()
    plt.yscale('log')

    plt.subplot(2, 4, 2)
    plt.plot(t, cs_v, label=r'$<v,x>$')
    plt.plot(t, cs_u, label=r'$<u,y>$')
    plt.title('Cosine Similarities over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Cosine Similarity')
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.plot(t, np.abs(beta_x_t), label=r'$\beta_x$')
    plt.plot(t, np.abs(beta_y_t), label=r'$\beta_y$')
    plt.plot(t, eigval * np.ones_like(t), 'k--', label='Eigenvalue')
    plt.title('Dual Terms over Time')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    plt.subplot(2, 4, 4)
    plt.plot(t, np.abs(a_t), label=r'$a$')
    plt.title('Parameter a over Time')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.yscale('log')

    plt.subplot(2, 4, 5)
    plt.plot(t[1:], np.abs(L_x_t), label=r'$x$')
    plt.plot(t[1:], np.abs(L_y_t), label=r'$y$')
    plt.title('Objective Functions')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.yscale('log')

    plt.subplot(2, 4, 6)
    plt.plot(t[1:], np.abs(L_bx_t), label=r'$\beta_x$')
    plt.plot(t[1:], np.abs(L_by_t), label=r'$\beta_y$')
    plt.plot(t[1:], np.abs(L_a_t), label=r'$a$')
    plt.title('Objective Functions')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.yscale('log')

    plt.subplot(2, 4, 7)
    for key, values in grad_norm_arrays.items():
        if key in ['||grad_x||', '||grad_y||']:
            plt.plot(t[1:], values, label=key)
    plt.title('Gradient Norms')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Norm')
    plt.yscale('log')

    plt.subplot(2, 4, 8)
    for key, values in grad_norm_arrays.items():
        if key in ['||grad_beta_x||', '||grad_beta_y||', '||grad_a||']:
            plt.plot(t[1:], values, label=key)
    plt.title('Gradient Norms')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Norm')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

def main(n, delta, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial, n_steps):
    # Create scaled graph Laplacian L
    L = generate_graph_laplacian(n, delta)

    # Compute eigenvalues and eigenvectors
    eigval, u, v = compute_eigen_decomposition(L)

    # Compute Jacobian at the dominant eigenvector
    J = compute_jacobian(L, v, u, eigval, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial)
    
    # Analyze stability
    analyze_stability(J)

    # Generate initial conditions
    x0, y0, b_x0, b_y0, a0 = generate_initial_conditions(n)

    # Simulate dynamics
    t, x_t, y_t, b_x_t, b_y_t, a_t, L_x_t, L_y_t, L_bx_t, L_by_t, L_a_t, grad_norm_arrays = simulate_dynamics(
        L, x0, y0, b_x0, b_y0, a0, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial, n_steps)
    
    # Compute metrics
    ny_t, e_x_t, e_xy_t, cs_v, cs_u = compute_metrics(x_t, y_t, u, v)

    # Visualize metrics
    visualize_metrics(
        t, ny_t, e_x_t, e_xy_t, cs_v, cs_u, b_x_t, b_y_t, eigval, a_t, 
        L_x_t, L_y_t, L_bx_t, L_by_t, L_a_t, grad_norm_arrays
    )


if __name__ == "__main__":
    n = 50
    delta = 5.0
    lr_x = 0.001
    lr_y = 0.0005
    lr_bx = 0.001
    lr_by = 0.001
    lr_a = 0.001
    b = 5
    adversarial = True
    n_steps = 20000
    main(n, delta, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial, n_steps)
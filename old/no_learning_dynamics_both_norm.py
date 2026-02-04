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
    return None

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
    return x0, y0

def control_x_norm(x, L, lambda_x):
    Lx = np.einsum('ij,jk->ik', L, x)
    xTLx = np.einsum('ij,ij->j', x, Lx).reshape(1,1)
    xTx = np.einsum('ij,ij->j', x, x).reshape(1,1)
    b_x = max(0, (xTLx + lambda_x/4) / (xTx + 1e-8))
    return b_x

def control_y_norm(y, L, lambda_y):
    Ly = np.einsum('ij,jk->ik', L, y)
    yTLy = np.einsum('ij,ij->j', y, Ly).reshape(1,1)
    yTy = np.einsum('ij,ij->j', y, y).reshape(1,1)
    b_y = max(0, (yTLy + lambda_y/4) / (yTy + 1e-8))
    return b_y

def control_xy_correlation(x, y, L, lambda_xy):
    Lx = np.einsum('ij,jk->ik', L, x)
    LTy = np.einsum('ji,jk->ik', L, y)

    xTLx = np.einsum('ij,ij->j', x, Lx).reshape(1,1)
    yTLy = np.einsum('ij,ij->j', y, LTy).reshape(1,1)

    yTLx = np.einsum('ij,ij->j', y, Lx).reshape(1,1)
    yTx = np.einsum('ij,ij->j', y, x).reshape(1,1)

    yTy = np.einsum('ij,ij->j', y, y).reshape(1,1)
    xTx = np.einsum('ij,ij->j', x, x).reshape(1,1)
    
    e_xy = yTx - 1
    V_xy = e_xy**2/2

    grad_f_bar_x_dot_grad_V_x = e_xy * (xTLx*yTx - yTLx)
    grad_f_bar_y_dot_grad_V_y = e_xy * (yTLy*yTx - yTLx)
    grad_f_bar_dot_grad_V = grad_f_bar_x_dot_grad_V_x + grad_f_bar_y_dot_grad_V_y

    alignment_term = grad_f_bar_dot_grad_V + lambda_xy * V_xy
    
    grad_V_x_squared = (yTy * e_xy**2)
    grad_V_y_squared = (xTx * e_xy**2)
    grad_V_squared = grad_V_x_squared + grad_V_y_squared

    b_xy = max(0, alignment_term) / (grad_V_squared + 1e-6)

    return b_xy

def dynamics(L, x, y, lambda_x, lambda_y):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    x2 = np.einsum('ij,ij->j', x, x).reshape(1,1)
    y2 = np.einsum('ij,ij->j', y, y).reshape(1,1)
    Lx = np.einsum('ij,jk->ik', L, x)
    LTy = np.einsum('ji,jk->ik', L, y)
    
    e_x = x2 - 1
    e_y = y2 - 1

    lambda_Lx = np.einsum('ij,ij->j', x, Lx).reshape(1,1)
    lambda_Ly = np.einsum('ij,ij->j', y, LTy).reshape(1,1)
    graph_loss = np.einsum('ij,ij->j', y, Lx).reshape(1,1)

    f_bar_x = -Lx + lambda_Lx * x
    f_bar_y = -LTy + lambda_Ly * y

    b_x = control_x_norm(x, L, lambda_x)
    b_y = control_y_norm(y, L, lambda_y)
    
    u_x = -b_x * e_x * x
    u_y = -b_y * e_y * y

    fx = f_bar_x + u_x
    fy = f_bar_y + u_y

    L_x = graph_loss - lambda_Lx*x2/2 + b_x*e_x**2/2
    L_y = graph_loss - lambda_Ly*y2/2 + b_y*e_y**2/2

    aux = {
        'f_bar_x': f_bar_x,
        'f_bar_y': f_bar_y,
        'b_x': b_x,
        'b_y': b_y,  
        'u_x': u_x,
        'u_y': u_y,
        'e_x': e_x,
        'e_y': e_y,
        'lambda_Lx': lambda_Lx,
        'lambda_Ly': lambda_Ly,
        'graph_loss': graph_loss,
        'x2': x2,
        'y2': y2,
        'L_x': L_x,
        'L_y': L_y,
    }
    
    return (fx, fy), aux

def step(L, x, y, lr, lambda_x, lambda_xy):
    
    # Compute gradients
    (fx, fy), aux = dynamics(
        L, x, y, lambda_x, lambda_xy
    )

    # Normalize gradients if needed
    grad_norm_x = np.linalg.norm(fx)
    grad_norm_y = np.linalg.norm(fy)
    if grad_norm_x > 1:
        fx = fx / grad_norm_x
    if grad_norm_y > 1:
        fy = fy / grad_norm_y

    # Update variables
    x_new = x + lr * fx
    y_new = y + lr * fy

    # Calculate gradient norms for monitoring
    grad_norms = {
        '$||f_x||$': np.linalg.norm(fx),
        '$||f_y||$': np.linalg.norm(fy),
        '$||\\bar{f}_x||$': np.linalg.norm(aux['f_bar_x']),
        '$||\\bar{f}_y||$': np.linalg.norm(aux['f_bar_y']),
        '$||u_x||$': np.linalg.norm(aux['u_x']),
        '$||u_y||$': np.linalg.norm(aux['u_y']),
    }

    return (x_new, y_new), aux, grad_norms

def simulate_dynamics(L, x0, y0, lr, lambda_x, lambda_y, n_steps):
    # Initialize variables
    x = x0
    y = y0
    
    # Record trajectory
    trajectory = []
    trajectory.append(
        (x.copy(), y.copy())
    )
    aux_variables = []
    grad_norms_list = []

    for _ in tqdm(range(n_steps), desc="Simulating dynamics", leave=True):
        (x, y), aux, grad_norms = step(
            L, x, y, lr, lambda_x, lambda_y
        )
        trajectory.append(
            (x.copy(), y.copy())
        )
        aux_variables.append(aux)
        grad_norms_list.append(grad_norms)
    
    # Separate recorded trajectory into arrays
    t = np.arange(n_steps+1)
    x_t = np.concatenate([state[0] for state in trajectory], axis=1).T
    y_t = np.concatenate([state[1] for state in trajectory], axis=1).T

    print("Shapes of the recorded trajectory:")
    print(x_t.shape, y_t.shape)

    # Convert auxiliary variables to arrays, handling different shapes
    aux_arrays = {}
    for key in aux_variables[0]:
        values = [aux[key] for aux in aux_variables]
        # Check if values are scalars (0D) or have shape (1,1)
        if values[0].ndim == 0 or (values[0].ndim == 2 and values[0].shape == (1, 1)):
            # Flatten scalars to 1D array
            aux_arrays[key] = np.array([v.item() if hasattr(v, 'item') else v for v in values])
        else:
            # For 2D arrays (vectors), concatenate along axis 1 and transpose
            aux_arrays[key] = np.concatenate(values, axis=1).T

    # Convert gradient norms to array
    grad_norm_arrays = {key: np.array([gn[key] for gn in grad_norms_list]) for key in grad_norms_list[0]
    }

    return t, x_t, y_t, aux_arrays, grad_norm_arrays

def compute_metrics(x_t, y_t, u, v):
    cs_v = np.abs(np.einsum('ij,j->i', x_t, v)) / (np.linalg.norm(x_t, axis=1) * np.linalg.norm(v))
    cs_u = np.abs(np.einsum('ij,j->i', y_t, u)) / (np.linalg.norm(y_t, axis=1) * np.linalg.norm(u))
    return cs_v, cs_u

def visualize_metrics(t, cs_v, cs_u, eigval, aux_arrays, grad_norm_arrays):    

    plt.figure(figsize=(16, 8))

    t1 = t[1:]

    plt.subplot(2, 4, 1)
    e_x_t = aux_arrays['e_x']
    e_y_t = aux_arrays['e_y']
    plt.plot(t1, np.abs(e_x_t), label=r'$e_x$')
    plt.plot(t1, np.abs(e_y_t), label=r'$e_{y}$')
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
    beta_x_t = aux_arrays['lambda_Lx']
    beta_y_t = aux_arrays['lambda_Ly']
    plt.plot(t1, np.abs(beta_x_t), label=r'$\lambda_{x}$')
    plt.plot(t1, np.abs(beta_y_t), label=r'$\lambda_{y}$')
    plt.plot(t1, eigval * np.ones_like(t1), 'k--', label='Eigenvalue')
    plt.title('Effective Dual Terms over Time')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    plt.subplot(2, 4, 4)
    b_x_t = aux_arrays['b_x']
    b_y_t = aux_arrays['b_y']
    plt.plot(t1, b_x_t, label=r'$b_x$')
    plt.plot(t1, b_y_t, label=r'$b_{y}$')
    plt.title('Barrier Terms over Time')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    plt.subplot(2, 4, 5)
    L_x_t = aux_arrays['L_x']
    L_y_t = aux_arrays['L_y']
    plt.plot(t1, L_x_t, label=r'$x$')
    plt.plot(t1, L_y_t, label=r'$y$')
    plt.title('Objective Functions')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    plt.subplot(2, 4, 6)
    graph_loss_t = aux_arrays['graph_loss']
    plt.plot(t1, graph_loss_t, label=r'Graph Loss')
    plt.title('Objective Functions')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Value')

    plt.subplot(2, 4, 7)
    for key, values in grad_norm_arrays.items():
        if key in ['$||f_x||$', '$||f_y||$']:
            plt.plot(t1, values, label=key)
    plt.title('Gradient Norms')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Norm')
    plt.yscale('log')

    plt.subplot(2, 4, 8)
    for key, values in grad_norm_arrays.items():
        if key in ['$||\\bar{f}_x||$', '$||\\bar{f}_y||$', '$||u_x||$', '$||u_y||$']:
            plt.plot(t1, values, label=key)
    plt.title('Gradient Norms')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Norm')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

def main(n, delta, lr, lambda_x, lambda_y, n_steps):
    # Create scaled graph Laplacian L
    L = generate_graph_laplacian(n, delta)

    # Compute eigenvalues and eigenvectors
    eigval, u, v = compute_eigen_decomposition(L)

    # # Compute Jacobian at the dominant eigenvector
    # J = compute_jacobian(L, v, u, eigval, lr_x, lr_y, lr_bx, lr_by, lr_a, b, adversarial)
    
    # # Analyze stability
    # analyze_stability(J)

    # Generate initial conditions
    x0, y0 = generate_initial_conditions(n)

    # Simulate dynamics
    t, x_t, y_t, aux_arrays, grad_norm_arrays  = simulate_dynamics(
        L, x0, y0, lr, lambda_x, lambda_y, n_steps)
    
    # Compute metrics
    cs_v, cs_u = compute_metrics(x_t, y_t, u, v)

    # Visualize metrics
    visualize_metrics(
        t, cs_v, cs_u, eigval, aux_arrays, grad_norm_arrays
    )


if __name__ == "__main__":
    n = 50
    delta = 0.1
    lr = 0.001
    lambda_x = 10
    lambda_y = 10
    n_steps = 20000
    main(n, delta, lr, lambda_x, lambda_y, n_steps)
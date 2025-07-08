import numpy as np

def adam(grad, x, callback=None, num_iters=200, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8, tol=1e-6):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""

    # *** CRITICAL CHANGE HERE ***
    m = np.zeros_like(x, dtype=x.dtype) # Initialize m with the exact shape and dtype of x
    v = np.zeros_like(x, dtype=x.dtype) # Initialize v with the exact shape and dtype of x

    for i in range(num_iters):
        g = grad(x, i)

        # *** Add a robust shape check ***
        if g.shape != x.shape:
            raise ValueError(f"Gradient shape mismatch! Expected {x.shape}, but got {g.shape}. "
                             "Ensure your grad function returns a gradient with the same shape as x.")

        if callback:
            callback(x, i, g)

        grad_norm = np.linalg.norm(g) # For 2D arrays, this computes the Frobenius norm
        if grad_norm < tol:
            print(f"Early stopping at iteration {i}, gradient norm {grad_norm:.2e} < tol {tol}")
            break

        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)

        if i % 10 == 0:
            print(f"Iteration {i}, step size: {step_size:.2e}")
    return x
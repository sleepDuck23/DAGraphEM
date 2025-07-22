import numpy as np

def adam(grad, x, callback=None, num_iters=200, step_size=0.001, clip=1.0, clip_flag = False, b1=0.9, b2=0.999, eps=1e-6, tol=1e-9):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""

    # *** CRITICAL CHANGE HERE ***
    m = np.zeros_like(x, dtype=np.float32) # Initialize m with the exact shape and dtype of x
    v = np.zeros_like(x, dtype=np.float32) # Initialize v with the exact shape and dtype of x

    for i in range(num_iters):
        g = grad(x)

        # shape check 
        if g.shape != x.shape:
            raise ValueError(f"Gradient shape mismatch! Expected {x.shape}, but got {g.shape}. "
                             "Ensure your grad function returns a gradient with the same shape as x.")

        if callback:
            callback(x, g)

        grad_norm = np.linalg.norm(g) 

        
        if grad_norm > clip and clip_flag:
            g = np.clip(g, -clip, clip)
            grad_norm = np.linalg.norm(g) 



        #if grad_norm < tol:
        #    print(f"Early stopping at iteration {i}, gradient norm {grad_norm:.2e} < tol {tol}")
        #    break

        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)

        #if i % 100 == 0:
        #    print(f"Iteration {i}, gradient norm {grad_norm:.2e}, step size: {step_size:.2e}")
        #    print(f"matrix A: {x}")

        if grad_norm < 1e-3 and step_size < 1e-1:
            step_size *= 1.1
        elif grad_norm > 1e4 and step_size > 1e-6:
            step_size *= 0.9
    return x, grad_norm

def adam_alpha(grad, x, alpha, callback=None, num_iters=200, step_size=0.001, clip=1.0, clip_flag = False, b1=0.9, b2=0.999, eps=1e-6, tol=1e-9):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""

    # *** CRITICAL CHANGE HERE ***
    m = np.zeros_like(x, dtype=np.float32) # Initialize m with the exact shape and dtype of x
    v = np.zeros_like(x, dtype=np.float32) # Initialize v with the exact shape and dtype of x

    for i in range(num_iters):
        g = grad(x,alpha)

        # shape check 
        if g.shape != x.shape:
            raise ValueError(f"Gradient shape mismatch! Expected {x.shape}, but got {g.shape}. "
                             "Ensure your grad function returns a gradient with the same shape as x.")

        if callback:
            callback(x, g)


        grad_norm = np.linalg.norm(g) 

        if grad_norm < 1e2:
            alpha = min(alpha * 5, 1e12)

        
        if grad_norm > clip and clip_flag:
            g = np.clip(g, -clip, clip)
            grad_norm = np.linalg.norm(g) 



        #if grad_norm < tol:
        #    print(f"Early stopping at iteration {i}, gradient norm {grad_norm:.2e} < tol {tol}")
        #    break

        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)

        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}, gradient norm {grad_norm:.2e}, step size: {step_size:.2e}")
            print(f"matrix A: {x}")
            print(f"alpha: {alpha}")

        if grad_norm < 1e-3 and step_size < 1e-1:
            step_size *= 1.1
        elif grad_norm > 1e4 and step_size > 1e-10:
            step_size *= 0.9
    return x, grad_norm
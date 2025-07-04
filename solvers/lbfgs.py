import numpy as np

def lbfgs_numpy(f, x0, grad_f, m=10, max_iter=100, tol=1e-5):
    x = x0.copy()
    s_list, y_list, rho_list = [], [], []

    def compute_grad(x):
        return grad_f(x)

    def line_search_wolfe(f, xk, pk, gk, grad_f=None, alpha0=1.0, c1=1e-4, c2=0.9, max_iter=20):
        def phi(alpha):
            return f(xk + alpha * pk)

        def phi_grad(alpha):
            x_new = xk + alpha * pk
            return np.dot(grad_f(x_new), pk)

        alpha = alpha0
        phi0 = phi(0.0)
        phi_grad0 = np.dot(gk, pk)

        for i in range(max_iter):
            phi_alpha = phi(alpha)
            if phi_alpha > phi0 + c1 * alpha * phi_grad0 or (i > 0 and phi_alpha >= phi(alpha / 2)):
                return zoom(phi, phi_grad, 0, alpha, phi0, phi_grad0, c1, c2)
            phi_grad_alpha = phi_grad(alpha)
            if abs(phi_grad_alpha) <= -c2 * phi_grad0:
                return alpha
            if phi_grad_alpha >= 0:
                return zoom(phi, phi_grad, alpha, 0, phi0, phi_grad0, c1, c2)
            alpha *= 2.0
        return alpha

    def zoom(phi, phi_grad, alo, ahi, phi0, phi_grad0, c1, c2, max_zoom_iter=10):
        for _ in range(max_zoom_iter):
            alpha = 0.5 * (alo + ahi)
            phi_alpha = phi(alpha)
            if phi_alpha > phi0 + c1 * alpha * phi_grad0 or phi_alpha >= phi(alo):
                ahi = alpha
            else:
                phi_grad_alpha = phi_grad(alpha)
                if abs(phi_grad_alpha) <= -c2 * phi_grad0:
                    return alpha
                if phi_grad_alpha * (ahi - alo) >= 0:
                    ahi = alo
                alo = alpha
        return alpha

    g = compute_grad(x)
    k = 0

    while np.linalg.norm(g) > tol and k < max_iter:
        q = g.copy()
        alpha_list = []

        for i in range(len(s_list) - 1, -1, -1):
            s, y, rho = s_list[i], y_list[i], rho_list[i]
            alpha = rho * np.dot(s, q)
            q = q - alpha * y
            alpha_list.append(alpha)

        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
            r = gamma * q
        else:
            r = q

        for i in range(len(s_list)):
            s, y, rho = s_list[i], y_list[i], rho_list[i]
            beta = rho * np.dot(y, r)
            r = r + s * (alpha_list[len(s_list) - 1 - i] - beta)

        p = -r
        alpha = line_search_wolfe(f, x, p, g, grad_f)
        s = alpha * p
        x_new = x + s
        g_new = compute_grad(x_new)
        y = g_new - g

        if np.dot(y, s) > 1e-10:
            if len(s_list) == m:
                s_list.pop(0), y_list.pop(0), rho_list.pop(0)
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / np.dot(y, s))

        x = x_new
        g = g_new
        k += 1

    return x, f(x), k


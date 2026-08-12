import numpy as np

def rnn_step_backward(dh, cache):
    x_t, h_prev, h_t, W, U, b = cache

    # Convert lists to NumPy arrays
    dh = np.asarray(dh)
    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    h_t = np.asarray(h_t)
    W = np.asarray(W)
    U = np.asarray(U)
    b = np.asarray(b)

    # Backprop through tanh
    dz = dh * (1 - h_t ** 2)

    # Gradients
    dx_t = W.T @ dz
    dh_prev = U.T @ dz
    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz

    return dx_t, dh_prev, dW, dU, db
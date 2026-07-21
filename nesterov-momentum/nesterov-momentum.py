import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    w=np.asarray(w,float)
    v=np.asarray(v,float)
    grad=np.asarray(grad,float)
    v=momentum*v+lr*grad
    w=w-v
    return w,v
import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    w=np.asarray(w,float)
    grad=np.asarray(grad,float)
    E_grad_sq=np.asarray(E_grad_sq,float)
    E_update_sq=np.asarray(E_update_sq,float)
    E_grad_sq=rho*E_grad_sq+(1-rho)*grad**2
    delta=-np.sqrt(E_update_sq+eps)/np.sqrt(E_grad_sq+eps)*grad
    E_update_sq=rho*E_update_sq+(1-rho)*delta**2
    w=w+delta
    return w,E_grad_sq,E_update_sq
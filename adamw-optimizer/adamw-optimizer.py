import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    w=np.asarray(w,float);m=np.asarray(m,float)
    v=np.asarray(v,float);grad=np.asarray(grad,float)
    m=beta1*m+(1-beta1)*grad
    v=beta2*v+(1-beta2)*(grad*grad)
    w=w-lr*weight_decay*w-lr*m/(np.sqrt(v)+eps)
    return w,m,v
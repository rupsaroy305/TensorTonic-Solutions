import numpy as np

def _sigmoid(x):
    return np.where(x>=0,1/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))

def _as2d(a,feat):
    a=np.asarray(a,dtype=float)
    if a.ndim==1:
        return a.reshape(1,feat),True
    return a,False

def gru_cell_forward(x,h_prev,params):
    x,one=_as2d(x,params["Wz"].shape[0])
    h_prev,_=_as2d(h_prev,params["Uz"].shape[0])

    z=_sigmoid(x@params["Wz"]+h_prev@params["Uz"]+params["bz"])
    r=_sigmoid(x@params["Wr"]+h_prev@params["Ur"]+params["br"])
    h=np.tanh(x@params["Wh"]+(r*h_prev)@params["Uh"]+params["bh"])
    h=(1-z)*h_prev+z*h

    return h[0] if one else h
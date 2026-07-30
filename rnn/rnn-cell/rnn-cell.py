import numpy as np

def rnn_cell(x_t,h_prev,W_xh,W_hh,b_h):
    x_t,h_prev=np.atleast_2d(x_t),np.atleast_2d(h_prev)
    y=np.tanh(x_t@W_xh.T+h_prev@W_hh.T+b_h)
    return y if len(y)>1 else y[0]
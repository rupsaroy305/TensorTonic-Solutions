import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        N,T,_=X.shape;H=self.hidden_dim
        h=C=np.zeros((N,H));y=[]
        for t in range(T):
            z=np.c_[h,X[:,t]]
            f=sigmoid(z@self.W_f.T+self.b_f)
            i=sigmoid(z@self.W_i.T+self.b_i)
            g=np.tanh(z@self.W_c.T+self.b_c)
            o=sigmoid(z@self.W_o.T+self.b_o)
            C=f*C+i*g
            h=o*np.tanh(C)
            y.append(h@self.W_y.T+self.b_y)
        return np.stack(y,1),h,C
import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        return np.round(np.tanh(np.array(z)@self.G_W),4).tolist()
    
    def discriminate(self, x):
        return np.round(1/(1+np.exp(-(np.array(x)@self.D_W))),4).tolist()
    
    def train_step(self, real_data, z):
        e=1e-8
        r=np.clip(1/(1+np.exp(-(np.array(real_data)@self.D_W))),e,1-e)
        f=np.clip(1/(1+np.exp(-(np.tanh(np.array(z)@self.G_W)@self.D_W))),e,1-e)
        return {"d_loss":round(float(-np.mean(np.log(r)+np.log(1-f))),4),
                "g_loss":round(float(-np.mean(np.log(f))),4)}
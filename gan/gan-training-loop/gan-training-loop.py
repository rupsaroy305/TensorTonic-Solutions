import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    e=1e-8
    s=lambda x:1/(1+np.exp(-x))
    r=np.clip(s(np.array(real_data)@np.array(D_W)),e,1-e)
    f=np.clip(s(np.array(fake_data)@np.array(D_W)),e,1-e)
    return {"d_loss":round(float(-np.mean(np.log(r)+np.log(1-f))),4),
            "g_loss":round(float(-np.mean(np.log(f))),4)}
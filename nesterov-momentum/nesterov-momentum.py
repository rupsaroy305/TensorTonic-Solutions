import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    new_v=momentum*np.array(v)+lr*np.array(grad)
    new_w=np.array(w)-new_v
    return {"new_w":new_w,"new_v":new_v}
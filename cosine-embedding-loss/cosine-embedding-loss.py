import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    x1,x2=np.array(x1),np.array(x2)
    c=(x1@x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
    return float(1-c if label==1 else max(0,c-margin))
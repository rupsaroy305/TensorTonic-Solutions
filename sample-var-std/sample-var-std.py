import numpy as np

def sample_var_std(x):
    import numpy as np
    x=np.array(x,float)
    v=x.var(ddof=1)
    return float(v),float(np.sqrt(v))
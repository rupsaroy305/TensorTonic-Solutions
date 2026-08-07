def xavier_initialization(W, fan_in, fan_out):
    import numpy as np
    l=np.sqrt(6/(fan_in+fan_out))
    return (np.array(W)*2*l-l).tolist()
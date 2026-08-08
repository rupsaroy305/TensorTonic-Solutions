def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    import numpy as np
    return float(min_lr+0.5*(base_lr-min_lr)*(1+np.cos(np.pi*current_step/total_steps)))
import numpy as np
def discriminator_loss(real_probs, fake_probs):
    e = 1e-8
    real_probs = np.clip(np.array(real_probs), e, 1-e)
    fake_probs = np.clip(np.array(fake_probs), e, 1-e)
    return float(-np.mean(np.log(real_probs) + np.log(1 - fake_probs)))

def generator_loss(fake_probs):
    e = 1e-8
    fake_probs = np.clip(np.array(fake_probs), e, 1-e)
    return float(-np.mean(np.log(fake_probs)))
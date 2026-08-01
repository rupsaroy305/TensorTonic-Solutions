import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    d=round(float(np.mean(np.std(np.array(generated_samples),axis=0))),4)
    return {"diversity_score":d,"is_collapsed":d<threshold}
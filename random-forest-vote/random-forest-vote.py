import numpy as np

def random_forest_vote(predictions):
    result=[]
    for i in range(len(predictions[0])):
        votes={}
        for tree in predictions:
            label=tree[i]
            votes[label]=votes.get(label,0)+1
        best_label=None
        best_count=-1
        for label,count in votes.items():
            if count>best_count or (count==best_count and label<best_label):
                best_count=count
                best_label=label
        result.append(best_label)
    return result
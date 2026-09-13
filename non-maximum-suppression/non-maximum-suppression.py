import numpy as np
def nms(boxes: list, scores: list, iou_threshold: float) -> list:
    if len(boxes) == 0:
        return []
    order = np.argsort(-np.array(scores), kind="stable")
    boxes = np.array(boxes, dtype=float)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        remaining = []
        for j in order[1:]:
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            iou = intersection / (area_i + area_j - intersection)
            if iou < iou_threshold:
                remaining.append(j)
        order = np.array(remaining)
    return keep
import math

def rotate_image(image: list, angle_degrees: float) -> list:
    H, W = len(image), len(image[0])
    cy, cx = (H - 1) / 2, (W - 1) / 2
    t = math.radians(angle_degrees)
    c, s = math.cos(t), math.sin(t)

    result = [[0] * W for _ in range(H)]

    for i in range(H):
        for j in range(W):
            dy, dx = i - cy, j - cx
            sy = round(cy + dy * c + dx * s)
            sx = round(cx - dy * s + dx * c)

            if 0 <= sy < H and 0 <= sx < W:
                result[i][j] = image[sy][sx]

    return result
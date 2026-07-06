def robust_scaling(values):

    if len(values) == 1:
        return [0.0]

    vals = sorted(values)
    n = len(vals)

    def median(arr):
        m = len(arr)
        if m % 2 == 1:
            return float(arr[m // 2])
        return (arr[m // 2 - 1] + arr[m // 2]) / 2.0

    med = median(vals)

    if n % 2 == 0:
        lower = vals[:n // 2]
        upper = vals[n // 2:]
    else:
        lower = vals[:n // 2]
        upper = vals[n // 2 + 1:]

    q1 = median(lower)
    q3 = median(upper)
    iqr = q3 - q1

    if iqr == 0:
        return [float(x - med) for x in values]

    return [float((x - med) / iqr) for x in values]
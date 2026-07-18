def linear_interpolation(values):
    res = values[:]
    n = len(res)
    i = 0
    while i < n:
        if res[i] is None:
            left = i - 1
            j = i
            while res[j] is None:
                j += 1
            right = j
            lv, rv = res[left], res[right]
            gap = right - left
            for k in range(1, gap):
                res[left + k] = lv + (rv - lv) * k / gap
            i = right
        else:
            i += 1
    return res
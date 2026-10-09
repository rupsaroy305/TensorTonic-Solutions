def winsorize(values: list, lower_pct: float, upper_pct: float) -> list:
    a = sorted(values)
    n = len(a)
    def percentile(p):
        k = (n-1)*p/100
        lo = int(k)
        hi = min(lo+1,n-1)
        return a[lo]+(k-lo)*(a[hi]-a[lo])
    low = percentile(lower_pct)
    high = percentile(upper_pct)
    return [max(low, min(x, high)) for x in values]
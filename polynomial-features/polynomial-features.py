def polynomial_features(values: list, degree: int) -> list:
    result=[]
    for x in values:
        row = [x**i for i in range(degree+1)]
        result.append(row)
    return result
def rank_transform(values: list) -> list:
    sorted_values= sorted(values)
    result= []
    for x in values:
        start= sorted_values.index(x)+1
        end= len(sorted_values)-sorted_values[::-1].index(x)
        result.append((start+end)/2)
    return result
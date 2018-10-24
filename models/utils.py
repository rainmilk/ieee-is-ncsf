def diversity(collection):
    div = 0
    for i in range(len(collection) - 1):
        curr_set = set(collection[i])
        for j in range(i + 1, len(collection)):
            cmp_set = set(collection[j])
            div += 1 - len(curr_set & cmp_set) / len(curr_set)

    return div / (0.5 * len(collection) * (len(collection) - 1))
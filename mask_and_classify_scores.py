import numpy as np

def mask_and_classify_scores(arr):

    if not isinstance(arr, np.ndarray):     # input validation
        return None

    if arr.ndim != 2:
        return None

    n, m = arr.shape
    if n != m or n < 4:
        return None

    cleaned = arr.copy()        # part A

    cleaned[cleaned < 0] = 0
    cleaned[cleaned > 100] = 100

    levels = np.zeros_like(cleaned, dtype=int)       # part B

    levels[cleaned >= 70] = 2
    levels[(cleaned >= 40) & (cleaned < 70)] = 1

    row_pass_counts = np.zeros(n, dtype=int)         # part C

    for i in range(n):
        count = 0
        for value in cleaned[i]:
            if value >= 50:
                count += 1
        row_pass_counts[i] = count

    return cleaned, levels, row_pass_counts


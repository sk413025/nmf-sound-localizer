import numpy as np
from typing import List


def angular_error(gt: List[int], pr: List[int], D: int) -> float:
    bin_deg = 360.0 / D
    diff = ((pr[0] - gt[0]) * bin_deg + 180) % 360 - 180
    return abs(diff)


def multi_match_error(gt: List[int], pr: List[int], D: int) -> float:
    from itertools import permutations
    bin_deg = 360.0 / D
    J = len(gt)
    gt = np.array(gt)
    best = 1e9
    for perm in permutations(pr, J):
        diff = ((np.array(perm) - gt) * bin_deg + 180) % 360 - 180
        best = min(best, float(np.mean(np.abs(diff))))
    return best


def accuracy_10deg(gt_list: List[List[int]], pr_list: List[List[int]], D: int) -> float:
    tol = 10.0
    ok = 0
    total = 0
    for gt, pr in zip(gt_list, pr_list):
        if angular_error([gt[0]], [pr[0]], D) <= tol:
            ok += 1
        total += 1
    return 100.0 * ok / max(total, 1)


def per_source_accuracy(gt_list: List[List[int]], pr_list: List[List[int]], D: int, tol: float = 10.0) -> float:
    accs = []
    for gt, pr in zip(gt_list, pr_list):
        J = len(gt)
        from itertools import permutations
        best = 0
        for perm in permutations(pr, J):
            good = 0
            for g, p in zip(gt, perm):
                if angular_error([g], [p], D) <= tol:
                    good += 1
            best = max(best, good)
        accs.append(best / J)
    return 100.0 * float(np.mean(accs)) if accs else 0.0


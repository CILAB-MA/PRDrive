from collections import Counter

SUCCESS_SOLVED = 0.999
MANY_FAIL_RETURN = 0.0
STRUCTURE_CATEGORIES = ("solved", "localized_failure", "globally_difficult", "many_fail")
MAX_LOCALIZED_CARS = 1
# Ties when pooling seeds: catastrophe > spread > one-car > solved.
_STRUCTURE_SEVERITY = {
    "many_fail": 0,
    "globally_difficult": 1,
    "localized_failure": 2,
    "solved": 3,
}


def failed_cars(log, score_key="score"):
    """Complete binary fails on a map: (1 - score) * n. Score is 0/1 per car."""
    n = float(log.get("n") or 0)
    if n <= 0 or score_key not in log:
        return 0.0
    return max((1.0 - float(log[score_key])) * n, 0.0)


def classify_structure(
    log,
    max_localized_cars=MAX_LOCALIZED_CARS,
    solved=SUCCESS_SOLVED,
    score_key="score",
    return_key="episode_return",
    many_fail_return=MANY_FAIL_RETURN,
):
    n = float(log.get("n") or 0)
    if n <= 0 or score_key not in log:
        return "solved"
    failed_n = int(round(failed_cars(log, score_key=score_key)))
    if failed_n <= 0:
        return "solved"
    ep_ret = log.get(return_key)
    try:
        if ep_ret is not None and float(ep_ret) < float(many_fail_return):
            return "many_fail"
    except (TypeError, ValueError):
        pass
    if failed_n <= int(max_localized_cars):
        return "localized_failure"
    return "globally_difficult"


def majority_structure(categories):
    cats = [c for c in categories if c]
    if not cats:
        return "solved"
    counts = Counter(cats)
    top = max(counts.values())
    tied = [c for c, n in counts.items() if n == top]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda c: _STRUCTURE_SEVERITY.get(c, 99))

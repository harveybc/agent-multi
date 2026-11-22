"""One budget-matched exploratory fork; no RNG draws or encoder parameters."""
import copy
import math

CYCLE = 20


def rate(step):
    if not 1 <= step <= CYCLE:
        raise ValueError("outside exploration cycle")
    return 1.2e-5 + (3e-4 - 1.2e-5) * (1 + math.cos(math.pi*(step-1)/(CYCLE-1)))/2


def clone_pair(model, optimizer):
    # Copy together: optimizer parameter references must point to the new model.
    candidate, opt = copy.deepcopy((model, optimizer))
    assert all(a is b for a,b in zip(candidate.parameters(), opt.param_groups[0]['params']))
    return candidate,opt

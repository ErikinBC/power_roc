import numpy as np
import pandas as pd

# MAKE INTO A CLASS
# SHOULD HAVE "TRUE" ROC CURVE

def dgp_bin(n, p, mu=1, seed=None):
    assert (n > 0) and isinstance(n, int)
    assert (p > 0) and (p < 1)
    if seed is not None:
        np.random.seed(seed)
    # Number of positive/negative samples
    n1 = np.random.binomial(n=n, p=p)
    n0 = n - n1
    # Generate positive and negative scores
    s1 = mu + np.random.randn(n1)
    s0 = np.random.randn(n0)
    scores = np.append(s1, s0)
    labels = np.append(np.repeat(1, n1), np.repeat(0, n0))
    res = pd.DataFrame({'y':labels, 's':scores})
    return res


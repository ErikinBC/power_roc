import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_curve as sk_roc_curve

# Fast method of calculating AUROC
def auc_rank(y, s):
    n1 = sum(y)
    n0 = len(y) - n1
    den = n0 * n1
    num = sum(rankdata(s)[y == 1]) - n1*(n1+1)/2
    auc = num / den
    return auc
    # n1n0=False
    # if n1n0:
    #     return auc, n1, n0

# SKLearn wrapper to calculate empirical ROC
def emp_roc_curve(y, s):
    fpr, tpr, thresh = sk_roc_curve(y, s)
    res = pd.DataFrame({'thresh':thresh, 'tpr':tpr, 'fpr':fpr})
    return res




# mu=1;p=0.1;n_points=500;ptail=1e-3
class dgp_bin():
    """
    mu:     Mean of positive class N(mu, 1)
    p:      Probability of an observation being a positive class
    """
    def __init__(self, mu, p):
        assert mu > 0, 'Mean needs to be greater than zero'
        self.mu = mu
        assert (p > 0) and (p < 1), 'p needs to be between 0 and 1'
        self.p = p
        # Calculate ground truth AUROC
        self.auroc = norm.cdf(mu/np.sqrt(2))

    def roc_curve(self, n_points=500, ptail=1e-3):
        s_lower = norm.ppf(ptail)
        s_upper = norm.ppf(1-ptail) + self.mu
        s_seq = np.linspace(s_lower, s_upper, n_points)
        tpr = 1 - norm.cdf(s_seq - self.mu)
        fpr = 1 - norm.cdf(s_seq)
        res = pd.DataFrame({'thresh':s_seq, 'tpr':tpr, 'fpr':fpr})
        return res

    def dgp_bin(self, n, seed=None):
        assert (n > 0) and isinstance(n, int), 'n needs to be an int > 0'
        if seed is not None:
            np.random.seed(seed)
        # Number of positive/negative samples
        n1 = np.random.binomial(n=n, p=self.p)
        n0 = n - n1
        # Generate positive and negative scores
        s1 = self.mu + np.random.randn(n1)
        s0 = np.random.randn(n0)
        scores = np.append(s1, s0)
        labels = np.append(np.repeat(1, n1), np.repeat(0, n0))
        res = pd.DataFrame({'y':labels, 's':scores})
        return res




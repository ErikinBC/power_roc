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

# SKLearn wrapper to calculate empirical ROC
def emp_roc_curve(y, s):
    fpr, sens, thresh = sk_roc_curve(y, s)
    spec = 1-fpr
    res = pd.DataFrame({'thresh':thresh, 'sens':sens, 'spec':spec})
    return res

# mu=1;p=0.1;n_points=500;ptail=1e-3
class dgp_bin():
    """
    mu:     Mean of positive class N(mu, 1)
    p:      Probability of an observation being a positive class
    """
    def __init__(self, mu, p, thresh=None, sens=None, spec=None):
        assert mu > 0, 'Mean needs to be greater than zero'
        self.mu = mu
        assert (p > 0) and (p < 1), 'p needs to be between 0 and 1'
        self.p = p
        # Calculate ground truth AUROC
        self.auroc_oracle = norm.cdf(mu/np.sqrt(2))
        # Calculate the oracle values
        self.thresh, self.sens, self.spec = thresh, sens, spec
        check_none = (thresh is not None) or (sens is not None) or (spec is not None)
        assert check_none, 'at least one field needs to be assigned: thresh, sens, or spec'
        if thresh is not None:
            self.thresh_oracle = thresh
        if sens is not None:
            self.thresh_oracle = norm.ppf(1-sens) + self.mu
        if spec is not None:
            self.thresh_oracle = norm.ppf(spec)
        

    def thresh2sens(self, thresh):
        sens = 1 - norm.cdf(thresh - self.mu)
        return sens

    def thresh2spec(self, thresh):
        spec = norm.cdf(thresh)
        return spec

    def roc_curve(self, n_points=500, ptail=1e-3):
        # Generate sequence of scores within distribution of 0 and 1 class
        s_lower = norm.ppf(ptail)
        s_upper = norm.ppf(1-ptail) + self.mu
        s_seq = np.linspace(s_lower, s_upper, n_points)
        sens = self.thresh2sens(s_seq)
        spec = self.thresh2spec(s_seq)
        res = pd.DataFrame({'thresh':s_seq, 'sens':sens, 'spec':spec})
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

    """
    Calculate the threshold based on empirical data
    If threshold was already supplied, empirical == oracle threshold
    If sens or spec was supplied, empirical != oracle threshol
    """
    def learn_threshold(self, y, s):
        if self.thresh is not None:
            self.thresh_emp = self.thresh_oracle
        if self.sens is not None:
            s1 = s[y == 1]
            self.thresh_emp = np.quantile(s1, 1-self.sens)
        if self.spec is not None:
            s0 = s[y == 0]
            self.thresh_emp = np.quantile(s0, self.spec)
        # Assign the empirical sensitivity and specificity
        self.sens_emp, self.spec_emp = self.get_tptn(y, s).values.flat
        # Get the oracle sensitivity and specificity based on the learned threshold
        self.sens_oracle = self.thresh2sens(self.thresh_emp)
        self.spec_oracle = self.thresh2spec(self.thresh_emp)
        # Save labels and score for later
        self.y_thresh = np.array(y)
        self.s_thresh = np.array(s)
        
    def predict(self, s):
        yhat = np.where(s >= self.thresh_emp, 1, 0)
        return yhat

    def get_tptn(self, y, s):
        yhat = self.predict(s)
        sens = np.mean(yhat[y == 1] == 1)
        spec = np.mean(yhat[y == 0] == 0)
        res = pd.DataFrame({'sens':sens, 'spec':spec}, index=[0])
        return res

    def create_df(self):
        msr = np.repeat(['sens', 'spec', 'thresh'], 2)
        tt = np.tile(['oracle','emp'],3)
        val = [self.sens_oracle, self.sens_emp, self.spec_oracle, self.spec_emp, self.thresh_oracle, self.thresh_emp]
        self.df_rv = pd.DataFrame({'msr':msr, 'tt':tt, 'val':val})

    """
    Function to calculate power of a binomial proportion for a 1-sided test

    p_act:                  True proportion
    p_null:                 Null hypothesis proportion
    n:                      Sample size
    alpha:                  Type-I error rate
    """
    @staticmethod
    def power_binom(p_act, p_null, n, alpha=0.05):
        assert (alpha > 0) and (alpha < 1), 'Type-I error must be between 0 to 1'
        t_alpha = norm.ppf(1-alpha)
        sigma_act = np.sqrt(p_act*(1-p_act)/n)
        sigma_null = np.sqrt(p_null*(1-p_null)/n)
        p_delta = p_act - p_null
        power = 1 - norm.cdf( (sigma_null*t_alpha - p_delta)/sigma_null )
        return power


    """
    Carry out the studentized bootstrapped on the test-set scores to determine 95% CI for sens/spec
    Because power is a monotonic function of sens/spec, these bounds can be used to estimate power
    http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
    """
    def run_bootstrap(self, s1, s0, alpha, n_bs=1000, n_student=None, seed=None):
        assert (s1 is not None) or (s0 is not None), 'Specify either s0 or s1'
        if n_student is None:
            n_student = n_bs
        alpha_vec = np.array([1-alpha/2, alpha/2])
        # (i) Bootstrap distribution
        n1, n0 = len(s1), len(s0)
        s1_bs = s1.sample(frac=n_bs,replace=True,random_state=seed).values.reshape([n_bs, n1])
        s0_bs = s0.sample(frac=n_bs,replace=True,random_state=seed).values.reshape([n_bs, n0])
        sens_bs = np.mean(s1_bs >= self.thresh_emp, 1)
        spec_bs = np.mean(s0_bs < self.thresh_emp, 1)
        sens_se = sens_bs.std(ddof=1)
        spec_se = spec_bs.std(ddof=1)

        # (ii) Bootstrap the bootstrapped samples to get a variance of each observation
        sens_se_student, spec_se_student = np.zeros(n_bs), np.zeros(n_bs)
        for j in range(n_bs):
            # Studentized score distribution
            s1_student = pd.Series(s1_bs[j]).sample(frac=n_student,replace=True,random_state=j).values.reshape([n_student, n1])
            sens_student = np.mean(s1_student >= self.thresh_emp, 1)
            sens_se_student[j] = sens_student.std(ddof=1)
            s0_student = pd.Series(s0_bs[j]).sample(frac=n_student,replace=True,random_state=j).values.reshape([n_student, n0])
            spec_student = np.mean(s0_student >= self.thresh_emp, 1)
            spec_se_student[j] = spec_student.std(ddof=1)
        t_sens = (sens_bs - self.sens_emp) / sens_se_student
        t_spec = (spec_bs - self.spec_emp) / spec_se_student

        # Find the alpha-quantile using the studentized formula
        t_alpha_sens = np.quantile(t_sens, alpha_vec)
        t_alpha_spec = np.quantile(t_spec, alpha_vec)

        # (iii) Get the confidence intervals
        sens_ci = self.sens_emp - t_alpha_sens*sens_se
        spec_ci = self.spec_emp - t_alpha_spec*spec_se
        res_sens = pd.DataFrame({'msr':'sens', 'tt':['lb','ub'], 'val':sens_ci})
        res_spec = pd.DataFrame({'msr':'spec', 'tt':['lb','ub'], 'val':spec_ci})
        res = pd.concat(objs=[res_sens, res_spec], axis=0).reset_index(drop=True)
        return res
        

    """
    null_sens:              Sensitivity null hypothesis to be rejected during silent period
    null_spec:              Specificity null hypothesis to be rejected during silent period
    n1_trail:               Number of positive samples during trial
    n0_trial:               Number of negative samples during trial
    """
    def run_power(self, null_sens, null_spec, n1_trial, n0_trial, alpha=0.05, n_bs=1000, seed=1):
        if null_sens > self.sens_emp:
            print('Warning! Null hypothesis for sensitivity should be less than empirical sensitivity')
        if null_spec > self.spec_emp:
            print('Warning! Null hypothesis for specificity should be less than empirical sensitivity')
        assert isinstance(n1_trial, int) and n1_trial > 0
        assert isinstance(n0_trial, int) and n0_trial > 0
        s1 = pd.Series(self.s_thresh[self.y_thresh == 1])
        s0 = pd.Series(self.s_thresh[self.y_thresh == 0])
        # Generate bootstrapped range of empirical sens/spec
        perf_ci = self.run_bootstrap(s1, s0, alpha, n_bs=n_bs, n_student=n_bs, seed=seed)
        # Merge on null and sample size
        lst_msr = ['sens','spec']
        lst_n = [n1_trial, n0_trial]
        lst_h0 = [null_sens, null_spec]
        dat_n = pd.DataFrame({'msr':lst_msr,'n':lst_n, 'h0':lst_h0})
        self.power_ci = perf_ci.merge(dat_n)
        self.power_ci = self.power_ci.assign(power=lambda x: self.power_binom(p_act=x['val'], p_null=x['h0'], n=x['n'], alpha=alpha))
        # Determine the "true" power
        power_sens = self.power_binom(p_act=self.sens_oracle, p_null=null_sens, n=n1_trial, alpha=alpha)
        power_spec = self.power_binom(p_act=self.spec_oracle, p_null=null_spec, n=n0_trial, alpha=alpha)
        tmp_df = pd.DataFrame({'msr':lst_msr, 'power':[power_sens, power_spec]})
        self.df_power = tmp_df.merge(self.power_ci.pivot('msr','tt','power').reset_index())


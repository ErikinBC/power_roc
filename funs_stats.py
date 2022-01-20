import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, t
from sklearn.metrics import roc_curve as sk_roc_curve
from statsmodels.stats.proportion import proportion_confint as prop_CI

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

# Add on CIs to a dataframe
def get_CI(df, cn_n, den, method='beta', alpha=0.05):
    tmp = pd.concat(prop_CI(df[cn_n], den, method=method),axis=1)
    tmp.columns = ['lb', 'ub']
    df = pd.concat(objs=[df, tmp], axis=1)
    return df


class dgp_bin():
    """
    mu:     Mean of positive class N(mu, 1)
    p:      Probability of an observation being a positive class
    """
    def __init__(self, mu, p, thresh=None, sens=None, spec=None, alpha=0.05):
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
        # Set up statistical testing
        assert (alpha > 0) and (alpha < 1), 'Type-I error must be between 0 to 1'
        self.alpha = alpha
        self.t_alpha = norm.ppf(1-alpha)

    # Oracle sensitivity for Gaussian distribution
    def thresh2sens(self, thresh):
        sens = 1 - norm.cdf(thresh - self.mu)
        return sens

    # Oracle specificity for Gaussian distribution
    def thresh2spec(self, thresh):
        spec = norm.cdf(thresh)
        return spec

    # Convert scores into binary outcomes
    def predict(self, s):
        yhat = np.where(s >= self.thresh_emp, 1, 0)
        return yhat

    # Get sens/spec after threshold has been set
    def get_tptn(self, y, s):
        yhat = self.predict(s)
        sens = np.mean(yhat[y == 1] == 1)
        spec = np.mean(yhat[y == 0] == 0)
        res = pd.DataFrame({'sens':sens, 'spec':spec}, index=[0])
        return res

    # Return a DataFrame for ROC plotting
    def roc_curve(self, n_points=500, ptail=1e-3):
        # Generate sequence of scores within distribution of 0 and 1 class
        s_lower = norm.ppf(ptail)
        s_upper = norm.ppf(1-ptail) + self.mu
        s_seq = np.linspace(s_lower, s_upper, n_points)
        sens = self.thresh2sens(s_seq)
        spec = self.thresh2spec(s_seq)
        res = pd.DataFrame({'thresh':s_seq, 'sens':sens, 'spec':spec})
        return res

    # Generate data from underlying parametric distribution
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

    """
    Collate the results after learn_threshold has been run
    """
    def create_df(self):
        msr = np.repeat(['sens', 'spec', 'thresh'], 2)
        tt = np.tile(['oracle','emp'],3)
        val = [self.sens_oracle, self.sens_emp, self.spec_oracle, self.spec_emp, self.thresh_oracle, self.thresh_emp]
        self.df_rv = pd.DataFrame({'msr':msr, 'tt':tt, 'val':val})

    """
    Set up the null hypothesis (usually run after learn_threshold)
    null_sens:              Sensitivity null hypothesis to be rejected during silent period
    null_spec:              Specificity null hypothesis to be rejected during silent period
    """    
    def set_null_hypotheis(self, null_sens, null_spec):
        if hasattr(self, 'sens_emp'):
            if null_sens > self.sens_emp:
                print('Warning! Null hypothesis for sensitivity should be less than empirical sensitivity')
        if hasattr(self, 'sens_emp'):
            if null_spec > self.spec_emp:
                print('Warning! Null hypothesis for specificity should be less than empirical sensitivity')
        self.null_sens, self.null_spec = null_sens, null_spec


    """
    Function to calculate power of a binomial proportion for a 1-sided test

    p_act:                  True proportion
    p_null:                 Null hypothesis proportion
    n:                      Sample size
    alpha:                  Type-I error rate
    """
    def power_binom(self, p_act, p_null, n, alpha=None):
        t_alpha = self.t_alpha
        if alpha is not None:
            t_alpha = norm.ppf(1-alpha)
        sigma_act = np.sqrt(p_act*(1-p_act)/n)
        sigma_null = np.sqrt(p_null*(1-p_null)/n)
        p_delta = p_act - p_null
        power = 1 - norm.cdf( (sigma_null*t_alpha - p_delta)/sigma_act )
        return power

    """
    Return the z-statistic and p-value of a (series) of tests
    """
    def binom_stat(self, p_act, p_null, n, alpha=None):
        t_alpha = self.t_alpha
        if alpha is not None:
            t_alpha = norm.ppf(1-alpha) 
        sigma_null = np.sqrt(p_null*(1-p_null)/n)
        z_test = (p_act - p_null) / sigma_null
        # One-sided test
        p_test = pd.Series(1 - norm.cdf(z_test))
        reject = np.where(z_test > t_alpha, 1, 0)
        res = pd.DataFrame({'z':z_test, 'pval':p_test, 'reject':reject}, index=p_test.index)
        return res


    """
    Run bootstrap for sensitivity/specificity
    For an explaination of different bootstrap methodologies: http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf

    basic:          Uses only bootstrap standard error for classic symmetric CIs
    quantile:       Uses [a/2, 1-a/2] quantiles of the bootstrap distribution
    expanded:       Modifies the percentiles of the quantile approach for sample-size
    studentized:    Bootstraps each bootstrapped samples to get a t-dist
    BCa:            Bias-corrected and accelerated approach
    """
    # method=None;seed=i
    def run_bootstrap(self, s1, s0, n1_trial, n0_trial, method='basic', n_bs=1000, seed=None, alpha=None):
        lst_method = ['basic', 'quantile', 'expanded', 'studentized', 'bca']
        di_leaf = {'sens':None, 'spec':None}
        if method is not None:
            assert method in lst_method
            di_ci = {method:di_leaf}
        else:
            # Calculate all methods except studentized
            di_ci = dict.fromkeys([m for m in lst_method if m is not 'studentized'])
            di_ci = {k:di_leaf.copy() for k in di_ci.keys()}
        if alpha is None:
            alpha = self.alpha
        alpha_vec = np.array([1-alpha/2, alpha/2])

        # (i) Get the baseline statistic
        n1, n0 = len(s1), len(s0)
        sens_hat = np.mean(s1 >= self.thresh_emp)
        spec_hat = np.mean(s0 < self.thresh_emp)
        power_sens = self.power_binom(sens_hat, self.null_sens, n1_trial)
        power_spec = self.power_binom(spec_hat, self.null_spec, n0_trial)
        
        # (ii) Bootstrap power distribution
        s1_bs = s1.sample(frac=n_bs,replace=True,random_state=seed).values.reshape([n_bs, n1])
        s0_bs = s0.sample(frac=n_bs,replace=True,random_state=seed).values.reshape([n_bs, n0])
        sens_bs = np.mean(s1_bs >= self.thresh_emp, 1)
        spec_bs = np.mean(s0_bs < self.thresh_emp, 1)
        power_bs_sens = self.power_binom(sens_bs, self.null_sens, n1_trial)
        power_bs_spec = self.power_binom(spec_bs, self.null_spec, n0_trial)
        power_bs_sens_se = power_bs_sens.std(ddof=1)
        power_bs_spec_se = power_bs_spec.std(ddof=1)

        if 'basic' in di_ci:
            di_ci['basic']['sens'] = power_sens - power_bs_sens_se * norm.ppf(alpha_vec)
            di_ci['basic']['spec'] = power_spec - power_bs_spec_se * norm.ppf(alpha_vec)
        
        if 'quantile' in di_ci:
            di_ci['quantile']['sens'] = np.flip(np.quantile(power_bs_sens, alpha_vec))
            di_ci['quantile']['spec'] = np.flip(np.quantile(power_bs_spec, alpha_vec))

        if 'expanded' in di_ci:
            alpha_adj1 = norm.cdf(np.sqrt(n1_trial/(n1_trial-1))*t(df=n1_trial-1).ppf(alpha/2))
            alpha_adj0 = norm.cdf(np.sqrt(n0_trial/(n0_trial-1))*t(df=n0_trial-1).ppf(alpha/2))
            alpha_adj1 = [alpha_adj1/2, 1-alpha_adj1/2]
            alpha_adj0 = [alpha_adj0/2, 1-alpha_adj0/2]
            di_ci['expanded']['sens'] = np.quantile(power_bs_sens, alpha_adj1)
            di_ci['expanded']['spec'] = np.quantile(power_bs_spec, alpha_adj0)

        if 'studentized' in di_ci:
            # Bootstrap the bootstrapped samples to get a variance of each observation
            # n_student == n_bs for computational simplicity
            power_sens_stud_se, power_spec_stud_se = np.zeros(n_bs), np.zeros(n_bs)
            for j in range(n_bs):
                # Studentized score distribution
                s1_stud = pd.Series(s1_bs[j]).sample(frac=n_bs,replace=True,random_state=j).values.reshape([n_bs, n1])
                s0_stud = pd.Series(s0_bs[j]).sample(frac=n_bs,replace=True,random_state=j).values.reshape([n_bs, n0])
                # Studentized sens/spec
                sens_stud = np.mean(s1_stud >= self.thresh_emp, 1)
                spec_stud = np.mean(s0_stud < self.thresh_emp, 1)
                # Studentized power SE
                power_sens_stud = self.power_binom(sens_stud, self.null_sens, n1_trial)
                power_spec_stud = self.power_binom(spec_stud, self.null_spec, n0_trial)
                power_sens_stud_se[j] = power_sens_stud.std(ddof=1)
                power_spec_stud_se[j] = power_spec_stud.std(ddof=1)
            # Find the alpha-quantile using the studentized formula
            t_sens = (power_bs_sens - power_sens) / power_sens_stud_se
            t_spec = (power_bs_spec - power_spec) / power_spec_stud_se
            t_alpha_sens = np.quantile(t_sens, alpha_vec)
            t_alpha_spec = np.quantile(t_spec, alpha_vec)
            alpha_adj1 = np.flip(norm.cdf(t_alpha_sens))
            alpha_adj0 = np.flip(norm.cdf(t_alpha_spec))
            di_ci['studentized']['sens'] = np.quantile(power_bs_sens, alpha_adj1)
            di_ci['studentized']['spec'] = np.quantile(power_bs_spec, alpha_adj0)

        if 'bca' in di_ci:
            # Leave-one-out sens/spec
            tp1, tn1 = int(sens_hat*n1), int(spec_hat*n0)
            sens_hat_loo = np.append(np.repeat((tp1-1)/(n1-1),tp1), np.repeat(tp1/(n1-1),n1-tp1))
            spec_hat_loo = np.append(np.repeat((tn1-1)/(n0-1), tn1), np.repeat(tn1/(n0-1), n0-tn1))
            power_sens_loo = self.power_binom(sens_hat_loo, self.null_sens, n1_trial)
            power_spec_loo = self.power_binom(spec_hat_loo, self.null_spec, n0_trial)
            # Calculate acceleration parameter
            num1 = np.sum((power_sens_loo.mean() - power_sens_loo)**3)
            den1 = 6*np.sum((power_sens_loo.mean() - power_sens_loo)**2)**(3/2)
            a1 = num1/den1
            num0 = np.sum((power_spec_loo.mean() - power_spec_loo)**3)
            den0 = 6*np.sum((power_spec_loo.mean() - power_spec_loo)**2)**(3/2)
            a0 = num0/den0
            # Calculate bias correction parameter
            zalpha = norm.ppf([alpha/2, 1-alpha/2])
            zhat1 = norm.ppf( (np.sum(power_bs_sens < power_sens)+1)/(n_bs+1) )
            zhat0 = norm.ppf( (np.sum(power_bs_spec < power_spec)+1)/(n_bs+1) )
            alpha_adj1 = norm.cdf(zhat1 + (zhat1+zalpha)/(1-a1*(zhat1+zalpha)))
            alpha_adj0 = norm.cdf(zhat0 + (zhat0+zalpha)/(1-a0*(zhat0+zalpha)))
            di_ci['bca']['sens'] = np.quantile(power_bs_sens, alpha_adj1)
            di_ci['bca']['spec'] = np.quantile(power_bs_spec, alpha_adj0)
        # Merge all
        res = pd.DataFrame.from_dict(di_ci,orient='index').reset_index()
        res = res.rename(columns={'index':'method'}).melt('method',None,'msr')
        res = res.explode('value').assign(idx=lambda x: x.groupby(['method','msr']).cumcount())
        res = res.pivot_table('value',['method','msr'],'idx', lambda x: x)
        res = res.rename(columns={0:'lb', 1:'ub'}).reset_index()
        return res
        

    """
    n1_trial:               Number of positive samples during trial
    n0_trial:               Number of negative samples during trial
    method:                 Bootstrapping method (see run_bootstrap)
    seed:                   Will seed the bootstrap
    n_bs:                   Number of bootstrap/studentitzed iterations
    alpha:                  None defaults to inherited attributed
    """
    def run_power(self, n1_trial, n0_trial, method=None, seed=1, n_bs=1000, alpha=None):
        assert isinstance(n1_trial, int) and n1_trial > 0
        assert isinstance(n0_trial, int) and n0_trial > 0
        s1 = pd.Series(self.s_thresh[self.y_thresh == 1])
        s0 = pd.Series(self.s_thresh[self.y_thresh == 0])
        # Generate bootstrapped range of empirical sens/spec
        perf_ci = self.run_bootstrap(s1, s0, n1_trial, n0_trial, method=method, n_bs=n_bs, seed=seed, alpha=alpha)
        # Merge  on null and sample size
        lst_msr = ['sens','spec']
        lst_n = np.array([n1_trial, n0_trial])
        lst_h0 = np.array([self.null_sens, self.null_spec])
        lst_oracle = np.array([self.sens_oracle, self.spec_oracle])
        lst_gt_power = self.power_binom(p_act=lst_oracle, p_null=lst_h0, n=lst_n, alpha=alpha)
        dat_gt = pd.DataFrame({'msr':lst_msr,'n_trial':lst_n, 'h0':lst_h0, 'power':lst_gt_power})
        self.df_power = dat_gt.merge(perf_ci)

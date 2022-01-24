import numpy as np
import pandas as pd
from scipy import randn
from scipy.stats import rankdata, norm, t, skewnorm
from scipy.optimize import minimize_scalar
from statsmodels.stats.proportion import proportion_confint as prop_CI
from funs_support import rvec, cvec, quantile_mapply


# Probability that one skewnorm is less than another
def sn_ineq(mu, skew, scale, alpha, n_points):
    dist_y1 = skewnorm(a=skew, loc=mu, scale=scale)
    dist_y0 = skewnorm(a=skew, loc=0, scale=scale)
    x_seq = np.linspace(dist_y0.ppf(alpha), dist_y1.ppf(1-alpha), n_points)
    dx = x_seq[1] - x_seq[0]
    prob = np.sum(dist_y0.cdf(x_seq)*dist_y1.pdf(x_seq)*dx)
    return prob

# Find the mean of a skewnorm that achieves a certain AUROC
def find_auroc(auc, skew, scale=1, alpha=0.001, n_points=100):
    optim = minimize_scalar(fun=lambda mu: (auc - sn_ineq(mu, skew, scale, alpha, n_points))**2,method='brent')
    assert optim.fun < 1e-10
    mu_star = optim.x
    return mu_star

# Fast method of calculating AUROC
def auc_rank(y, s):
    n1 = sum(y)
    n0 = len(y) - n1
    den = n0 * n1
    num = sum(rankdata(s)[y == 1]) - n1*(n1+1)/2
    auc = num / den
    return auc

# SKLearn wrapper to calculate empirical ROC
def emp_roc_curve(y, s, n_points=100, tol=1e-10):
    # dat_test=df_scores.query('ds=="iowa"')
    # y=dat_test['y']; s=dat_test['s']; n_points=100; tol=1e-10
    s1, s0 = s[y == 1], s[y == 0]
    s1 = np.sort(np.unique(s1))
    s0 = np.sort(np.unique(s0))
    thresh_seq = np.flip(np.sort(np.append(s1, s0)))
    # Get range of sensitivity and specificity
    sens = [np.mean(s1 >= thresh) for thresh in thresh_seq]
    spec = [np.mean(s0 < thresh) for thresh in thresh_seq]
    # Store
    res = pd.DataFrame({'thresh':thresh_seq, 'sens':sens, 'spec':spec})
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
        # Save scores used to generate the threshold for later
        self.s1_thresh = np.array(s[y == 1])
        self.s0_thresh = np.array(s[y == 0])


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
            if np.any(null_sens > self.sens_emp):
                print('Warning! At least one null hypothesis is greater than empirical sensitivity')
        if hasattr(self, 'sens_emp'):
            if np.any(null_spec > self.spec_emp):
                print('Warning! At least one null hypothesis is greater than empirical specificity')
        self.null_sens = rvec(null_sens)
        self.null_spec = rvec(null_spec)

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
    # p_act=vec_act; p_null=vec_null; n=vec_n
    def binom_stat(self, p_act, p_null, n, alpha=None):
        t_alpha = self.t_alpha
        if alpha is not None:
            t_alpha = norm.ppf(1-alpha) 
        sigma_null = np.sqrt(p_null*(1-p_null)/n)
        z_test = (p_act - p_null) / sigma_null
        # One-sided test
        p_test = 1 - norm.cdf(z_test)
        reject = np.where(z_test > t_alpha, 1, 0)
        idx = range(len(p_test))
        res = pd.DataFrame({'z':z_test, 'pval':p_test, 'reject':reject}, index=idx)
        return res

    @staticmethod
    def clean_h0(arr, h0):
        di_h0 = dict(zip(range(h0.shape[1]), list(h0.flat)))
        res = pd.DataFrame(arr).assign(tt=['lb','ub']).melt('tt',None,'h0')
        res['h0'] = res['h0'].map(di_h0)
        res = res.pivot('h0','tt','value').reset_index()
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
    # method=None;seed=i;n_trial=100;msr='sens';alpha=None
    def run_bootstrap(self, msr, n_trial, method='basic', n_bs=1000, seed=None, alpha=None):
        lst_method = ['basic', 'quantile', 'expanded', 'studentized', 'bca']
        if method is not None:
            assert method in lst_method
            di_ci = {method:None}
        else:
            # Calculate all methods except studentized
            di_ci = dict.fromkeys([m for m in lst_method if m is not 'studentized'])
        if alpha is None:
            alpha = self.alpha
        alpha_vec = np.array([1-alpha/2, alpha/2])
        # Assign scores and threshold
        if msr == 'sens':
            scores = pd.Series(self.s1_thresh)
            thresh = self.thresh_emp
            h0 = self.null_sens
        else:  # Remove the equality condition for specificity (>=) with 1e-10
            scores = pd.Series(-self.s0_thresh)
            thresh = -self.thresh_emp + 1e-10
            h0 = self.null_spec
        n = len(scores)
        
        # (i) Get the baseline statistic
        theta_hat = np.mean(scores >= thresh)
        power_hat = self.power_binom(theta_hat, h0, n_trial)
        
        # (ii) Bootstrap power distribution
        scores_bs = scores.sample(frac=n_bs,replace=True,random_state=seed)
        scores_bs = scores_bs.values.reshape([n_bs, n])
        theta_bs = cvec(np.mean(scores_bs >= thresh, 1))
        power_bs = self.power_binom(theta_bs, h0, n_trial)
        power_bs_se = rvec(power_bs.std(ddof=1,axis=0))
        
        if 'basic' in di_ci:
            arr = power_hat - power_bs_se * cvec(norm.ppf(alpha_vec))
            di_ci['basic'] = self.clean_h0(arr, h0)
        
        if 'quantile' in di_ci:
            arr = np.quantile(power_bs,np.flip(alpha_vec),axis=0)
            di_ci['quantile'] = self.clean_h0(arr, h0)

        if 'expanded' in di_ci:
            alpha_adj = norm.cdf(np.sqrt(n_trial/(n_trial-1))*t(df=n_trial-1).ppf(alpha/2))
            alpha_adj = [alpha_adj/2, 1-alpha_adj/2]
            arr = np.quantile(power_bs, alpha_adj, axis=0)
            di_ci['expanded'] = self.clean_h0(arr, h0)

        if 'studentized' in di_ci:
            # Bootstrap the bootstrapped samples to get a variance of each observation
            # n_student == n_bs for simplicity
            power_stud_se = np.zeros([n_bs, h0.shape[1]])
            for j in range(n_bs):
                # Studentized score distribution
                scores_bs_j = pd.Series(scores_bs[j])
                scores_stud = scores_bs_j.sample(frac=n_bs,replace=True,random_state=j)
                scores_stud = scores_stud.values.reshape([n_bs, n])
                # Studentized performance
                theta_stud = cvec(np.mean(scores_stud >= self.thresh_emp, 1))
                # Studentized power SE
                power_stud = self.power_binom(theta_stud, h0, n_trial)
                power_stud_se[j] = power_stud.std(ddof=1, axis=0)                
            # Find the alpha-quantile using the studentized formula
            t_power = (power_bs - power_hat) / power_stud_se
            t_alpha = np.quantile(t_power, np.flip(alpha_vec), 0)
            # Backfill values with insufficient variation
            t_alpha = np.where(np.abs(t_alpha) == np.inf, np.nan, t_alpha)
            t_alpha = pd.DataFrame(t_alpha.T).fillna(method='bfill')
            t_alpha = t_alpha.values.T
            # Get implied percentile
            alpha_adj = norm.cdf(t_alpha)
            arr = quantile_mapply(power_bs, alpha_adj)
            di_ci['studentized'] = self.clean_h0(arr, h0)


        if 'bca' in di_ci:
            # Leave-one-out sens/spec
            n_acc = theta_hat * n
            theta_loo = cvec(np.append(np.repeat((n_acc-1)/(n-1),n_acc), np.repeat(n_acc/(n-1),n-n_acc)))
            power_loo = self.power_binom(theta_loo, h0, n_trial)
            # Calculate acceleration parameter
            num = rvec(np.sum((power_loo.mean(0) - power_loo)**3, 0))
            den = 6*rvec(np.sum((power_loo.mean(0) - power_loo)**2, 0)**(3/2))
            a = num/den
            a = np.where(np.isnan(a), 0, a)
            # Calculate bias correction parameter
            zalpha = cvec(norm.ppf([alpha/2, 1-alpha/2]))
            zhat = rvec(norm.ppf( (np.sum(power_bs <= power_hat, axis=0)+1)/(n_bs+1) ))
            # For infinities, fill with very large number instead
            zhat = np.where(zhat == np.inf, 10, zhat)
            alpha_adj = norm.cdf(zhat + (zhat+zalpha)/(1-a*(zhat+zalpha)))
            arr = quantile_mapply(power_bs, alpha_adj)
            di_ci['bca'] = self.clean_h0(arr, h0)
        # Merge all
        res = pd.concat(objs=[v.assign(method=k) for k,v in di_ci.items()], axis=0)
        res.reset_index(drop=True, inplace=True)
        res.insert(0, 'msr', msr)
        return res
        

    """
    msr:                    Whether power analysis is being run for "sens", "spec", or "both"
    n1_trial:               Number of positive samples during trial
    n0_trial:               Number of negative samples during trial
    method:                 Bootstrapping method (see run_bootstrap)
    seed:                   Will seed the bootstrap
    n_bs:                   Number of bootstrap/studentitzed iterations
    alpha:                  None defaults to inherited attributed
    """
    # msr='spec';method='quantile';seed=1;#;n1_trial=100;n0_trial=100
    def run_power(self, msr, n1_trial=None, n0_trial=None, method=None, seed=1, n_bs=1000, alpha=None):
        assert msr in ['sens','spec','both'], 'msr must be either "sens" or "spec" or "both"'
        
        # (i) Bootstrap CI around power
        holder = []
        if (msr == 'sens') or (msr == 'both'):
            assert isinstance(n1_trial, int) and n1_trial > 0
            res_sens = self.run_bootstrap('sens', n1_trial, method, n_bs, seed, alpha)
            holder.append(res_sens)
        if (msr == 'spec') or (msr == 'both'):
            assert isinstance(n0_trial, int) and n0_trial > 0
            res_spec = self.run_bootstrap('spec', n0_trial, method, n_bs, seed, alpha)
            holder.append(res_spec)
        df_ci = pd.concat(holder)
        
        # (ii) Oracle power
        holder = []
        if (msr == 'sens') or (msr == 'both'):
            power_sens = self.power_binom(p_act=self.sens_oracle, p_null=self.null_sens, n=n1_trial, alpha=alpha)
            power_sens = pd.DataFrame({'msr':'sens', 'n_trial':n1_trial, 'power':power_sens.flat, 'h0': self.null_sens.flat})
            holder.append(power_sens)
        if (msr == 'spec') or (msr == 'both'):
            power_spec = self.power_binom(p_act=self.spec_oracle, p_null=self.null_spec, n=n0_trial, alpha=alpha)
            power_spec = pd.DataFrame({'msr':'spec', 'n_trial':n0_trial, 'power':power_spec.flat, 'h0': self.null_spec.flat})
            holder.append(power_spec)
        df_power = pd.concat(holder)
        
        # (iii) Merge
        self.df_power = df_ci.merge(df_power)


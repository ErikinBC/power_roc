import os
import numpy as np
import pandas as pd
import plotnine as pn
import patchworklib as pw
from matplotlib import rc
from time import time
from scipy.stats import skewnorm, norm
from scipy.optimize import minimize_scalar
from funs_support import makeifnot, gg_save, grid_save, interp_df
from funs_stats import dgp_bin, emp_roc_curve, auc_rank, get_CI
dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
makeifnot(dir_figures)

# Tidy up the different labels
di_msr = {'sens':'Sensitivity', 'spec':'Specificity', 'thresh':'Threshold'}
di_method = {'basic':'Basic', 'quantile':'Quantile', 'expanded':'Quantile (t-adj)', 'studentized':'Studentized', 'bca':'BCa'}


############################
# --- (0) ROC EXAMPLES --- #

# Function to approximate one skew normal being larger than another
def sn_ineq(mu, skew, scale, alpha, n_points):
    dist_y1 = skewnorm(a=skew, loc=mu, scale=scale)
    dist_y0 = skewnorm(a=skew, loc=0, scale=scale)
    x_seq = np.linspace(dist_y0.ppf(alpha), dist_y1.ppf(1-alpha), n_points)
    dx = x_seq[1] - x_seq[0]
    prob = np.sum(dist_y0.cdf(x_seq)*dist_y1.pdf(x_seq)*dx)
    return prob

def find_auroc(auc, skew, scale=1, alpha=0.001, n_points=100):
    optim = minimize_scalar(fun=lambda mu: (auc - sn_ineq(mu, skew, scale, alpha, n_points))**2,method='brent')
    assert optim.fun < 1e-10
    mu_star = optim.x
    return mu_star
    
n1, n0 = 1000, 1000
labels = np.append(np.repeat(1,n1), np.repeat(0,n0))
auc_target = 0.75
skew_seq = [-4, 0, 4]
di_skew = dict(zip(skew_seq, ['Left-skew','No skew','Right skew']))
holder_dist, holder_roc = [], []
np.random.seed(n1)
for skew in skew_seq:
    skew_lbl = di_skew[skew]
    # Find AUROC equivalent mean
    mu_skew = find_auroc(auc=auc_target, skew=skew)
    dist_y1 = skewnorm(a=skew, loc=mu_skew, scale=1)
    dist_y0 = skewnorm(a=skew, loc=0, scale=1)
    scores = np.append(dist_y1.rvs(n1), dist_y0.rvs(n0))
    emp_auc = auc_rank(labels, scores)
    print('Skew = %s, AUROC=%0.3f' % (skew_lbl, emp_auc))
    df_dist = pd.DataFrame({'skew':skew_lbl,'y':labels, 's':scores})
    df_roc = emp_roc_curve(labels, scores).assign(skew=skew_lbl)
    holder_dist.append(df_dist)
    holder_roc.append(df_roc)
# Merge
roc_skew = pd.concat(holder_roc).reset_index(drop=True)
dist_skew = pd.concat(holder_dist).reset_index(drop=True)
auroc_skew = dist_skew.groupby('skew').apply(lambda x: auc_rank(x['y'],x['s'])).reset_index()
auroc_skew.rename(columns={0:'auroc'}, inplace=True)

# (i) Empirical ROC curves and skew
gg_roc_skew = (pn.ggplot(roc_skew,pn.aes(x='1-spec',y='sens',color='skew')) + 
    pn.theme_bw() + pn.labs(x='1-Specificity',y='Sensitivity') + 
    pn.geom_text(pn.aes(x=0.25,y=0.95,label='100*auroc'),size=9,data=auroc_skew,format_string='AUROC={:.1f}%') + 
    pn.facet_wrap('~skew') + pn.geom_step() + pn.ggtitle('A') + 
    pn.theme(legend_position='none'))
# gg_save('gg_roc_skew.png', dir_figures, gg_roc_skew, 9, 3)

# (ii) Empirical score distribution and skew
gg_dist_skew = (pn.ggplot(dist_skew,pn.aes(x='s',fill='y.astype(str)')) + 
    pn.theme_bw() + pn.labs(x='Scores',y='Frequency') + 
    pn.theme(legend_position=(0.3, -0.03), legend_direction='horizontal',legend_box_margin=0) + 
    pn.facet_wrap('~skew') + pn.ggtitle('B') + 
    pn.geom_histogram(position='identity',color='black',alpha=0.5,bins=25) + 
    pn.scale_fill_discrete(name='Label'))
# gg_save('gg_dist_skew.png', dir_figures, gg_dist_skew, 9, 3.25)
# Combine both plots
g1 = pw.load_ggplot(gg_roc_skew, figsize=(9,3))
g2 = pw.load_ggplot(gg_dist_skew, figsize=(9,3))
gg_roc_dist_skew = pw.vstack(g2, g1, margin=0.25, adjust=False)
grid_save('gg_roc_dist_skew.png', dir_figures, gg_roc_dist_skew)


##############################################
# --- (1) CHECK EMPIRICAL TO GROUNDTRUTH --- #

nsim, n_test = 250, 100
mu, p = 1, 0.5
enc_dgp = dgp_bin(mu=mu, p=p, thresh=mu)
holder_auc = np.zeros(nsim)
holder_roc = []
for i in range(nsim):
    test_i = enc_dgp.dgp_bin(n=n_test, seed=i)
    # (i) Empirical AUROC
    holder_auc[i] = auc_rank(test_i['y'], test_i['s'])
    # (ii) Empirical ROC curve
    tmp_roc = emp_roc_curve(test_i['y'], test_i['s']).assign(sim=i)
    holder_roc.append(tmp_roc)
emp_roc = pd.concat(holder_roc).reset_index(drop=True)
gt_roc = enc_dgp.roc_curve()
df_roc = pd.concat(objs=[emp_roc, gt_roc.assign(sim=-1)])
df_roc = df_roc.assign(tt=lambda x: np.where(x['sim']==-1,'Ground Truth','Simulation'))
df_auc = pd.DataFrame({'auc':holder_auc,'gt':enc_dgp.auroc_oracle})

# (i) Empirical ROC to actual
gg_roc_gt = (pn.ggplot(df_roc,pn.aes(x='1-spec',y='sens',size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.theme_bw() + pn.labs(x='1-Specificity',y='Sensitivity') + 
    pn.ggtitle('ROC Curve (Ground Truth vs Simulation)') + 
    pn.geom_step() + 
    pn.theme(legend_position=(0.7,0.3)) + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.1]) + 
    pn.scale_size_manual(name=' ',values=[2,0.5]))
gg_save('gg_roc_gt.png', dir_figures, gg_roc_gt, 5, 3.5)

# (ii) Empirical AUROC to actual
gg_auc_gt = (pn.ggplot(df_auc,pn.aes(x='auc')) + pn.theme_bw() + 
    pn.geom_histogram(fill='grey',alpha=0.5,color='red',bins=20) + 
    pn.geom_vline(pn.aes(xintercept='gt'),color='black') + 
    pn.labs(x='AUROC',y='Frequency') + 
    pn.ggtitle('Black lines show ground truth AUROC'))
gg_save('gg_auc_gt.png', dir_figures, gg_auc_gt, 5, 3.5)

# (iii) Operating threshold on x-axis
sens_target = 0.5
df_emp_thresh = df_roc.melt(['sim','thresh','tt'], None, 'msr', 'val')
df_emp_thresh['msr'] = df_emp_thresh['msr'].map(di_msr)

# (a) What threshold is chosen to get to sens_target?
thresh_sens = df_emp_thresh.query('sim>=0 & msr=="Sensitivity"').drop(columns=['msr','tt'])
thresh_spec = df_emp_thresh.query('sim>=0 & msr=="Specificity"').drop(columns=['msr','tt'])
thresh_chosen = interp_df(thresh_sens, 'val', 'thresh', sens_target, 'sim')
thresh_spec = thresh_spec.merge(thresh_chosen)
thresh_spec = thresh_spec.groupby('sim').apply(lambda x: interp_df(x, 'thresh', 'val', x['thresh_interp'].values[0], 'sim'))
thresh_spec.reset_index(drop=True, inplace=True)
thresh_spec = thresh_spec.rename(columns={'val_interp':'spec'}).assign(msr=di_msr['spec'])
thresh_chosen = thresh_chosen.rename(columns={'thresh_interp':'thresh'}).assign(msr=di_msr['sens'])
thresh_spec = thresh_spec.merge(thresh_chosen.drop(columns='msr'))
# Long-run sensitivity
thresh_chosen = thresh_chosen.assign(sens=lambda x: 1-norm(loc=mu).cdf(x['thresh']))

# (a) Intersection of thresholds to 50% sensitivity
gtit = 'A: Empirical threshold choice for %i%% sensivitiy' % (100*sens_target)
gg_roc_thresh = (pn.ggplot(df_emp_thresh,pn.aes(x='val',y='thresh')) + 
    pn.theme_bw() + pn.labs(y='Operating threshold',x='Performance target') + 
    pn.ggtitle(gtit) + pn.facet_wrap('~msr') + 
    pn.geom_step(pn.aes(size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.geom_point(pn.aes(x=sens_target,y='thresh'),size=0.5,alpha=0.5,color='red',data=thresh_chosen) + 
    pn.geom_point(pn.aes(x='spec',y='thresh'),size=0.5,alpha=0.5,color='blue',data=thresh_spec) + 
    # pn.theme(legend_position=(0.5,-0.05),legend_direction='horizontal') + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.2]) + 
    pn.scale_size_manual(name=' ',values=[1.5,0.5]))
# gg_save('gg_roc_thresh.png', dir_figures, gg_roc_thresh, 9, 3.5)

# (b) Distribution of thresholds
gg_dist_thresh = (pn.ggplot(thresh_chosen, pn.aes(x='thresh')) + pn.theme_bw() + 
    pn.geom_histogram(fill='grey',alpha=0.5,color='red',bins=20) + 
    pn.labs(x='Threshold',y='Frequency',title='B: Threshold distribution'))
# gg_save('gg_dist_thresh.png', dir_figures, gg_dist_thresh, 4.5, 3.5)

# (c) Distribution of long-run sensitivity
rc('text', usetex=True)
gg_dist_sens = (pn.ggplot(thresh_chosen, pn.aes(x='sens')) + pn.theme_bw() + 
    pn.geom_histogram(fill='grey',alpha=0.5,color='green',bins=20) + 
    pn.geom_vline(xintercept=sens_target) + 
    pn.ggtitle('C: Expected sensitivity') + 
    pn.labs(x='$P(x_{i1} \geq \hat{t})$',y='Frequency'))
# gg_save('gg_dist_sens.png', dir_figures, gg_dist_sens, 4.5, 3.5)
rc('text', usetex=False)

g1 = pw.load_ggplot(gg_roc_thresh, figsize=(9,3))
g2 = pw.load_ggplot(gg_dist_thresh, figsize=(4.5,3.5))
g3 = pw.load_ggplot(gg_dist_sens, figsize=(4.5,3.5))

gg_roc_process = g1 / (g2 | g3)
grid_save('gg_roc_process.png', dir_figures, gg_roc_process)


###################################
# --- (2) SENSITIVITY TARGET  --- #

# Tidy tt labels
di_tt = {'oracle':'Oracle', 'emp':'Empirical', 'trial':'Trial'}
nsim = 250
mu, p = 1, 0.5
n_test, n_trial = 200, 200
sens = 0.5
enc_dgp = dgp_bin(mu=mu, p=p, sens=sens)

holder = []
for i in range(nsim):
    test_i = enc_dgp.dgp_bin(n=n_test, seed=i)
    enc_dgp.learn_threshold(y=test_i['y'], s=test_i['s'])
    enc_dgp.create_df()
    trial_i = enc_dgp.dgp_bin(n=n_trial, seed=i+1)
    res_trial = enc_dgp.get_tptn(trial_i['y'], trial_i['s'])
    res_trial = res_trial.assign(tt='trial').melt('tt',None,'msr','val')
    tmp_df = pd.concat(objs=[enc_dgp.df_rv, res_trial], axis=0)
    tmp_df.insert(0, 'sim', i)
    holder.append(tmp_df)
res_tfpr = pd.concat(holder).reset_index(drop=True)
res_tfpr['msr'] = res_tfpr['msr'].map(di_msr)
res_tfpr['tt'] = res_tfpr['tt'].map(di_tt)
res_tfpr_wide = res_tfpr.pivot_table('val',['sim','msr'],'tt').reset_index()
res_tfpr_wide = res_tfpr_wide.melt(['sim','msr',di_tt['oracle']])
res_tfpr_wide = res_tfpr_wide.dropna().reset_index(drop=True)

# (i) Oracle vs OOS vs Empirical
gg_sens_scatter = (pn.ggplot(res_tfpr_wide, pn.aes(x='value',y='Oracle',color='tt')) + 
    pn.theme_bw() + pn.scale_color_discrete(name=' ') + 
    pn.labs(x='Observed',y='Oracle') + 
    pn.geom_point() + pn.geom_abline(slope=1,intercept=0,linetype='--') + 
    pn.facet_wrap('~msr'))
gg_sens_scatter.save(os.path.join(dir_figures,'gg_sens_scatter.png'),height=4,width=12)

# (ii) Histogram distribution
gg_sens_hist = (pn.ggplot(res_tfpr, pn.aes(x='val',fill='tt')) + 
    pn.theme_bw() + pn.scale_fill_discrete(name=' ') + 
    pn.labs(x='Value',y='Frequency') + 
    pn.geom_histogram(alpha=0.5,color='grey',bins=30,position='identity') + 
    pn.facet_wrap('~msr',scales='free_x'))
gg_sens_hist.save(os.path.join(dir_figures,'gg_sens_hist.png'),height=4,width=12)


##################################
# --- (3) POWER SIMULATIONS  --- #

mu, p = 1, 0.5
n_test, n_trial = 400, 400
n_bs, alpha = 1000, 0.05
sens = 0.5
nsim = 2500

# TEST OVER RANGES OF SENSITIVITY/SPECIFCITY/SAMPLE SIZE
lst_sens = [0.5, 0.6, 0.7]
lst_n_test = [250, 500, 1000]
lst_n_trial = [250, 500, 1000]
lst_margin = [0.025, 0.050, 0.075]
lst_hp = [lst_sens, lst_n_test, lst_n_trial, lst_margin]
n_hp = np.prod([len(l) for l in lst_hp])

stime = time()
holder_power = []
j = 0
for sens in lst_sens:
    enc_dgp = dgp_bin(mu=mu, p=p, sens=sens, alpha=alpha)
    for n_trial in lst_n_trial:
        for n_test in lst_n_test:
            for margin in lst_margin:
                j += 1
                for i in range(nsim):
                    a = None
                    # (i) Generate data (going to to use oracle n1, n0)
                    dat_test = enc_dgp.dgp_bin(n=n_test, seed=i)
                    dat_trial = enc_dgp.dgp_bin(n=n_trial, seed=i+1)
                    n0_trial, n1_trial = dat_trial['y'].value_counts().sort_index()

                    # (ii) Learn threshold on sensitivity target and estimate power
                    enc_dgp.learn_threshold(y=dat_test['y'], s=dat_test['s'])
                    enc_dgp.create_df()
                    null_sens = enc_dgp.sens_emp - margin
                    null_spec = enc_dgp.spec_emp - margin
                    enc_dgp.set_null_hypotheis(null_sens, null_spec)
                    enc_dgp.run_power(n1_trial, n0_trial, method=None, seed=i, n_bs=n_bs)
                    
                    # (iii) Run hypothesis test
                    vec_msr = ['sens', 'spec']
                    vec_act = enc_dgp.get_tptn(y=dat_trial['y'],s=dat_trial['s']).values.flatten()
                    vec_null = np.array([null_sens, null_spec])
                    vec_n = np.array([n1_trial, n0_trial])    
                    res_trial = enc_dgp.binom_stat(p_act=vec_act, p_null=vec_null, n=vec_n)
                    res_trial.insert(0, 'msr', vec_msr)

                    # (iv) Store
                    res_i = enc_dgp.df_power.merge(res_trial)
                    res_i = res_i.merge(enc_dgp.df_rv.pivot('msr','tt','val').reset_index(),'left')
                    # Add on hyperparameters
                    res_i = res_i.assign(sens=sens, n_trial=n_trial, n_test=n_test, margin=margin, sim=i)
                    holder_power.append(res_i)

                    # Run-time update
                    if (i + 1) % 100 == 0:
                        dtime = time() - stime
                        n_left = (n_hp-j)*nsim + (nsim-(i+1))
                        n_run = (j-1)*nsim + (i+1)
                        srate = n_run/dtime
                        seta = n_left / srate
                        meta = seta / 60
                        print('Iteration %i/%i for param %i of %i (ETA: %i seconds, %0.1f minutes)' % (i+1, nsim, j, n_hp, seta, meta))
# Merge
res_power = pd.concat(holder_power).reset_index(drop=True)
res_power.to_csv('res_power.csv',index=False)
# res_power = pd.read_csv('res_power.csv')
res_power.drop(columns=['h0','z','pval','emp','oracle'], inplace=True, errors='ignore')
cn_gg = ['msr', 'sens', 'n_trial', 'n_test', 'margin', 'method']
assert np.all(res_power.groupby(cn_gg).size() == nsim)
res_power['msr'] = res_power['msr'].map(di_msr)
res_power['method'] = res_power['method'].map(di_method)
# Calculate the coverage
res_power = res_power.assign(cover=lambda x: (x['power']>x['lb']) & (x['power']<x['ub']))
res_cover = res_power.groupby(cn_gg)['cover'].sum().reset_index()
res_cover = get_CI(res_cover, 'cover', nsim)
res_cover = res_cover.assign(cover=lambda x: x['cover']/nsim)
# Compare how expected power lines up to actual (independent of method)
cn_val = ['power','reject']
res_calib = res_power.groupby(cn_gg)[cn_val].sum().reset_index()
res_calib = get_CI(res_calib, 'reject', nsim)
res_calib[cn_val] = res_calib[cn_val] / nsim
res_calib = res_calib.drop(columns='method').drop_duplicates()

# (i) Coverage plot
posd = pn.position_dodge(0.5)
gg_cover_bs = (pn.ggplot(res_cover, pn.aes(x='msr',y='cover',color='method',shape='sens.astype(str)')) + 
    pn.theme_bw() + pn.labs(y='Coverage') + 
    pn.ggtitle('Dashed line shows coverage target') + 
    pn.geom_hline(yintercept=1-alpha,linetype='--') + 
    pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.scale_color_discrete(name='Bootstrap method') + 
    pn.scale_shape_discrete(name='Sensitivity target') + 
    pn.theme(legend_position=(0.5,-0.05),legend_direction='horizontal',legend_box='horizontal',axis_title_x=pn.element_blank(), axis_text_x=pn.element_text(angle=90)) + 
    pn.facet_grid('margin~n_trial+n_test',labeller=pn.label_both))
gg_cover_bs.save(os.path.join(dir_figures,'gg_cover_bs.png'),height=6,width=16)


# (ii) Calibration plot (how well does power formula actually approximate)
gtit = 'Predicted rejection rate is average of power'
gg_power_calib = (pn.ggplot(res_calib, pn.aes(x='power',y='reject',color='msr',shape='sens.astype(str)')) + 
    pn.theme_bw() + pn.geom_point() + pn.ggtitle(gtit) + 
    pn.geom_abline(slope=1,intercept=0,linetype='--') + 
    pn.labs(y='Rejection rate (actual)',x='Rejection rate (predicted)') + 
    pn.facet_grid('margin~n_trial+n_test',labeller=pn.label_both) + 
    pn.scale_color_discrete(name='Measure') + 
    pn.scale_shape_discrete(name='Sensitivity target') + 
    pn.scale_x_continuous(limits=[0,1]) + 
    pn.scale_y_continuous(limits=[0,1]) + 
    pn.theme(legend_position=(0.5,-0.05),legend_direction='horizontal',legend_box='horizontal', axis_text_x=pn.element_text(angle=90)))
gg_power_calib.save(os.path.join(dir_figures,'gg_power_calib.png'),height=5,width=15)


####################################
# --- (4) COMPARATIVE STATICS  --- #

method = 'quantile'
lst_margin = np.arange(0.01, 0.31, 0.01)
lst_n_trial = [50, 100, 250, 1000]
lst_n_test = [250, 500, 1000]
# Use a single instance
enc_dgp = dgp_bin(mu=1, p=0.5, sens=0.5, alpha=0.05)
holder_static = []
for n_test in lst_n_test:
    dat_test = enc_dgp.dgp_bin(n=n_test, seed=1)
    enc_dgp.learn_threshold(y=dat_test['y'], s=dat_test['s'])
    for margin in lst_margin:
        null_sens = enc_dgp.sens_emp - margin
        null_spec = enc_dgp.spec_emp - margin
        for n_trial in lst_n_trial:
            n1_trial, n0_trial = n_trial, n_trial
            enc_dgp.set_null_hypotheis(null_sens, null_spec)
            enc_dgp.run_power(n1_trial, n0_trial, method=method, seed=1, n_bs=1000)
            tmp_df = enc_dgp.df_power.assign(margin=margin, n_test=n_test)
            holder_static.append(tmp_df)
# Merge and plot
res_static = pd.concat(holder_static).reset_index(drop=True)
cn_ns = ['n_trial', 'n_test']
res_static[cn_ns] = res_static[cn_ns].apply(pd.Categorical)
res_static['msr'] = res_static['msr'].map(di_msr)

# Plot
height = 3 * len(res_static['msr'].unique())
width = 3 * len(res_static['n_trial'].unique())
gg_power_margin = (pn.ggplot(res_static, pn.aes(x='margin',fill='n_test')) + 
    pn.theme_bw() + 
    pn.scale_fill_discrete(name='# of test samples') + 
    pn.facet_grid('msr~n_trial',labeller=pn.labeller(n_trial=pn.label_both)) + 
    pn.ggtitle('95% CI based on quantile approach') +
    pn.labs(x='Null hypothesis margin',y='Power estimate (CI)') + 
    pn.geom_ribbon(pn.aes(ymin='lb',ymax='ub'),color='black',alpha=0.5))
gg_power_margin.save(os.path.join(dir_figures,'gg_power_margin.png'),height=height,width=width)


############################
# --- (5) ROC & POWER  --- #


# 3. (i) Oracle ROC curve and power of the trial , (ii) CI around some randomly chosen points -->

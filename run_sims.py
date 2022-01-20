import os
import numpy as np
import pandas as pd
import plotnine as pn
from time import time
from funs_support import makeifnot
from funs_stats import dgp_bin, emp_roc_curve, auc_rank, get_CI
dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
makeifnot(dir_figures)

# Tidy up the different labels
di_msr = {'sens':'Sensitivity', 'spec':'Specificity', 'thresh':'Threshold'}
di_method = {'basic':'Basic', 'quantile':'Quantile', 'expanded':'Quantile (t-adj)', 'studentized':'Studentized', 'bca':'BCa'}

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
gg_roc_gt.save(os.path.join(dir_figures,'gg_roc_gt.png'),height=4,width=6)

# (ii) Empirical AUROC to actual
gg_auc_gt = (pn.ggplot(df_auc,pn.aes(x='auc')) + pn.theme_bw() + 
    pn.geom_histogram(fill='grey',alpha=0.5,color='red',bins=20) + 
    pn.geom_vline(pn.aes(xintercept='gt'),color='black') + 
    pn.labs(x='AUROC',y='Frequency') + 
    pn.ggtitle('Black lines show ground truth AUROC'))
gg_auc_gt.save(os.path.join(dir_figures,'gg_auc_gt.png'),height=4,width=6)

# (iii) Operating threshold on x-axis
df_emp_thresh = df_roc.melt(['sim','thresh','tt'], None, 'msr', 'val')
df_emp_thresh['msr'] = df_emp_thresh['msr'].map(di_msr)

gg_roc_tresh = (pn.ggplot(df_emp_thresh,pn.aes(x='thresh',y='val',size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.theme_bw() + pn.labs(x='Operating threshold',y='Value') + 
    pn.ggtitle('Operating threshold curve underlying AUROC') + 
    pn.geom_step() + pn.facet_wrap('~msr') + 
    pn.theme(legend_position=(0.5,-0.01),legend_direction='horizontal') + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.1]) + 
    pn.scale_size_manual(name=' ',values=[1.5,0.5]))
gg_roc_tresh.save(os.path.join(dir_figures,'gg_roc_tresh.png'),height=4,width=10)


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
nsim = 2000

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
                    res_i = res_i.assign(sens=sens, n_trial=n_trial, n_test=n_test, margin=margin)
                    holder_power.append(res_i)

                    # Run-time update
                    if (i + 1) % 100 == 0:
                        dtime = time() - stime
                        n_left = (n_hp-j)*nsim + (nsim-(i+1))
                        n_run = (j-1)*nsim + (i+1)
                        srate = n_run/dtime
                        seta = n_left / srate
                        meta = seta / 60
                        print('Iteration %i/%i for param %i of %i (ETA: %i seconds, %0.1f minutes)' % (i+1, nsim, j, n_hp seta, meta))
# Merge
res_power = pd.concat(holder_power).reset_index(drop=True)
res_power = res_power.drop(columns=['n','h0','z','pval'])
# (i) Calculate the coverage
res_power = res_power.assign(cover=lambda x: (x['power']>x['lb']) & (x['power']<x['ub']))
res_cover = res_power.groupby(['msr'])['cover'].sum().reset_index()
res_cover = get_CI(res_cover, 'cover', nsim)
print(res_cover)
# (ii) Compare how expected power lines up to actual
res_calib = res_power.groupby('msr')[['power','reject']].sum().reset_index()
res_calib = get_CI(res_calib, 'reject', nsim).assign(power=lambda x: x['power']/nsim)



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



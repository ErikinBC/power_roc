# Script for real-world data
import os
import numpy as np
import pandas as pd
import plotnine as pn
from funs_support import makeifnot, round_up
from funs_stats import dgp_bin, emp_roc_curve
dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
makeifnot(dir_figures)
dir_data = os.path.join(dir_base, 'data')
makeifnot(dir_data)

# Set up shared parameters
nsim = 250
alpha = 0.05
di_msr = {'sens':'Sensitivity', 'spec':'Specificity', 'thresh':'Threshold'}
di_ds = {'test':'Test','hsk':'HSK', 'stan':'Stanford', 'iowa':'Iowa'}
di_metric = {'pval':'P-value', 'stat':'Statistic', 'z':'Z-score'}
lst_msr = ['sens','spec']


#########################
# --- (1) LOAD DATA --- #

dat_test = pd.read_csv(os.path.join(dir_data, 'ValPredictions_withTarget.csv')) 
dat_test = dat_test.rename(columns={'mod_pred_ep30':'s'}).assign(ds='test')
dat_hsk = pd.read_csv(os.path.join(dir_data, 'SilentTrialPredictions_withTarget.csv'))
dat_hsk = dat_hsk.rename(columns={'st_pred_mod_ep30':'s'}).assign(ds='hsk')
dat_stan = pd.read_csv(os.path.join(dir_data, 'StanfordPredictions_withTarget.csv'))
dat_stan = dat_stan.rename(columns={'stan_pred_mod_ep30':'s'}).assign(ds='stan')
dat_iowa = pd.read_csv(os.path.join(dir_data, 'UIowaPredictions_withTarget.csv'))
dat_iowa = dat_iowa.rename(columns={'ui_pred_mod_ep30':'s'}).assign(ds='iowa')
df_scores = pd.concat(objs=[dat_test, dat_hsk, dat_stan, dat_iowa])
df_scores = df_scores.rename(columns={'target01':'y'}).drop(columns=['Unnamed: 0'],errors='ignore')
df_scores = df_scores.assign(logits=lambda x: np.log(x['s']/(1-x['s'])))
# Use validation as the test calibration
ds_test = 'test'
dat_test = df_scores.query('ds == @ds_test').reset_index(drop=True)
dat_trial = df_scores.query('ds !=@ds_test').reset_index(drop=True)
# Get the n counts
df_n_ds = df_scores.groupby(['ds','y']).size().reset_index().rename(columns={0:'n'})
print(df_n_ds.sort_values('y'))


###############################
# --- (3) ROC CALIBRATION --- #

# (i) Examine empirical ROC curve for HSK data
roc_test_point = emp_roc_curve(dat_test['y'], dat_test['s'])
holder_roc_test = []
for i in range(nsim):
    dat_test_bs = dat_test.sample(frac=1,replace=True,random_state=i)
    roc_test_bs = emp_roc_curve(dat_test_bs['y'], dat_test_bs['s'])
    holder_roc_test.append(roc_test_bs.assign(sim=i))
roc_test_bs = pd.concat(holder_roc_test).reset_index(drop=True)
df_roc_test = pd.concat(objs=[roc_test_bs, roc_test_point.assign(sim=-1)])
df_roc_test = df_roc_test.assign(tt=lambda x: np.where(x['sim']==-1,'Point Estimate','Simulation'))
df_roc_test = df_roc_test.query('thresh >= 0 and thresh <= 1')

# (i) Range of ROC curves 
gg_roc_test = (pn.ggplot(df_roc_test,pn.aes(x='1-spec',y='sens',size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.theme_bw() + pn.labs(x='1-Specificity',y='Sensitivity') + 
    pn.ggtitle('Empirical ROC curve on HSK validation data') + 
    pn.geom_step() + 
    pn.theme(legend_position=(0.7,0.3)) + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.1]) + 
    pn.scale_size_manual(name=' ',values=[2,0.5]))
gg_roc_test.save(os.path.join(dir_figures,'gg_roc_test.png'),height=3.5,width=4.5)


# (ii) Range of operating threshold curves
df_thresh_test = df_roc_test.melt(['sim','thresh','tt'], None, 'msr', 'val')
df_thresh_test['msr'] = df_thresh_test['msr'].map(di_msr)

gg_thresh_test = (pn.ggplot(df_thresh_test,pn.aes(x='thresh',y='val',size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.theme_bw() + pn.labs(x='Operating threshold',y='Value') + 
    pn.ggtitle('Operating threshold curve underlying AUROC') + 
    pn.geom_step() + pn.facet_wrap('~msr') + 
    pn.theme(legend_position=(0.5,-0.06),legend_direction='horizontal') + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.1]) + 
    pn.scale_size_manual(name=' ',values=[1.5,0.5]))
gg_thresh_test.save(os.path.join(dir_figures,'gg_thresh_test.png'),height=3.5,width=9)

# (iii) How normally distributed are logits?
gg_logits_ds = (pn.ggplot(df_scores,pn.aes(x='logits',fill='y.astype(str)')) + 
    pn.theme_bw() + pn.labs(x='Logits',y='Frequency') + 
    pn.theme(subplots_adjust={'hspace': 0.25, 'wspace':0.25}) + 
    pn.facet_wrap('~ds',labeller=pn.labeller(ds=di_ds),ncol=2,scales='free') + 
    pn.geom_histogram(position='identity',color='black',alpha=0.5,bins=25) + 
    pn.scale_fill_discrete(name='Label'))
gg_logits_ds.save(os.path.join(dir_figures,'gg_logits_ds.png'),height=8,width=10)

# (iv) ROC and uncertainty by dataset
roc_ds_point = df_scores.groupby('ds').apply(lambda x: emp_roc_curve(x['y'], x['s'])).reset_index()
holder_roc_ds = []
for i in range(nsim):
    dat_ds_bs = df_scores.groupby('ds').sample(frac=1,replace=True,random_state=i)
    roc_ds_bs = dat_ds_bs.groupby('ds').apply(lambda x: emp_roc_curve(x['y'], x['s']))
    roc_ds_bs = roc_ds_bs.reset_index().assign(sim=i)
    holder_roc_ds.append(roc_ds_bs)
roc_ds_bs = pd.concat(holder_roc_ds).reset_index(drop=True)
df_roc_ds = pd.concat(objs=[roc_ds_bs, roc_ds_point.assign(sim=-1)])
df_roc_ds = df_roc_ds.assign(tt=lambda x: np.where(x['sim']==-1,'Point Estimate','Simulation'))
df_roc_ds = df_roc_ds.query('thresh >= 0 and thresh <= 1').drop(columns='level_1')
df_roc_ds['ds'] = df_roc_ds['ds'].map(di_ds)

gg_roc_ds = (pn.ggplot(df_roc_ds,pn.aes(x='1-spec',y='sens',size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.theme_bw() + pn.labs(x='1-Specificity',y='Sensitivity') + 
    pn.ggtitle('ROC curves on all datasets') + 
    pn.geom_step() + pn.facet_wrap('~ds') + 
    pn.theme(legend_position=(0.7,0.3)) + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.1]) + 
    pn.scale_size_manual(name=' ',values=[2,0.5]))
gg_roc_ds.save(os.path.join(dir_figures,'gg_roc_ds.png'),height=6,width=8)

# (v) Threshold range by dataset
df_thresh_ds = df_roc_ds.melt(['sim','thresh','tt','ds'], None, 'msr', 'val')
df_thresh_ds['msr'] = df_thresh_ds['msr'].map(di_msr)

gg_thresh_ds = (pn.ggplot(df_thresh_ds,pn.aes(x='thresh',y='val',size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.theme_bw() + pn.labs(x='Operating threshold',y='Value') + 
    pn.ggtitle('Operating threshold curve underlying AUROC') + 
    pn.geom_step() + pn.facet_grid('ds~msr') + 
    pn.theme(legend_position=(0.5,-0.0),legend_direction='horizontal') + 
    pn.scale_color_manual(name=' ',values=['black','grey']) + 
    pn.scale_alpha_manual(name=' ',values=[1,0.1]) + 
    pn.scale_size_manual(name=' ',values=[1.5,0.5]))
gg_thresh_ds.save(os.path.join(dir_figures,'gg_thresh_ds.png'),height=12,width=8)


##################################
# --- (3) POWER EXPECTATIONS --- #

# Threshold was already determined
thresh_val = 0.08497446
enc_test = dgp_bin(thresh=thresh_val, alpha=alpha, mu=1,p=0.5)
enc_test.learn_threshold(y=dat_test['y'], s=dat_test['s'])
enc_test.create_df()
# Empirical sensitivity is ~61% and empirical specificity is ~94%
print(enc_test.df_rv.query('tt == "emp"'))

# Margin range to test over
lst_margin = np.arange(0.01, 0.31, 0.01).round(2)

# Maximum sample size calculation
n_lb = 5
n0_max, n1_max = df_n_ds.groupby(['y'])['n'].max().sort_index().values.flat
dat_n_max = pd.DataFrame({'msr':lst_msr,'n':[n1_max, n0_max]})
n_prop_seq = np.array([1/3, 2/3, 3/3])
n1_trial_seq = (round_up(n1_max, n_lb)*n_prop_seq).astype(int)
n0_trial_seq = (round_up(n0_max, n_lb)*n_prop_seq).astype(int)

null_sens = enc_test.sens_emp - lst_margin
null_spec = enc_test.spec_emp - lst_margin
tmp1 = pd.DataFrame({'msr':'sens', 'margin':lst_margin, 'h0':null_sens})
tmp2 = pd.DataFrame({'msr':'spec', 'margin':lst_margin, 'h0':null_spec})
df_margin = pd.concat(objs=[tmp1, tmp2])

enc_test.set_null_hypotheis(null_sens, null_spec)
holder_n = []
for n1, n0 in zip(n1_trial_seq, n0_trial_seq):
    n1, n0 = int(n1), int(n0)
    enc_test.run_power('both', n1_trial=n1, n0_trial=n0, method='quantile')
    tmp_df = enc_test.df_power.drop(columns=['method','power'])
    tmp_df= tmp_df.merge(df_margin)
    holder_n.append(tmp_df)
res_test_margin = pd.concat(holder_n).reset_index(drop=True)
res_test_margin['n_trial'] = pd.Categorical(res_test_margin['n_trial'])

gg_test_margin = (pn.ggplot(res_test_margin, pn.aes(x='margin',fill='n_trial')) + 
    pn.theme_bw() + 
    pn.scale_fill_discrete(name='# trial') + 
    pn.facet_wrap('~msr',labeller=pn.labeller(msr=di_msr)) + 
    pn.ggtitle('95% CI based on quantile approach') +
    pn.labs(x='Null hypothesis margin',y='Power estimate (CI)') + 
    pn.geom_ribbon(pn.aes(ymin='lb',ymax='ub'),color='black',alpha=0.5))
gg_test_margin.save(os.path.join(dir_figures,'gg_test_margin.png'),height=4,width=9)


#############################
# --- (4) TRIAL RESULTS --- #

# Set the margin
margin_sens = 0.25
margin_spec = 0.05
null_sens = enc_test.sens_emp - margin_sens
null_spec = enc_test.spec_emp - margin_spec
vec_target = np.array([enc_test.sens_emp, enc_test.spec_emp])  
vec_null = np.array([null_sens, null_spec])
enc_test.set_null_hypotheis(null_sens, null_spec)

# Empirical CI with median
lst_alpha = [alpha/2,0.5,1-alpha/2]
di_alpha = dict(zip(lst_alpha, ['lb','med','ub']))

# Simulated the range of sensitivity/specificities through random patient order
nsim = 100
lst_ds = ['stan', 'iowa', 'hsk']
holder_ds = []
for ds in lst_ds:
    print('--- Dataset: %s ---' % di_ds[ds])
    dat_ds = dat_trial.query('ds == @ds')
    n0_ds, n1_ds = dat_ds['y'].value_counts().sort_index().values
    n0_ds, n1_ds = int(n0_ds), int(n1_ds)
    # Run the statistical test over a simulated silent trial
    lst_frac = np.arange(0.25, 1.01, 0.05).round(2)
    for j in range(nsim):
        if (j+1) % 10 == 0:
            print('Simulation %i of %i' % (j+1, nsim))
        # Shuffle the rows of the data
        dat_ds_rand = dat_ds.sample(frac=1,random_state=j,replace=False)
        # Index the patients (groupby label to preserve balance)
        dat_ds_rand['idx'] = dat_ds_rand.groupby('y').cumcount()+1
        dat_ds_rand = dat_ds_rand.reset_index(drop=True).rename_axis('idt').reset_index()
        dat_ds_rand = dat_ds_rand.merge(df_n_ds).assign(idx=lambda x: x['idx']/x['n'])
        for frac in lst_frac:
            dat_ds_frac = dat_ds_rand.query('idx <= @frac')
            n0_frac, n1_frac = dat_ds_frac['y'].value_counts().sort_index().values
            n0_frac, n1_frac = int(n0_frac), int(n1_frac)
            # (i) Calculate expected power at this sample size
            enc_test.run_power('both', n1_trial=n1_frac, n0_trial=n0_frac, method='quantile')
            tmp_power_frac = enc_test.df_power.drop(columns=['power','method'])
            # (ii) Calculate actual test statistics
            p_act_frac = enc_test.get_tptn(dat_ds_frac['y'], dat_ds_frac['s'])
            p_act_frac = p_act_frac.values.flatten()
            n_vec_frac = np.array([n1_frac, n0_frac])
            tmp_frac_stat = enc_test.binom_stat(p_act=p_act_frac, p_null=vec_null, n=n_vec_frac)
            tmp_frac_stat = tmp_frac_stat.assign(msr=lst_msr).drop(columns='reject')
            tmp_frac_stat = tmp_frac_stat.assign(stat=p_act_frac, frac=frac, sim=j, ds=ds)
            # (iii) Merge and store
            tmp_frac_stat = tmp_frac_stat.merge(tmp_power_frac)
            holder_ds.append(tmp_frac_stat)
# Merge and clean
res_sim_trial = pd.concat(holder_ds).reset_index(drop=True)
cn_gg = ['ds','msr','frac','n_trial','lb','ub','h0']
cn_val = list(res_sim_trial.columns.drop(cn_gg))
assert np.all(res_sim_trial.groupby(cn_gg).size() == nsim)
# Calculate the 95% range across fractions
res_ci_trial = res_sim_trial.groupby(cn_gg)[cn_val].quantile(lst_alpha).reset_index()
res_ci_trial.rename(columns={'level_%s'%len(cn_gg):'tt'},inplace=True)
# To avoid confusion, make lower and upper bound estimates of power into type-2 errors
res_ci_trial = res_ci_trial.assign(beta_lb=lambda x: 1-x['ub'], beta_ub=lambda x: 1-x['lb'])
res_ci_trial.drop(columns=['lb','ub'],inplace=True)
cn_gg = res_ci_trial.columns.drop(cn_val)
res_ci_trial = res_ci_trial.melt(cn_gg,cn_val,'metric','val')
# Get empirical CI around simulated performance
res_ci_trial = res_ci_trial.pivot_table('val',list(cn_gg.drop('tt'))+['metric'],'tt')
res_ci_trial = res_ci_trial.rename(columns=di_alpha).reset_index()
# Remove sim and z-score for now
res_ci_trial = res_ci_trial[~res_ci_trial['metric'].isin(['z','sim'])].reset_index(drop=True)
res_ci_trial['metric'] = res_ci_trial['metric'].map(di_metric)

# Horizontal lines to show null
df_null = pd.DataFrame({'msr':lst_msr, 'null':vec_null, 'target':vec_target})
df_null = df_null.melt('msr',None,'calib')

# (i) Plot trial results
gtit = 'Horizontal lines shows null hypothesis\nLine range shows values of simulations'
# Put beta as a type of metric
grp1 = ['ds','msr','frac','n_trial']
tmp1 = res_ci_trial.melt(grp1,['beta_lb','beta_ub'])
tmp1 = tmp1.drop_duplicates().assign(metric='Type-II error')
tmp1['tt'] = tmp1['tt'].str.replace('beta_','',regex=False)
tmp1 = tmp1.pivot_table('value',grp1+['metric'],'tt').reset_index()
tmp2 = res_ci_trial.drop(columns=['beta_lb','beta_ub'])
tmp_df = pd.concat(objs=[tmp1, tmp2],axis=0).reset_index(drop=True)

colz = ["#F8766D","#619CFF","#00BA38"]
gg_trial_sim = (pn.ggplot(tmp_df,pn.aes(x='frac',y='med',color='metric',fill='metric')) + 
    pn.geom_line() + 
    pn.geom_ribbon(pn.aes(ymin='lb',ymax='ub'),alpha=0.5,color='black') + 
    pn.theme_bw() + pn.ggtitle(gtit) + 
    pn.labs(x='Fraction of patients',y='Value') + 
    pn.theme(legend_position=(0.5,0.0), legend_direction='horizontal', legend_box='horizontal') +
    pn.scale_color_manual(name='Measure',values=colz) + 
    pn.scale_fill_manual(name='Measure',values=colz) + 
    pn.geom_hline(pn.aes(yintercept='value',linetype='calib'),data=df_null) + 
    pn.scale_linetype_discrete(name='Target', labels=['Null (H0)','Calibrated']) + 
    pn.facet_grid('msr~ds',labeller=pn.labeller(msr=di_msr, ds=di_ds)))
gg_trial_sim.save(os.path.join(dir_figures,'gg_trial_sim.png'),height=7,width=12)



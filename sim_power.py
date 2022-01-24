import os
from time import time
import numpy as np
import pandas as pd
import plotnine as pn
from funs_support import makeifnot
from funs_stats import dgp_bin, get_CI

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
makeifnot(dir_figures)

di_msr = {'sens':'Sensitivity', 'spec':'Specificity', 'thresh':'Threshold'}
di_method = {'basic':'Basic', 'quantile':'Quantile', 'expanded':'Quantile (t-adj)', 'studentized':'Studentized', 'bca':'BCa'}


##################################
# --- (1) POWER SIMULATIONS  --- #

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

########################
# --- (2) FIGURES  --- #

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

print('~~~ End of sim_power.py ~~~')
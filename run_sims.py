import os
import numpy as np
import pandas as pd
import plotnine as pn
from funs_stats import dgp_bin, emp_roc_curve, auc_rank
from funs_support import makeifnot

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
makeifnot(dir_figures)

##############################################
# --- (1) CHECK EMPIRICAL TO GROUNDTRUTH --- #

enc_dgp = dgp_bin(mu=1, p=0.5)
nsim = 250
n = 100
holder_auc = np.zeros(nsim)
holder_roc = []
for i in range(nsim):
    dat_i = enc_dgp.dgp_bin(n=n, seed=i)
    # (i) Empirical AUROC
    holder_auc[i] = auc_rank(dat_i.y, dat_i.s)
    # (ii) Empirical ROC curve
    holder_roc.append(emp_roc_curve(dat_i.y, dat_i.s).assign(sim=i))

emp_roc = pd.concat(holder_roc).reset_index(drop=True)
gt_roc = enc_dgp.roc_curve()
df_roc = pd.concat(objs=[emp_roc, gt_roc.assign(sim=-1)])
df_roc = df_roc.assign(tt=lambda x: np.where(x['sim']==-1,'Ground Truth','Simulation'))
df_auc = pd.DataFrame({'auc':holder_auc,'gt':enc_dgp.auroc})


##############################################
# --- (2)  --- #


#######################
# --- (X) FIGURES --- #

# (i) Empirical ROC to actual
gg_roc_gt = (pn.ggplot(df_roc,pn.aes(x='fpr',y='tpr',size='tt',color='tt',alpha='tt',group='sim')) + 
    pn.theme_bw() + pn.labs(x='1-FPR',y='TPR') + 
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

import os
import numpy as np
import pandas as pd

# Make a folder if it does not exist
def makeifnot(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Round up to some factor
def round_up(num, factor):
    w, r = divmod(num, factor)
    return factor*(w + int(r>0))

# Save plotnine objects, delete existing file
def gg_save(fn, fold, gg, width, height):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.save(path, width=width, height=height, limitsize=False)

# Save a patchworklib grid
def grid_save(fn, fold, gg):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.savefig(path)

# Row vector
def rvec(x):
    return np.atleast_2d(x)

# Column vector
def cvec(x):
    return rvec(x).T

# Apply a column-wise quantile 
def quantile_mapply(arr, q, axis=0):  
    # arr=power_bs.copy();q=alpha_adj.copy()
    j = int(np.where(axis == 0, 1, 0))
    assert arr.shape[j] == q.shape[j]
    if axis == 0:
        arr = arr.copy().T
        q = q.copy().T                
    holder = np.zeros(q.shape)
    for k in range(arr.shape[0]):
        holder[k] = np.quantile(arr[k], q[k])
    if axis == 0:
        holder = holder.T
    return holder

# Interpolate values for some dataframe
def interp_df(df, cn_y, cn_x, target_y, *groups):
    # df=thresh_match; cn_x='val'; cn_y='thresh'; target_y=1.06; groups=('sim',)
    assert df.columns.isin([cn_y, cn_x]).sum() == 2, 'cn_y and cn_x must be columns in df'
    if len(groups) == 0:
        df = df.assign(gg='group1')
        groups = ['group1']
    else:
        groups = list(groups)
    # Sort data and reset index
    df = df.sort_values(groups+[cn_y]).reset_index(drop=True)
    # Get value just above and below target_y
    df = df.assign(is_below=lambda x: np.where(x[cn_y] < target_y, 'below', 'above'))
    t1 = df.groupby(groups+['is_below']).apply(lambda x: x.loc[x[cn_y].idxmax()])
    t2 = df.groupby(groups+['is_below']).apply(lambda x: x.loc[x[cn_y].idxmin()])
    t1 = t1.reset_index(drop=True).query('is_below=="below"')
    t2 = t2.reset_index(drop=True).query('is_below=="above"')
    # Pivot wide
    dat = pd.concat(objs=[t1, t2], axis=0)
    dat = dat.pivot_table([cn_y, cn_x], groups, 'is_below')
    dat.columns = ['_'.join(col).strip() for col in dat.columns.values]
    cn_x_below = cn_x+'_below'
    cn_x_above = cn_x+'_above'
    cn_y_below = cn_y+'_below'
    cn_y_above = cn_y+'_above'
    cn_x_interp = cn_x+'_interp'
    # Calculate the slope
    dat = dat.assign(slope=lambda x: (x[cn_y_above]-x[cn_y_below])/(x[cn_x_above]-x[cn_x_below]))
    dat = dat.assign(interp=lambda x: (x[cn_y_above]-target_y)/x['slope']+x[cn_x_above] )
    dat = dat.reset_index().rename(columns={'interp':cn_x_interp})
    dat = dat[groups+[cn_x_interp]]
    return dat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import scipy.stats as stats
import matplotlib
np.random.seed(0)
# ----------------------------
# 1. Generate In‑Distribution Data
# ----------------------------
num_samples = 500
domains = np.array(['A']*(num_samples//2) + ['B']*(num_samples - num_samples//2))
np.random.shuffle(domains)

U = np.where(domains=='A',
             np.random.normal(loc=2, scale=0.5, size=num_samples),
             np.random.normal(loc=-2, scale=0.5, size=num_samples))
X1 = 0.75 * U + np.random.normal(loc=0, scale=1.5, size=num_samples)
Y  = X1 - 3 * U + np.random.normal(loc=0, scale=0.5, size=num_samples)
S_U = 0.6 * U + np.random.normal(loc=0, scale=0.3, size=num_samples)

df = pd.DataFrame({
    'Domain': domains,
    'U':       U,
    'X':       X1,
    'S':       S_U,
    'Y':       Y
})

# ----------------------------
# 2. Generate OOD (Domain C/D) Test Data
# ----------------------------
num_samples     = 500
domains_test    = np.array(['C']*(num_samples//2) + ['D']*(num_samples - num_samples//2))
np.random.shuffle(domains_test)

U_test = np.where(domains_test=='C',
                  np.random.normal(loc=0, scale=0.5, size=num_samples),
                  np.random.normal(loc=4, scale=0.5, size=num_samples))
X_test = 0.75 * U_test + np.random.normal(loc=0, scale=1.5, size=num_samples)
Y_test = X_test - 3 * U_test + np.random.normal(loc=0, scale=0.5, size=num_samples)
S_U_test = 0.6 * U_test + np.random.normal(loc=0, scale=0.3, size=num_samples)
df_test = pd.DataFrame({
    'Domain': domains_test,
    'U':       U_test,
    'X':       X_test,
    'S':       S_U_test,
    'Y':       Y_test
})

# ----------------------------
# 4. Compute Domain‑Level Statistics of X
# ----------------------------
# Train set
train_stats = df.groupby('Domain', observed=True)['X'].agg(
    mean='mean', sd='std'
)
# MAD
train_stats['mad'] = df.groupby('Domain', observed=True)['X']\
    .apply(lambda x: stats.median_abs_deviation(x, scale='normal'))
# Quantiles
quantiles = df.groupby('Domain', observed=True)['X']\
    .quantile([0.25, 0.75]).unstack(level=1).rename(columns={0.25:'q25', 0.75:'q75'})
train_stats = pd.concat([train_stats, quantiles], axis=1)

# Map stats to train samples
for stat in ['mean','sd','mad','q25','q75']:
    df[f'{stat.upper()}_X'] = df['Domain'].map(train_stats[stat])

# Test set: compute test stats for unseen domains
test_stats = df_test.groupby('Domain', observed=True)['X'].agg(
    mean='mean', sd='std'
)
test_stats['mad'] = df_test.groupby('Domain', observed=True)['X']\
    .apply(lambda x: stats.median_abs_deviation(x, scale='normal'))
quantiles_t = df_test.groupby('Domain', observed=True)['X']\
    .quantile([0.25, 0.75]).unstack(level=1).rename(columns={0.25:'q25', 0.75:'q75'})
test_stats = pd.concat([test_stats, quantiles_t], axis=1)

# Map stats to test samples, filling C/D from test_stats
for stat in ['mean','sd','mad','q25','q75']:
    df_test[f'{stat.upper()}_X'] = (
        df_test['Domain'].map(train_stats[stat])
        .fillna(df_test['Domain'].map(test_stats[stat]))
    )

extra_stats = ['MEAN_X','SD_X','MAD_X','Q25_X','Q75_X']

# ----------------------------
# 5. Design Matrix Helpers
# ----------------------------
def design_matrix_no(X):
    return np.column_stack((np.ones(len(X)), X)).astype(np.float64)

def design_matrix_with(X, D):
    return np.column_stack((np.ones(len(X)), X, D)).astype(np.float64)

# ----------------------------
# 6. Masks & Data Splits
# ----------------------------
mask_A = df['Domain']=='A'
mask_B = df['Domain']=='B'

X_train_no = design_matrix_no(df['X'].values)
X_test_no  = design_matrix_no(df_test['X'].values)

D_train = df[extra_stats].values
D_test  = df_test[extra_stats].values

X_train_with = design_matrix_with(df['X'].values,    D_train)
X_test_with  = design_matrix_with(df_test['X'].values, D_test)

# ----------------------------
# 7. Training & Prediction
# ----------------------------
# A. Linear Regression
lr_no   = LinearRegression().fit(df[['X']], df['Y'])
lr_with = LinearRegression().fit(df[['X'] +  extra_stats], df['Y'])
lr_with_S = LinearRegression().fit(df[['X','S']], df['Y'])
lr_with_U = LinearRegression().fit(df[['X','U']], df['Y'])

# Compute Mean Squared Error (MSE) for ID and OOD data

pred_lr_train_no   = lr_no.predict(df[['X']])
pred_lr_train_with = lr_with.predict(df[['X'] +  extra_stats])
pred_lr_train_with_S = lr_with_S.predict(df[['X','S']])
pred_lr_train_with_U = lr_with_U.predict(df[['X','U']])

pred_lr_test_no   = lr_no.predict(df_test[['X']])
pred_lr_test_with = lr_with.predict(df_test[['X'] +  extra_stats])
pred_lr_test_with_S = lr_with_S.predict(df_test[['X','S']])
pred_lr_test_with_U = lr_with_U.predict(df_test[['X','U']])

mse_id_no = mean_squared_error(df['Y'], pred_lr_train_no)
mse_id_with = mean_squared_error(df['Y'], pred_lr_train_with)
mse_id_with_S = mean_squared_error(df['Y'], pred_lr_train_with_S)
mse_id_with_U = mean_squared_error(df['Y'], pred_lr_train_with_U)


mse_ood_no = mean_squared_error(df_test['Y'], pred_lr_test_no)
mse_ood_with = mean_squared_error(df_test['Y'], pred_lr_test_with)
mse_ood_with_S = mean_squared_error(df_test['Y'], pred_lr_test_with_S)
mse_ood_with_U = mean_squared_error(df_test['Y'], pred_lr_test_with_U)


# ----------------------------
# Visualization: 2x4 Subplots
# ----------------------------
# Top row: Methods without S_U (using only X)
# Bottom row: Methods with S_U (using X and S_U)
font = {'size': 22}
matplotlib.rc('font', **font)
fig, axes = plt.subplots(1, 4, figsize=(37, 6))

# Scatter settings for training and aggregated test data
train_scatter_kwargs = {'alpha': 0.7, 'color': 'black', 's': 50, 'marker': '^'}
test_scatter_kwargs = {'alpha': 0.7, 'color': 'gray', 's': 50, 'marker':'v'}
pred_train_kwargs = {'color': '#0173b3', 'alpha': 0.7, 'marker': '*', 's': 50}
pred_test_kwargs = {'color': '#009e74', 'alpha': 0.7, 'marker': 'X', 's': 50}

def plot_model(ax, x_train, y_train, x_test, y_test, pred_train, pred_test, title, ydata=False):
    ax.scatter(x_train, y_train, **train_scatter_kwargs, label='Train data')
    ax.scatter(x_test, y_test, **test_scatter_kwargs, label='Test data')
    ax.scatter(x_train, pred_train, **pred_train_kwargs, label='Predictions on train data')
    ax.scatter(x_test, pred_test, **pred_test_kwargs, label='Predictions on test data')
    ax.set_xlabel('X',fontsize=28, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold')
    if ydata:
        ax.set_ylabel('Y',fontsize=28, fontweight='bold')
    else:
        ax.set_yticks([])
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold')

# import pdb; pdb.set_trace()
# Top row (without S_U):
# 1. Linear Regression (X only)
plot_model(axes[0], df['X'], df['Y'], df_test['X'], df_test['Y'],
           pred_lr_train_no, pred_lr_test_no, f'L.R. Using X\nTrain MSE: {mse_id_no:.1f}\n Test MSE: {mse_ood_no:.1f}', ydata=True)
           
plot_model(axes[1], df['X'], df['Y'], df_test['X'], df_test['Y'],
           pred_lr_train_with, pred_lr_test_with, f'L.R. Using X, E\nTrain MSE: {mse_id_with:.1f}\n Test MSE: {mse_ood_with:.1f}',)

plot_model(axes[2], df['X'], df['Y'], df_test['X'], df_test['Y'],
           pred_lr_train_with_S, pred_lr_test_with_S, f'L.R. Using X,'+ r'$\mathbf{X_i}$'+f'\nTrain MSE: {mse_id_with_S:.1f}\n Test MSE: {mse_ood_with_S:.1f}',)

plot_model(axes[3], df['X'], df['Y'], df_test['X'], df_test['Y'],
           pred_lr_train_with_U, pred_lr_test_with_U, f'L.R. Using X, U (Oracle) \nTrain MSE: {mse_id_with_U:.1f}\n Test MSE: {mse_ood_with_U:.1f}',)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='right', ncol=1, fontsize=30, markerscale=3,bbox_to_anchor=(0.83, 0.55))

# fig.text(0.11, 0.99, 'Linear Regression', fontsize=25, fontweight='bold',
#          bbox=dict(facecolor='lightblue', edgecolor='none', boxstyle='square,pad=0.4'))
# fig.text(0.37, 0.99, 'XGBoost', fontsize=25, fontweight='bold',
#          bbox=dict(facecolor='lightblue', edgecolor='none', boxstyle='square,pad=0.4'))
# fig.text(0.60, 0.99, 'Group DRO', fontsize=25, fontweight='bold',
#          bbox=dict(facecolor='lightblue', edgecolor='none', boxstyle='square,pad=0.4'))
# fig.text(0.85, 0.99, 'VREX', fontsize=25, fontweight='bold',
#          bbox=dict(facecolor='lightblue', edgecolor='none', boxstyle='square,pad=0.4'))
eqs = [
    r"$\mathbf{Structural\ Equations:}$",
    "",
    r"(Hidden Confounder) $U \sim \mathcal{N}(\mu_e,0.5^2)$",
    r"$\quad \quad \quad \quad \mu_e \in \{-2,2\} \text{ for ID }$",
    r"$\quad \quad \quad \quad \mu_e \in \{0,4\} \text{ for OOD }$",
    r"(Input) $X = 0.75\,U + \mathcal{N}(0,1.5^2)$",
    r"(Outcome) $Y = X - 3\,U + \mathcal{N}(0,0.5^2)$",
    r"(Informative) $X_i = 0.6\,U + \mathcal{N}(0,0.3^2)$",
]

# Join them with newlines
eq_text = "\n".join(eqs)
fig.subplots_adjust(left=0.23)  
# Place the text at the top‐left inside the axes, in axis‐fraction coordinates
fig.text(
    0.02, 0.92,         # x, y in axes‐fraction (0 to 1)
    eq_text,
    fontsize=32,
    ha='left',  
    va='top',                # horizontal alignment
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.5,edgecolor='none'),
    transform=fig.transFigure
)

plt.tight_layout(rect=[0.24, 0.1, 0.65, 1])
plt.savefig('results/introfigure.pdf', bbox_inches='tight', dpi=500)
plt.close()

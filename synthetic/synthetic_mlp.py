import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import scipy.stats as stats
import matplotlib
np.random.seed(0)

num_samples = 500
domains = np.array(['A']*(num_samples//2) + ['B']*(num_samples - num_samples//2))
np.random.shuffle(domains)

# ----------------------------
# 1. Generate In‑Distribution Data
# ----------------------------
which = {'setting':'lowoverlap', 'mu1':-2, 'mu2':2, 'mu3':0, 'mu4':4}
# which = {'setting':'highoverlap', 'mu1':-1, 'mu2':0, 'mu3':-0.5, 'mu4':0.5}


U = np.where(domains=='A',
             np.random.normal(loc=which['mu1'], scale=0.5, size=num_samples),
             np.random.normal(loc=which['mu2'], scale=0.5, size=num_samples))
X1 = 0.75 * U + np.random.normal(loc=0, scale=1.5, size=num_samples)
Y  = X1**2 - 3 * U + np.random.normal(loc=0, scale=0.5, size=num_samples)
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
                  np.random.normal(loc=which['mu3'], scale=0.5, size=num_samples),
                  np.random.normal(loc=which['mu4'], scale=0.5, size=num_samples))
X_test = 0.75 * U_test + np.random.normal(loc=0, scale=1.5, size=num_samples)
Y_test = X_test**2 - 3 * U_test + np.random.normal(loc=0, scale=0.5, size=num_samples)
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

df['X2'] = df['X']**2

# ----------------------------
# 7. Training & Prediction
# ----------------------------
# A. MLP Regression (replace linear models with a one‐hidden‐layer MLP)
# you can tune hidden_layer_sizes, activation, solver, max_iter, etc.
mlp_params = {
    'hidden_layer_sizes': (8,),    # one hidden layer of 50 units
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 10000,
    'random_state': 0
}
mlp_no     = MLPRegressor(**mlp_params).fit(df[['X']], df['Y'])
mlp_with   = MLPRegressor(**mlp_params).fit(df[['X'] + extra_stats], df['Y'])
mlp_with_S = MLPRegressor(**mlp_params).fit(df[['X','S']], df['Y'])
mlp_with_U = MLPRegressor(**mlp_params).fit(df[['X','U']], df['Y'])

# Compute Mean Squared Error (MSE) for ID and OOD data

pred_mlp_train_no     = mlp_no.predict(df[['X']])
pred_mlp_train_with   = mlp_with.predict(df[['X'] + extra_stats])
pred_mlp_train_with_S = mlp_with_S.predict(df[['X','S']])
pred_mlp_train_with_U = mlp_with_U.predict(df[['X','U']])

pred_mlp_test_no     = mlp_no.predict(df_test[['X']])
pred_mlp_test_with     = mlp_with.predict(df_test[['X'] + extra_stats])
pred_mlp_test_with_S   = mlp_with_S.predict(df_test[['X','S']])
pred_mlp_test_with_U   = mlp_with_U.predict(df_test[['X','U']])

mse_id_no     = mean_squared_error(df['Y'], pred_mlp_train_no)
mse_id_with   = mean_squared_error(df['Y'], pred_mlp_train_with)
mse_id_with_S = mean_squared_error(df['Y'], pred_mlp_train_with_S)
mse_id_with_U = mean_squared_error(df['Y'], pred_mlp_train_with_U)

mse_ood_no     = mean_squared_error(df_test['Y'], pred_mlp_test_no)
mse_ood_with   = mean_squared_error(df_test['Y'], pred_mlp_test_with)
mse_ood_with_S = mean_squared_error(df_test['Y'], pred_mlp_test_with_S)
mse_ood_with_U = mean_squared_error(df_test['Y'], pred_mlp_test_with_U)


# ----------------------------
# Visualization: 2x4 Subplots
# ----------------------------
# Top row: Methods without S_U (using only X)
# Bottom row: Methods with S_U (using X and S_U)
font = {'size': 22}
matplotlib.rc('font', **font)
fig, axes = plt.subplots(2, 4, figsize=(37, 12))

# Scatter settings for training and aggregated test data
train_scatter_kwargs = {'alpha': 0.7, 'color': 'black', 's': 50, 'marker': '^'}
test_scatter_kwargs = {'alpha': 0.7, 'color': 'gray', 's': 50, 'marker':'v'}
pred_train_kwargs = {'color': '#0173b3', 'alpha': 0.7, 'marker': '*', 's': 50}
pred_test_kwargs = {'color': '#009e74', 'alpha': 0.7, 'marker': 'X', 's': 50}

def plot_model(ax, x_train, y_train, x_test, y_test, pred_train, pred_test, title=None, ydata=False, xlabel='X'):
    ax.scatter(x_train, y_train, **train_scatter_kwargs, label='Train data')
    ax.scatter(x_test, y_test, **test_scatter_kwargs, label='Test data')
    ax.scatter(x_train, pred_train, **pred_train_kwargs, label='Predictions on train data')
    ax.scatter(x_test, pred_test, **pred_test_kwargs, label='Predictions on test data')
    ax.set_xlabel(xlabel,fontsize=28, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold')
    if ydata:
        ax.set_ylabel('Y',fontsize=28, fontweight='bold')
    else:
        ax.set_yticks([])
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold')

# import pdb; pdb.set_trace()
# 1. Linear Regression (X only)
plot_model(axes[0,0], df['X'], df['Y'], df_test['X'], df_test['Y'],
           pred_mlp_train_no, pred_mlp_test_no, f'MLP Using X\nTrain MSE: {mse_id_no:.1f}\n Test MSE: {mse_ood_no:.1f}', ydata=True, xlabel='X')
           
plot_model(axes[0,1], df['X'], df['Y'], df_test['X'], df_test['Y'],
           pred_mlp_train_with, pred_mlp_test_with, f'MLP Using X, E\nTrain MSE: {mse_id_with:.1f}\n Test MSE: {mse_ood_with:.1f}',xlabel='X')

plot_model(axes[0,2], df['X'], df['Y'], df_test['X'], df_test['Y'],
           pred_mlp_train_with_S, pred_mlp_test_with_S, f'MLP Using X,'+ r'$\mathbf{X_i}$'+f'\nTrain MSE: {mse_id_with_S:.1f}\n Test MSE: {mse_ood_with_S:.1f}',xlabel='X')

plot_model(axes[0,3], df['X'], df['Y'], df_test['X'], df_test['Y'],
           pred_mlp_train_with_U, pred_mlp_test_with_U, f'MLP Using X, U (Oracle) \nTrain MSE: {mse_id_with_U:.1f}\n Test MSE: {mse_ood_with_U:.1f}',xlabel='X')

plot_model(axes[1,0], df['U'], df['Y'], df_test['U'], df_test['Y'],pred_mlp_train_no, pred_mlp_test_no, ydata=True,xlabel='U')
           
plot_model(axes[1,1], df['U'], df['Y'], df_test['U'], df_test['Y'],pred_mlp_train_with, pred_mlp_test_with,xlabel='U')

plot_model(axes[1,2], df['U'], df['Y'], df_test['U'], df_test['Y'],pred_mlp_train_with_S, pred_mlp_test_with_S,xlabel='U')

plot_model(axes[1,3], df['U'], df['Y'], df_test['U'], df_test['Y'],pred_mlp_train_with_U, pred_mlp_test_with_U,xlabel='U')



handles, labels = axes[0,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left', ncol=1, fontsize=30, markerscale=3,bbox_to_anchor=(0.02, 0.2))

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
    r"(Outcome) $Y = X^2 - 3\,U + \mathcal{N}(0,0.5^2)$",
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
plt.savefig('results/informative/synthetic_mlp_'+which['setting']+'.pdf', bbox_inches='tight', dpi=500)
plt.close()

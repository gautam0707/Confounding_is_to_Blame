import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import npeet.entropy_estimators as ee

# Setting 1: Varying noise level in S_U
def generate_data_setting1(lambda_s, num_samples=500, domain_C_size=250):
    domains = np.array(['A']*(num_samples//2) + ['B']*(num_samples - num_samples//2))
    np.random.shuffle(domains)
    U = np.zeros(num_samples)
    for i in range(num_samples):
        if domains[i] == 'A':
            U[i] = np.random.normal(loc=2, scale=0.5)
        else:
            U[i] = np.random.normal(loc=-2, scale=0.5)
    X1 = 0.3 * U + np.random.normal(loc=0, scale=1, size=num_samples)
    S_U = U + np.random.normal(loc=0, scale=lambda_s, size=num_samples)
    Y = X1 - 2 * U + np.random.normal(loc=0, scale=0.5, size=num_samples)
    df = pd.DataFrame({'Domain': domains, 'U': U, 'X': X1, 'S_U': S_U, 'Y': Y})
    
    U_C = np.random.normal(loc=0, scale=0.5, size=domain_C_size)
    X1_C = 0.3 * U_C + np.random.normal(loc=0, scale=1, size=domain_C_size)
    S_U_C = U_C + np.random.normal(loc=0, scale=lambda_s, size=domain_C_size)
    Y_C = X1_C - 2 * U_C + np.random.normal(loc=0, scale=0.5, size=domain_C_size)
    df_C = pd.DataFrame({'Domain': ['C']*domain_C_size, 'U': U_C, 'X': X1_C, 'S_U': S_U_C, 'Y': Y_C})

    U_D = np.random.normal(loc=4, scale=0.5, size=domain_C_size)
    X1_D = 0.3 * U_D + np.random.normal(loc=0, scale=1, size=domain_C_size)
    S_U_D = U_D + np.random.normal(loc=0, scale=lambda_s, size=domain_C_size)
    Y_D = X1_D - 2 * U_D + np.random.normal(loc=0, scale=0.5, size=domain_C_size)
    df_D = pd.DataFrame({'Domain': ['D']*domain_C_size, 'U': U_D, 'X': X1_D, 'S_U': S_U_D, 'Y': Y_D})

    df_test = pd.concat([df_C, df_D], ignore_index=True)

    return df, df_test

def design_matrix_no(X):
    return np.column_stack((np.ones(len(X)), X))

def design_matrix_with(X, S):
    return np.column_stack((np.ones(len(X)), X, S))

def train_group_dro(X, Y, mask_A, mask_B, num_iters=2000, lr=0.001, eta=0.01):
    theta = np.random.randn(X.shape[1])
    q = np.array([0.5, 0.5])
    X_A = X[mask_A]
    Y_A = Y[mask_A]
    X_B = X[mask_B]
    Y_B = Y[mask_B]
    for it in range(num_iters):
        pred_A = X_A @ theta
        pred_B = X_B @ theta
        loss_A = np.mean((Y_A - pred_A)**2)
        loss_B = np.mean((Y_B - pred_B)**2)
        grad_A = -2 * X_A.T @ (Y_A - pred_A) / len(Y_A)
        grad_B = -2 * X_B.T @ (Y_B - pred_B) / len(Y_B)
        grad = q[0]*grad_A + q[1]*grad_B
        theta = theta - lr * grad
        q[0] *= np.exp(eta * loss_A)
        q[1] *= np.exp(eta * loss_B)
        q = q / np.sum(q)
    return theta

def train_irm(X, Y, envs, num_iters=2000, lr=0.001, lambda_irm=1.0):
    """
    Invariant Risk Minimization via gradient‐penalty on per‐env losses.
    envs: array of environment labels (e.g. 'A','B')
    """
    theta = np.random.randn(X.shape[1])
    # map env labels to integer indices
    uniq = np.unique(envs)
    for it in range(num_iters):
        grads = []
        losses = []
        # compute per‐env gradient
        for e in uniq:
            mask = (envs == e)
            Xe, Ye = X[mask], Y[mask]
            pred = Xe @ theta
            loss = np.mean((Ye - pred)**2)
            grad = -2 * Xe.T @ (Ye - pred) / len(Ye)
            grads.append(grad)
            losses.append(loss)
        # IRM penalty = sum of squared norms of per‐env gradients
        penalty_grad = 2 * sum(grads)  # gradient of sum ||grad_e||^2 wrt theta approx
        # ERM gradient
        erm_grad = sum(grads)
        # update
        theta = theta - lr * (erm_grad + lambda_irm * penalty_grad)
    return theta

# Updated VREX with gradient clipping and lower learning rate for stability.
def train_vrex(X, Y, mask_A, mask_B, num_iters=500, learning_rate=0.001, lambda_var=1, grad_clip=1e3):
    theta = np.random.randn(X.shape[1])
    X_A = X[mask_A]
    Y_A = Y[mask_A]
    X_B = X[mask_B]
    Y_B = Y[mask_B]
    for it in range(num_iters):
        Y_hat_A = X_A @ theta
        Y_hat_B = X_B @ theta
        loss_A = np.mean((Y_A - Y_hat_A)**2)
        loss_B = np.mean((Y_B - Y_hat_B)**2)
    
        grad_A = (2 / len(Y_A)) * X_A.T @ (Y_hat_A - Y_A)
        grad_B = (2 / len(Y_B)) * X_B.T @ (Y_hat_B - Y_B)

        mean_grad = 0.5 * (grad_A + grad_B)
        var_grad = 0.5 * (loss_A - loss_B) * (grad_A - grad_B)
        total_grad = mean_grad + lambda_var * var_grad
        total_grad = np.clip(total_grad, -grad_clip, grad_clip)
        theta = theta - learning_rate * total_grad
        if np.isnan(total_grad).any():
            break
    return theta

#######################################
# 4. MI DECOMPOSITION FUNCTION
#######################################
def compute_mi_terms(rep, hatY, Y_vals, E, label_shift):
    try:
        inf = ee.cmi(rep, Y_vals, E)    # Cond. Informativeness
        inv = ee.cmi(rep, E, Y_vals)    # Variation
        lcs = ee.mi(rep, E)             # Feature shift (covariate shift)
        cs  = ee.cmi(Y_vals, E, rep)     # concept shift
        res = ee.cmi(rep, Y_vals, hatY)  # residual
        overall = inf - 0.5 * inv + 0.5 * label_shift + 0.5 * lcs - 0.5 * cs - res
        return inf, inv, lcs, cs, res, overall
    except Exception as e:
        print("Error computing MI terms:", e)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def evaluate_setting1(lambda_s,ID):
    df, df_C = generate_data_setting1(lambda_s)
    X_train = design_matrix_with(df['X'].values, df['S_U'].values)
    Y_train = df['Y'].values
    mask_A = (df['Domain'] == 'A').values
    mask_B = (df['Domain'] == 'B').values
    # Methods:
    model_lr = LinearRegression().fit(df[['X','S_U']], df['Y'])
    pred_lr = model_lr.predict(df[['X','S_U']]).reshape(-1,1)
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(df[['X','S_U']], df['Y'])
    pred_xgb = xgb_model.predict(df[['X','S_U']]).reshape(-1,1)
    theta_dro = train_group_dro(X_train, Y_train, mask_A, mask_B)
    pred_dro = (X_train @ theta_dro).reshape(-1,1)
    theta_irm = train_irm(X_train, Y_train, df['Domain'].values, num_iters=2000, lr=0.001, lambda_irm=1.0)
    pred_irm = (X_train @ theta_irm).reshape(-1,1)
    theta_vrex = train_vrex(X_train, Y_train, mask_A, mask_B)
    pred_vrex = (X_train @ theta_vrex).reshape(-1,1)
    # Compute Mean Squared Error (MSE) for each method
    
    if not ID:
        df = df_C
        E, _ = pd.factorize(df['Domain'])
        E = E.reshape(-1,1)
        Y_vals = df['Y'].values.reshape(-1,1)
        label_shift = ee.mi(Y_vals, E)
        pred_lr = model_lr.predict(df[['X','S_U']]).reshape(-1,1)
        pred_xgb = xgb_model.predict(df[['X','S_U']]).reshape(-1,1)
        X_train = design_matrix_with(df['X'].values, df['S_U'].values)
        pred_dro = (X_train @ theta_dro).reshape(-1,1)
        pred_vrex = (X_train @ theta_vrex).reshape(-1,1)
        pred_irm = (X_train @ theta_irm).reshape(-1,1)  


        mse_lr = mean_squared_error(df['Y'], pred_lr)
        mse_xgb = mean_squared_error(df['Y'], pred_xgb)
        mse_dro = mean_squared_error(df['Y'], pred_dro)
        mse_vrex = mean_squared_error(df['Y'], pred_vrex)
        mse_irm = mean_squared_error(df['Y'], pred_irm)

        mi_lr = compute_mi_terms(pred_lr, pred_lr, Y_vals, E, label_shift)
        mi_xgb = compute_mi_terms(pred_xgb, pred_xgb, Y_vals, E, label_shift)
        mi_dro = compute_mi_terms(pred_dro, pred_dro, Y_vals, E, label_shift)
        mi_vrex = compute_mi_terms(pred_vrex, pred_vrex, Y_vals, E, label_shift)
        mi_irm = compute_mi_terms(pred_irm, pred_irm, Y_vals, E, label_shift)

        results = {
        "Linear Regression": (mi_lr[0], mi_lr[1],mi_lr[2],mi_lr[3],mi_lr[4],mi_lr[5], mse_lr),
        "XGB": (mi_xgb[0], mi_xgb[1],mi_xgb[2],mi_xgb[3],mi_xgb[4],mi_xgb[5], mse_xgb),  
        "GDRO": (mi_dro[0], mi_dro[1],mi_dro[2],mi_dro[3],mi_dro[4],mi_dro[5], mse_dro),
        "VREX": (mi_vrex[0], mi_vrex[1],mi_vrex[2],mi_vrex[3],mi_vrex[4],mi_vrex[5], mse_vrex),
        "IRM": (mi_irm[0], mi_irm[1],mi_irm[2],mi_irm[3],mi_irm[4],mi_irm[5], mse_irm)
        }
        return results
    else:
    # MI computations:
        E, _ = pd.factorize(df['Domain'])
        E = E.reshape(-1,1)
        Y_vals = df['Y'].values.reshape(-1,1)
        label_shift = ee.mi(Y_vals, E)
        mse_lr = mean_squared_error(df['Y'], pred_lr)
        mse_xgb = mean_squared_error(df['Y'], pred_xgb)
        mse_dro = mean_squared_error(df['Y'], pred_dro)
        mse_vrex = mean_squared_error(df['Y'], pred_vrex)
        mse_irm = mean_squared_error(df['Y'], pred_irm)

        mi_lr = compute_mi_terms(pred_lr, pred_lr, Y_vals, E, label_shift)
        mi_xgb = compute_mi_terms(pred_xgb, pred_xgb, Y_vals, E, label_shift)
        mi_dro = compute_mi_terms(pred_dro, pred_dro, Y_vals, E, label_shift)
        mi_vrex = compute_mi_terms(pred_vrex, pred_vrex, Y_vals, E, label_shift)
        mi_irm = compute_mi_terms(pred_irm, pred_irm, Y_vals, E, label_shift)
        results = {
        "Linear Regression": (mi_lr[0], mi_lr[1],mi_lr[2],mi_lr[3],mi_lr[4],mi_lr[5], mse_lr),
        "XGB": (mi_xgb[0], mi_xgb[1],mi_xgb[2],mi_xgb[3],mi_xgb[4],mi_xgb[5], mse_xgb),  
        "GDRO": (mi_dro[0], mi_dro[1],mi_dro[2],mi_dro[3],mi_dro[4],mi_dro[5], mse_dro),
        "VREX": (mi_vrex[0], mi_vrex[1],mi_vrex[2],mi_vrex[3],mi_vrex[4],mi_vrex[5], mse_vrex),
        "IRM": (mi_irm[0], mi_irm[1],mi_irm[2],mi_irm[3],mi_irm[4],mi_irm[5], mse_irm)
        }
        return results
    
lambda_s_rep = np.linspace(0, 2.0, 20)

def accumulate_results(param_values, eval_function,ID):
    methods_list = ["Linear Regression", "XGB", "GDRO", "VREX", "IRM"]
    res = {meth: {"inf": [], "inv": [], "lcs":[], "cs":[], "mse":[]} for meth in methods_list}
    for p in param_values:
        r = eval_function(p,ID)
        for meth in methods_list:
            res[meth]["inf"].append(r[meth][0])
            res[meth]["inv"].append(r[meth][1])
            res[meth]["lcs"].append(r[meth][2])
            res[meth]["cs"].append(r[meth][3])
            res[meth]["mse"].append(r[meth][6])
    return res

results_setting1_id = accumulate_results(lambda_s_rep, evaluate_setting1,ID=True)
results_setting1_ood = accumulate_results(lambda_s_rep, evaluate_setting1,ID=False)



#######################################
# 7. PLOTTING: THREE SUBPLOTS
#######################################
methods_list = ["Linear Regression", "XGB", "GDRO", "IRM", "VREX"]
colors = {"Linear Regression": "red", "XGB": "#0173b3", "GDRO": "#de8f08", "VREX": "#009e74", "IRM": "black"}
markers = ['v','^','<','>','o']
import matplotlib
font = {'size': 22}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(1, 6, figsize=(30, 5), sharey=False)


ax = axs[0]
for meth in methods_list:
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_id[meth]["mse"][::-1], marker=markers[methods_list.index(meth)], linestyle='-', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel(r"$\sigma_s$", fontsize=22, fontweight='bold')
ax.set_title("Train", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.invert_xaxis()
ax.grid(True)

ax = axs[1]
for meth in methods_list:
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_ood[meth]["mse"][::-1], marker=markers[methods_list.index(meth)], linestyle='-', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel(r"$\sigma_s$", fontsize=22, fontweight='bold')
ax.set_title("Test", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.invert_xaxis()
ax.grid(True)


ax = axs[2]
for meth in methods_list:
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_id[meth]["inf"][::-1], marker=markers[methods_list.index(meth)], linestyle=':', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_id[meth]["inv"][::-1], marker=markers[methods_list.index(meth)], linestyle='-', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel(r"$\sigma_s$", fontsize=22, fontweight='bold')
ax.set_title("Train", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.invert_xaxis()
ax.grid(True)

ax = axs[3]
for meth in methods_list:
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_ood[meth]["inf"][::-1], marker=markers[methods_list.index(meth)], linestyle=':', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_ood[meth]["inv"][::-1], marker=markers[methods_list.index(meth)], linestyle='-', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel(r"$\sigma_s$", fontsize=22, fontweight='bold')
ax.set_title("Test", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.invert_xaxis()
ax.grid(True)

ax = axs[4]
for meth in methods_list:
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_id[meth]["lcs"][::-1], marker=markers[methods_list.index(meth)], linestyle='-.', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_id[meth]["cs"][::-1], marker=markers[methods_list.index(meth)], linestyle='--', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel(r"$\sigma_s$", fontsize=22, fontweight='bold')
ax.set_title("Train", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.invert_xaxis()
ax.grid(True)

ax = axs[5]
for meth in methods_list:
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_ood[meth]["lcs"][::-1], marker=markers[methods_list.index(meth)], linestyle='-.', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
    ax.plot(np.linspace(2.0, 0, 20), results_setting1_ood[meth]["cs"][::-1], marker=markers[methods_list.index(meth)], linestyle='--', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel(r"$\sigma_s$", fontsize=22, fontweight='bold')
ax.set_title("Test", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.invert_xaxis()
ax.grid(True)


handles1, labels1 = axs[0].get_legend_handles_labels()
handles2 = [plt.Line2D([0], [0], color='black', linestyle=':', label='Cond. Informativeness',linewidth=4),
            plt.Line2D([0], [0], color='black', linestyle='-', label='Variation',linewidth=4),
            plt.Line2D([0], [0], color='black', linestyle='-.', label='Feature Shift',linewidth=4),
            plt.Line2D([0], [0], color='black', linestyle='--', label='Concept Shift',linewidth=4)]
labels2 = ['Cond. Informativeness', 'Variation', 'Feature Shift', 'Concept Shift']

handles = handles1+handles2
labels = labels1+labels2

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=9, fontsize=24, markerscale=2)

fig.text(0.1, 0.99, 'Mean Squared Error', fontsize=26,fontweight='bold',
         bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.4'))

fig.text(0.4, 0.99, 'Cond. Informativeness & Variation', fontsize=26, fontweight='bold',
         bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.4'))

fig.text(0.71, 0.99, 'Feature Shift & Concept Shift', fontsize=26, fontweight='bold',
         bbox=dict(facecolor='none', edgecolor='none', boxstyle='square,pad=0.4'))

plt.tight_layout(rect=[0, 0, 1, 0.93])

fig.savefig('results/informative/varyingnoise.pdf', dpi=300, bbox_inches='tight')
plt.close()
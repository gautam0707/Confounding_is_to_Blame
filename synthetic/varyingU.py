import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import npeet.entropy_estimators as ee
import matplotlib

# Setting 1: Varying noise level in S_U
def generate_data_setting3(num_samples=500, domain_C_size=250):
    # Generate training data
    domains = np.array(['A']*(num_samples//2) + ['B']*(num_samples - num_samples//2))
    np.random.shuffle(domains)
    
    # For each sample, generate 20 hidden confounders U_i.
    U = np.zeros((num_samples, 20))
    for i in range(num_samples):
        if domains[i] == 'A':
            U[i, :] = np.random.normal(loc=2, scale=0.5, size=20)
        else:
            U[i, :] = np.random.normal(loc=-2, scale=0.5, size=20)
    # X and Y are generated from the average of the 20 U's.
    U_avg = np.mean(U, axis=1)
    X = 0.3 * U_avg + np.random.normal(loc=0, scale=1, size=num_samples)
    Y = X - 2 * U_avg + np.random.normal(loc=0, scale=0.5, size=num_samples)
    
    # Generate 20 proxy variables; each S_Ui is generated from its corresponding U_i.
    S_U_all = {}
    for j in range(1, 21):
        S_U_all[f'S_U{j}'] = 0.2 * U[:, j-1] + np.random.normal(loc=0, scale=0.1, size=num_samples)
    
    df = pd.DataFrame({'Domain': domains, 'X': X, 'Y': Y})
    # Save the hidden variables if needed:
    for j in range(1, 21):
        df[f'U{j}'] = U[:, j-1]
        df[f'S_U{j}'] = S_U_all[f'S_U{j}']
    
    # Generate OOD test data for Domain C: U_i ~ N(0, 0.5^2)
    U_C = np.zeros((domain_C_size, 20))
    for i in range(domain_C_size):
        U_C[i, :] = np.random.normal(loc=0, scale=0.5, size=20)
    U_avg_C = np.mean(U_C, axis=1)
    X_C = 0.3 * U_avg_C + np.random.normal(loc=0, scale=1, size=domain_C_size)
    Y_C = X_C - 2 * U_avg_C + np.random.normal(loc=0, scale=0.5, size=domain_C_size)
    S_U_all_C = {}
    for j in range(1, 21):
        S_U_all_C[f'S_U{j}'] = 0.2 * U_C[:, j-1] + np.random.normal(loc=0, scale=0.1, size=domain_C_size)
    
    df_C = pd.DataFrame({'Domain': ['C']*domain_C_size, 'X': X_C, 'Y': Y_C})
    for j in range(1, 21):
        df_C[f'U{j}'] = U_C[:, j-1]
        df_C[f'S_U{j}'] = S_U_all_C[f'S_U{j}']

    U_D = np.zeros((domain_C_size, 20))
    for i in range(domain_C_size):
        U_D[i, :] = np.random.normal(loc=4, scale=0.5, size=20)
    U_avg_D = np.mean(U_D, axis=1)
    X_D = 0.3 * U_avg_D + np.random.normal(loc=0, scale=1, size=domain_C_size)
    Y_D = X_D - 2 * U_avg_D + np.random.normal(loc=0, scale=0.5, size=domain_C_size)
    S_U_all_D = {}
    for j in range(1, 21):
        S_U_all_D[f'S_U{j}'] = 0.2 * U_D[:, j-1] + np.random.normal(loc=0, scale=0.1, size=domain_C_size)
    
    df_D = pd.DataFrame({'Domain': ['D']*domain_C_size, 'X': X_D, 'Y': Y_D})
    for j in range(1, 21):
        df_D[f'U{j}'] = U_D[:, j-1]
        df_D[f'S_U{j}'] = S_U_all_D[f'S_U{j}']
    

    df_test = pd.concat([df_C, df_D], ignore_index=True)

    # For modeling with S_U, we will use only the first m proxy variables.
    return df, df_test

def design_matrix_no(X):
    return np.column_stack((np.ones(len(X)), X))

def design_matrix_with(X, S):
    return np.column_stack((np.ones(len(X)), X, S))

def train_group_dro(X, Y, mask_A, mask_B, num_iters=2000, lr=0.001, eta=0.01):
    theta = np.random.randn(X.shape[1])
    q = np.array([0.5, 0.5])  # initial weights for groups A and B
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

# ----------------------------
# VREX Implementation
# ----------------------------
def train_vrex(X, Y, mask_A, mask_B, num_iters=5000, learning_rate=0.01, lambda_var=1):
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
        grad_A = (2/len(Y_A)) * X_A.T @ (Y_hat_A - Y_A)
        grad_B = (2/len(Y_B)) * X_B.T @ (Y_hat_B - Y_B)
        mean_grad = 0.5*(grad_A+grad_B)
        var_grad = 0.5*(loss_A - loss_B)*(grad_A - grad_B)
        total_grad = mean_grad + lambda_var*var_grad
        theta = theta - learning_rate * total_grad
    return theta

#######################################
# 4. MI DECOMPOSITION FUNCTION
#######################################
def compute_mi_terms(rep, hatY, Y_vals, E, label_shift):
    try:
        inf = ee.cmi(rep, Y_vals, E)    # Cond. Informativeness
        inv = ee.cmi(rep, E, Y_vals)    # Variation
        lcs = ee.mi(rep, E)             # Feature Shift (covariate shift)
        cs  = ee.cmi(Y_vals, E, rep)     # concept shift
        res = ee.cmi(rep, Y_vals, hatY)  # residual
        overall = inf - 0.5 * inv + 0.5 * label_shift + 0.5 * lcs - 0.5 * cs - res
        return inf, inv, lcs, cs, res, overall
    except Exception as e:
        print("Error computing MI terms:", e)
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def evaluate_setting3(m_val,ID):
    df, df_C = generate_data_setting3()
    
    Y_train = df['Y'].values
    Y_test = df_C['Y'].values

    # For models "with S_U", extract first m proxy columns:
    su_cols = [f'S_U{j}' for j in range(1, m_val+1)]
    S_train = df[su_cols].values
    S_test = df_C[su_cols].values
    X_train = design_matrix_with(df['X'].values, S_train)  # dims: (n, 2+m)
    X_test = design_matrix_with(df_C['X'].values, S_test)
    
    # Create domain masks for training (for DRO and VREX, use original Domain column):
    mask_A = (df['Domain'] == 'A').values
    mask_B = (df['Domain'] == 'B').values

    # import pdb; pdb.set_trace()
    # With S_U: (X + first m S_U's)
    model_lr = LinearRegression().fit(df[['X']+su_cols], df['Y'])
    theta_dro = train_group_dro(X_train, Y_train, mask_A, mask_B, num_iters=2000, lr=0.001, eta=0.01)
    xgb = XGBRegressor(random_state=42)
    xgb.fit(df[['X']+su_cols], df['Y'])
    theta_vrex = train_vrex(X_train, Y_train, mask_A, mask_B, num_iters=5000, learning_rate=0.01, lambda_var=1)
    theta_irm = train_irm(X_train, Y_train, df['Domain'].values, num_iters=2000, lr=0.001, lambda_irm=1.0)


    if ID:
        E, _ = pd.factorize(df['Domain'])
        E = E.reshape(-1,1)
        pred_lr = model_lr.predict(df[['X']+su_cols])
        pred_xgb = xgb.predict(df[['X']+su_cols])
        pred_dro = X_train @ theta_dro
        pred_vrex = X_train @ theta_vrex
        pred_irm = X_train @ theta_irm
        mse_lr = mean_squared_error(Y_train, pred_xgb)
        mse_dro = mean_squared_error(Y_train, pred_dro)
        mse_xgb = mean_squared_error(Y_train, pred_xgb)
        mse_vrex = mean_squared_error(Y_train, pred_vrex)
        mse_irm = mean_squared_error(Y_train, X_train @ theta_irm)
        label_shift = ee.mi(Y_train, E)
        mi_lr = compute_mi_terms(pred_lr, pred_lr, Y_train, E, label_shift)
        mi_xgb = compute_mi_terms(pred_xgb, pred_xgb, Y_train, E, label_shift)
        mi_dro = compute_mi_terms(pred_dro, pred_dro, Y_train, E, label_shift)
        mi_vrex = compute_mi_terms(pred_vrex, pred_vrex, Y_train, E, label_shift)
        mi_irm = compute_mi_terms(pred_irm, pred_irm, Y_train, E, label_shift)
        return {
            "Linear Regression": (mi_lr[0], mi_lr[1],mi_lr[2],mi_lr[3],mi_lr[4],mi_lr[5], mse_lr),
            "XGB": (mi_xgb[0], mi_xgb[1],mi_xgb[2],mi_xgb[3],mi_xgb[4],mi_xgb[5], mse_xgb),  
            "GDRO": (mi_dro[0], mi_dro[1],mi_dro[2],mi_dro[3],mi_dro[4],mi_dro[5], mse_dro),
            "VREX": (mi_vrex[0], mi_vrex[1],mi_vrex[2],mi_vrex[3],mi_vrex[4],mi_vrex[5], mse_vrex),
            "IRM": (mi_irm[0], mi_irm[1],mi_irm[2],mi_irm[3],mi_irm[4],mi_irm[5], mse_irm)
            }

    else:
        df = df_C
        E, _ = pd.factorize(df['Domain'])
        E = E.reshape(-1,1)
        pred_lr = model_lr.predict(df[['X']+su_cols])
        pred_xgb = xgb.predict(df[['X']+su_cols])
        pred_dro = X_test @ theta_dro
        pred_vrex = X_test @ theta_vrex
        pred_irm = X_test @ theta_irm
        mse_lr = mean_squared_error(Y_test, pred_xgb)
        mse_dro = mean_squared_error(Y_test, pred_dro)
        mse_xgb = mean_squared_error(Y_test, pred_xgb)
        mse_vrex = mean_squared_error(Y_test, pred_vrex)
        mse_irm = mean_squared_error(Y_test, pred_irm)
        label_shift = ee.mi(Y_test, E)
        mi_lr = compute_mi_terms(pred_lr, pred_lr, Y_test, E, label_shift)
        mi_xgb = compute_mi_terms(pred_xgb, pred_xgb, Y_test, E, label_shift)
        mi_dro = compute_mi_terms(pred_dro, pred_dro, Y_test, E, label_shift)
        mi_vrex = compute_mi_terms(pred_vrex, pred_vrex, Y_test, E, label_shift)
        mi_irm = compute_mi_terms(pred_irm, pred_irm, Y_test, E, label_shift)
        return {
            "Linear Regression": (mi_lr[0], mi_lr[1],mi_lr[2],mi_lr[3],mi_lr[4],mi_lr[5], mse_lr),
            "XGB": (mi_xgb[0], mi_xgb[1],mi_xgb[2],mi_xgb[3],mi_xgb[4],mi_xgb[5], mse_xgb),  
            "GDRO": (mi_dro[0], mi_dro[1],mi_dro[2],mi_dro[3],mi_dro[4],mi_dro[5], mse_dro),
            "VREX": (mi_vrex[0], mi_vrex[1],mi_vrex[2],mi_vrex[3],mi_vrex[4],mi_vrex[5], mse_vrex),
            "IRM": (mi_irm[0], mi_irm[1],mi_irm[2],mi_irm[3],mi_irm[4],mi_irm[5], mse_irm)
            }

def accumulate_results(param_values, eval_function,ID):
    methods_list = ["Linear Regression", "XGB", "GDRO", "IRM","VREX"]
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

m_values = np.arange(0, 21)

results_setting1_id = accumulate_results(m_values, evaluate_setting3,ID=True)
results_setting1_ood = accumulate_results(m_values, evaluate_setting3,ID=False)



#######################################
# 7. PLOTTING: THREE SUBPLOTS
#######################################
methods_list = ["Linear Regression", "XGB", "GDRO", "IRM","VREX"]
colors = {"Linear Regression": "red", "XGB": "#0173b3", "GDRO": "#de8f08", "VREX": "#009e74","IRM":"black"}
markers = ['v','^','<','>','o']
font = {'size': 22}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(1, 6, figsize=(30, 5), sharey=False)


ax = axs[0]
for meth in methods_list:
    ax.plot(np.arange(0, 21), results_setting1_id[meth]["mse"], marker=markers[methods_list.index(meth)], linestyle='-', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel("|U|", fontsize=22, fontweight='bold')
ax.set_title("Train", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.grid(True)

ax = axs[1]
for meth in methods_list:
    ax.plot(np.arange(0, 21), results_setting1_ood[meth]["mse"], marker=markers[methods_list.index(meth)], linestyle='-', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel("|U|", fontsize=22, fontweight='bold')
ax.set_title("Test", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.grid(True)


ax = axs[2]
for meth in methods_list:
    ax.plot(np.arange(0, 21), results_setting1_id[meth]["inf"], marker=markers[methods_list.index(meth)], linestyle=':', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
    ax.plot(np.arange(0, 21), results_setting1_id[meth]["inv"], marker=markers[methods_list.index(meth)], linestyle='-', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel("|U|", fontsize=22, fontweight='bold')
ax.set_title("Train", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.grid(True)

ax = axs[3]
for meth in methods_list:
    ax.plot(np.arange(0, 21), results_setting1_ood[meth]["inf"], marker=markers[methods_list.index(meth)], linestyle=':', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
    ax.plot(np.arange(0, 21), results_setting1_ood[meth]["inv"], marker=markers[methods_list.index(meth)], linestyle='-', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel("|U|", fontsize=22, fontweight='bold')
ax.set_title("Test", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.grid(True)

ax = axs[4]
for meth in methods_list:
    ax.plot(np.arange(0, 21), results_setting1_id[meth]["lcs"], marker=markers[methods_list.index(meth)], linestyle='-.', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
    ax.plot(np.arange(0, 21), results_setting1_id[meth]["cs"], marker=markers[methods_list.index(meth)], linestyle='--', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel("|U|", fontsize=22, fontweight='bold')
ax.set_title("Train", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
ax.grid(True)

ax = axs[5]
for meth in methods_list:
    ax.plot(np.arange(0, 21), results_setting1_ood[meth]["lcs"], marker=markers[methods_list.index(meth)], linestyle='-.', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
    ax.plot(np.arange(0, 21), results_setting1_ood[meth]["cs"], marker=markers[methods_list.index(meth)], linestyle='--', color=colors[meth],
            label=meth, markersize=10, markevery=2,linewidth=3)
ax.set_xlabel("|U|", fontsize=22, fontweight='bold')
ax.set_title("Test", fontsize=22, fontweight='bold')
plt.setp(ax.get_yticklabels(), fontweight='bold')
plt.setp(ax.get_xticklabels(), fontweight='bold')
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

fig.savefig('results/informative/varyingU.pdf', dpi=300, bbox_inches='tight')
plt.close()
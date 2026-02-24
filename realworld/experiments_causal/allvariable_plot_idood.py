from experiments_causal.plot_experiment import get_results
from experiments_causal.plot_config_colors import *
from experiments_causal.plot_config_tasks import dic_title
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
import matplotlib.markers as mmark
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

# Set plot configurations
sns.set_context("paper")
sns.set_style("white")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 1200
list_mak = [
    mmark.MarkerStyle("s"),
    mmark.MarkerStyle("D"),
    mmark.MarkerStyle("o"),
    mmark.MarkerStyle("X"),
]
list_lab = ["All", "Arguably causal", "Causal", "Constant"]
list_color = [color_all, color_arguablycausal, color_causal, color_constant]


class MarkerHandler(HandlerBase):
    def create_artists(
        self, legend, tup, xdescent, ydescent, width, height, fontsize, trans
    ):
        return [
            plt.Line2D(
                [width / 2],
                [height / 2.0],
                ls="",
                marker=tup[1],
                markersize=markersize,
                color=tup[0],
                transform=trans,
            )
        ]

models = [
    "deepcoral", "lightgbm", "dann", "resnet", "causirl_mmd", "group", "and_mask",
    "mlp", "vrex", "ib_irm", "histgbm", "mixup", "label", "ib_erm", "aldro",
    "node", "mmd", "dro", "irm", "saint", "xgb", "causirl_coral", "tabtransformer", "ft",

]
# models = ['mlp', 'tableshift:mlp']

# Define list of experiments to plot
experiments = [
    "acsfoodstamps",
    "acsincome",
    "acspubcov",
    "acsunemployment",
    "anes",
    "assistments",
    "brfss_blood_pressure",
    "brfss_diabetes",
    "college_scorecard",
    "diabetes_readmission",
    "mimic_extract_mort_hosp",
    "mimic_extract_los_3",
    "nhanes_lead",
    "physionet",
    "meps",
    "sipp",
]

import pandas as pd
import matplotlib.pyplot as plt
eval_experiments = pd.DataFrame()

for index, experiment_name in enumerate(experiments):
    eval_all = get_results(experiment_name)
    eval_experiments = pd.concat([eval_experiments, eval_all])

    # eval_all["task"] = dic_title[experiment_name]

    # eval_plot = pd.DataFrame()
    # for feature_set in eval_all["features"].unique():
    #     eval_feature = eval_all[eval_all["features"] == feature_set]
        
    #     eval_feature = eval_feature[
    #         eval_feature["ood_test"] == eval_feature["ood_test"].max()
    #     ]
    #     eval_feature.drop_duplicates(inplace=True)
    #     eval_plot = pd.concat([eval_plot, eval_feature])

    # eval_experiments = pd.concat([eval_experiments, eval_plot])


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming eval_experiments is already created and populated
# Group by 'features' and 'model', and compute averages
# import pdb
# pdb.set_trace()
eval_experiments = eval_experiments[eval_experiments["model"].isin(models)]

average_results = (
    eval_experiments[eval_experiments["features"] != "constant"]  # Exclude 'constant'
    .groupby(["features", "model"])
    .agg(
        avg_id_test=("id_test", "mean"),
        avg_ood_test=("ood_test", "mean")
    )
    .reset_index()
)

# Get unique features (excluding 'constant')
unique_features = average_results["features"].unique()

# Define markers and colors for unique models
markers = [
    "o", "s", "^", "v", "<", ">", "d", "p", "*", "H",  "P", "X",  "D", "h", "s", "p", "*", "h",  "v", "<", ">", "o", "^", 
    "s", "d", "H", "X","o", "s", "^", "v", "<", ">", "d", "p", "*", "H", "P", "X",  "D", "h", "s", "p", "*", "h", "v", "<", ">", "o", "^", 
    "s", "d", "H", "X"
]

colors = sns.color_palette("tab20b", len(average_results["model"].unique()))

# Map markers and colors to models
unique_models = average_results["model"].unique()
model_styles = {model: (markers[i % len(markers)], colors[i % len(colors)]) for i, model in enumerate(unique_models)}

# Font and marker size settings
font_size = 22
marker_size = 300

# Create subplots for each unique feature
fig, axes = plt.subplots(1, len(unique_features), figsize=(7 * len(unique_features), 6), sharey=True)

# Plot data for each feature
for i, feature in enumerate(unique_features):
    ax = axes[i] if len(unique_features) > 1 else axes  # Handle single-plot case
    feature_data = average_results[average_results["features"] == feature]

    # Plot 'mlp' 
    if "mlp" in feature_data["model"].unique():
        mlp_data = feature_data[feature_data["model"] == "mlp"]
        marker, color = model_styles["mlp"]

        ax.scatter(
            mlp_data["avg_id_test"],
            mlp_data["avg_ood_test"],
            label="ERM",  # Capitalize 'mlp'
            marker=marker,
            color=color,
            alpha=0.8,
            edgecolor="black",
            s=marker_size  # Set marker size
        )
    if "tableshift:mlp" in feature_data["model"].unique():
        mlp_data = feature_data[feature_data["model"] == "tableshift:mlp"]
        marker, color = model_styles["tableshift:mlp"]

        ax.scatter(
            mlp_data["avg_id_test"],
            mlp_data["avg_ood_test"],
            label="ERM",  # Capitalize 'mlp'
            marker=marker,
            color=color,
            alpha=0.8,
            edgecolor="black",
            s=marker_size  # Set marker size
        )
    # Plot data points
    for model in feature_data["model"].unique():
        if model.lower() not in  ["mlp","tableshift:mlp"]:
            model_data = feature_data[feature_data["model"] == model]
            marker, color = model_styles[model]

            ax.scatter(
                model_data["avg_id_test"],
                model_data["avg_ood_test"],
                label=model.title(),
                marker=marker,
                color=color,
                alpha=0.8,
                edgecolor="black",
                s=marker_size  # Set marker size
            )

    
    x_min = min(average_results["avg_id_test"].min()-0.05, average_results["avg_ood_test"].min()-0.05)
    x_max = max(average_results["avg_id_test"].max()+0.05, average_results["avg_ood_test"].max()+0.05)
    y_min = x_min  # Match the minimum of x for consistency
    y_max = x_max  # Match the maximum of x for consistency


    # Set limits dynamically
    ax.set_xlim(x_min+0.05, x_max)
    ax.set_ylim(y_min, y_max-0.05)

    # Add a dashed line where X = Y
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
    
    # Linear fit
    x_vals = feature_data["avg_id_test"]
    y_vals = feature_data["avg_ood_test"]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

    # Plot the regression line
    line = slope * x_vals + intercept
    ax.plot(x_vals, line, color="red",linewidth=2)


    # Adjust the axis limits to include the diagonal line
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Set subplot-specific labels and titles
    ax.set_xlabel("Average In-Domain Accuracy", fontsize=font_size)
    if i == 0:  # Add y-label to the first subplot only
        ax.set_ylabel("Average Out-of-Domain Accuracy", fontsize=font_size)
    
    ft=feature.title() if feature != "arguablycausal" else "Arguably Causal"

    ax.set_title(f"{ft}"+" Variables", fontsize=font_size + 2)

    # Increase tick label font sizes
    ax.tick_params(axis="both", which="major", labelsize=font_size - 2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


# Create a common legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=8,
    title="Models",
    fontsize=font_size-4,
    title_fontsize=font_size,
)

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0.005, 1, 1])  # Reserve space at the bottom


fig.savefig(
    str(Path(__file__).parents[0] / f"allvariablesplots/allvariable_idood_regression.pdf"),
    bbox_inches="tight",
)


# eval_experiments = pd.DataFrame()
# for index, experiment_name in enumerate(experiments):
#     eval_all = get_results(experiment_name)
#     import pdb
#     pdb.set_trace()
#     eval_all["task"] = dic_title[experiment_name]

#     eval_plot = pd.DataFrame()
#     for set in eval_all["features"].unique():
#         eval_feature = eval_all[eval_all["features"] == set]
#         eval_feature = eval_feature[
#             eval_feature["ood_test"] == eval_feature["ood_test"].max()
#         ]
#         eval_feature.drop_duplicates(inplace=True)
#         eval_plot = pd.concat([eval_plot, eval_feature])
#     eval_experiments = pd.concat([eval_experiments, eval_plot])
#     dic_shift = {}
#     dic_shift_acc = {}


# fig = plt.figure(figsize=(6.75, 1.5))
# ax = fig.subplots(
#     1, 2, gridspec_kw={"width_ratios": [0.5, 0.5], "wspace": 0.3}
# )  # create 1x4 subplots on subfig1

# ax[0].set_xlabel(f"Tasks")
# ax[0].set_ylabel(f"Out-of-domain accuracy")

# #############################################################################
# # plot ood accuracy
# #############################################################################
# markers = {"constant": "X", "all": "s", "causal": "o", "arguablycausal": "D"}

# sets = list(eval_experiments["features"].unique())
# sets.sort()

# for index, set in enumerate(sets):
#     eval_plot_features = eval_experiments[eval_experiments["features"] == set]
#     eval_plot_features = eval_plot_features.sort_values("ood_test")
#     ax[0].errorbar(
#         x=eval_plot_features["task"],
#         y=eval_plot_features["ood_test"],
#         yerr=eval_plot_features["ood_test_ub"] - eval_plot_features["ood_test"],
#         color=eval(f"color_{set}"),
#         ecolor=color_error,
#         fmt=markers[set],
#         markersize=markersize,
#         capsize=capsize,
#         label=set.capitalize() if set != "arguablycausal" else "Arguably causal",
#         zorder=len(sets) - index,
#     )
#     # get pareto set for shift vs accuracy
#     shift_acc = eval_plot_features
#     shift_acc["type"] = set
#     shift_acc["gap"] = shift_acc["ood_test"] - shift_acc["id_test"]
#     shift_acc["id_test_var"] = ((shift_acc["id_test_ub"] - shift_acc["id_test"])) ** 2
#     shift_acc["ood_test_var"] = ((shift_acc["ood_test_ub"] - shift_acc["ood_test"])) ** 2
#     shift_acc["gap_var"] = shift_acc["id_test_var"] + shift_acc["ood_test_var"]
#     dic_shift_acc[set] = shift_acc

# ax[0].tick_params(axis="x", labelrotation=90)
# ax[0].set_ylim(top=1.0)
# ax[0].grid(axis="y")


# ax[1].set_xlabel(f"Tasks")
# ax[1].set_ylabel(f"Shift gap (higher is better)")
# #############################################################################
# # plot shift gap
# #############################################################################
# shift_acc = pd.concat(dic_shift_acc.values(), ignore_index=True)
# sets = list(eval_experiments["features"].unique())
# sets.sort()

# for index, set in enumerate(sets):
#     shift_acc_plot = shift_acc[shift_acc["features"] == set]
#     shift_acc_plot = shift_acc_plot.sort_values("ood_test")
#     ax[1].errorbar(
#         x=shift_acc_plot["task"],
#         y=shift_acc_plot["gap"],
#         yerr=shift_acc_plot["gap_var"] ** 0.5,
#         color=eval(f"color_{set}"),
#         ecolor=color_error,
#         fmt=markers[set],
#         markersize=markersize,
#         capsize=capsize,
#         label=set.capitalize() if set != "arguablycausal" else "Arguably causal",
#         zorder=len(sets) - index,
#     )

# ax[1].axhline(
#     y=0,
#     color="black",
#     linestyle="--",
# )
# ax[1].tick_params(axis="x", labelrotation=90)

# ax[1].grid(axis="y")

# list_mak.append("_")
# list_lab.append("Same performance")
# list_color.append("black")
# # plt.tight_layout()
# fig.legend(
#     list(zip(list_color, list_mak)),
#     list_lab,
#     handler_map={tuple: MarkerHandler()},
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.9),
#     fancybox=True,
#     ncol=5,
# )

# fig.savefig(
#     str(Path(__file__).parents[0] / f"plots_paper/allvariable_idood.pdf"),
#     bbox_inches="tight",
# )

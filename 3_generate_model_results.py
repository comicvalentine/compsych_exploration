# Generate results of interest for all models.
# idata (arviz) object is saved in "HBA_model/{model_name}/{run_name}/idata"
# summary of model diagnostics are saved as json file ("diagnostics.json") in "HBA_results/{model_name}_{run_name}"
# model comparison result is saved as HBA_results/model_comparison.csv and HBA_results/model_comparison.png
# Traceplots, KDE plots of mu and scatter plots comparing HBM with MAP estimates are saved in "HBA_results/{model_name}_{run_name}"

from cmdstanpy import from_csv
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import json
from shared import generate_idata, generate_group_trace_plot, get_diagnostics

def generate_mu_KDE(par, idata, fig_dir):

    colors = ["#FB2E46", "#1393FD"]

    mu_name = f"mu_{par}"
    posterior = idata.posterior[mu_name]

    if posterior.ndim != 3:
        fig, ax = plt.subplots(figsize=(4.5,4.5))

        samples = posterior.values.flatten()

        hdi = az.hdi(samples, prob=0.95)

        sns.kdeplot(
            samples,
            color=colors[0],
            ax=ax
        )
        
        x, y = ax.lines[-1].get_data()

        mask = (x >= hdi[0]) & (x <= hdi[1])

        ax.fill_between(
            x[mask],
            y[mask],
            alpha=0.25,
            color=colors[0]
        )
        
        ax.set_facecolor("#F9F9F9")
        ax.grid(color='#E2E2E2', linewidth=0.5)
 

        ax.set_title("Posterior distributions")
        ax.set_ylabel("Density")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.suptitle(
            rf"$\mu_{{{par}}}$",
            x=0.12,
            fontsize=18
        )

        fig.tight_layout()

        fig.savefig(
            f"{fig_dir}/{mu_name}_KDE.png",
            dpi=300,
            bbox_inches="tight"
        )

        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 4.5)
    )

    # left : short vs long

    ax = axes[0]

    for h_idx, label in enumerate(["short", "long"]):

        samples = posterior.sel({f"{mu_name}_dim_0": h_idx}).values.flatten()
        

        hdi = az.hdi(samples, prob=0.95)

        sns.kdeplot(
            samples,
            color=colors[h_idx],
            label=label,
            ax=ax
        )

        x, y = ax.lines[-1].get_data()

        mask = (x >= hdi[0]) & (x <= hdi[1])

        ax.fill_between(
            x[mask],
            y[mask],
            alpha=0.25,
            color=colors[h_idx]
        )

    ax.set_title("Posterior distributions")
    ax.set_ylabel("Density")
    ax.legend(fontsize=15, frameon=False)

    # right : delta (long - short)

    ax = axes[1]

    diff_samples = (
        posterior.sel({f"{mu_name}_dim_0": 1}).values.flatten()
        - posterior.sel({f"{mu_name}_dim_0": 0}).values.flatten()
    )

    hdi = az.hdi(diff_samples, prob=0.95)
    prob = np.mean(diff_samples>0)

    sns.kdeplot(
        diff_samples,
        color="#6A6A6A",
        ax=ax
    )

    x, y = ax.lines[-1].get_data()

    mask = (x >= hdi[0]) & (x <= hdi[1])

    ax.fill_between(
        x[mask],
        y[mask],
        alpha=0.25,
        color="#6A6A6A"
    )

    delta_hdi = rf"HDI 95%: [{hdi[0]:.2f}, {hdi[1]:.2f}]"
    p_long_short = rf"P(Long > Short): {prob:.3f}"

    if par == "beta":
        # The smaller beta means more stochastic (explorative) choice.
        p_long_short = rf"P(Long < Short): {1-prob:.3f}"

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_title(f"Long - Short \n {delta_hdi} \n {p_long_short}")

    # visual

    for ax in axes:

        ax.set_facecolor("#F9F9F9")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(color="#E2E2E2", linewidth=0.5)

    fig.suptitle(rf"$\mu_{{\{par}}}$", x=0.12,fontsize=30)

    fig.tight_layout()

    fig.savefig(f"{fig_dir}/{par}_KDE.png", dpi=300, bbox_inches="tight")

    plt.close()

def generate_scatter_vs_MAP(par, lower, upper, idata, map_estimates, fig_dir, prob=0.95):
    
    posterior = idata.posterior[par]

    if par != "Q_0":
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,4.5))
        
        for h_idx, horizon in enumerate(["short", "long"]):
            x = map_estimates[f"{par}_{horizon}"]
            
            samples = posterior.sel({f"{par}_dim_0":h_idx})
            y = samples.mean(axis=(0,1))

            hdi = az.hdi(samples, prob=prob, var_names=par)
            hdi_lower = hdi[:, 0]
            hdi_upper = hdi[:, 1]

            ax = axes[h_idx]
            ax.scatter(
                       x=x,
                       y=y,
                       s=10, 
                       facecolors='none',
                       edgecolors=["#FB2E46", "#1393FD"][h_idx]
                       )

            ax.errorbar(
                    x=x,
                    y=y,
                    yerr=np.vstack([y - hdi_lower, hdi_upper - y ]),
                    fmt="none",
                    ecolor=["#FB2E46", "#1393FD"][h_idx],
                    alpha=0.25,
                    elinewidth=0.6,
                    capsize=0,
                    zorder=1
                )

            ax.set_xlim(xmin=lower[h_idx]-0.1*(lower[h_idx]+upper[h_idx]),
                        xmax=upper[h_idx]+0.1*(lower[h_idx]+upper[h_idx]))
            ax.set_ylim(ymin=lower[h_idx]-0.1*(lower[h_idx]+upper[h_idx]),
                        ymax=upper[h_idx]+0.1*(lower[h_idx]+upper[h_idx]))
            
            ax.plot(np.arange(lower[h_idx], upper[h_idx], (upper[h_idx]-lower[h_idx])/20),
                    np.arange(lower[h_idx], upper[h_idx], (upper[h_idx]-lower[h_idx])/20),
                    color="#6A6A6A",
                    linestyle="--")
            
            ax.grid(color='#E2E2E2', linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_title(horizon, fontsize=15)

            ax.set_xlabel("MAP estimates", fontsize=15)
            ax.set_ylabel("Posterior means", fontsize=15)
            ax.set_facecolor("#F9F9F9")
            
        title = rf"\{par}"

    else:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        x = map_estimates[par]
        y = posterior.mean(dim=["chain", "draw"]).values
        hdi = az.hdi(posterior, prob=prob, var_names=par)
        hdi_lower = hdi[:, 0]
        hdi_upper = hdi[:, 1]

        ax.scatter(x=x,
                   y=y,
                   s=10,
                   facecolors='none',
                   edgecolors="#FB2E46")
        
        ax.errorbar(
                x=x,
                y=y,
                yerr=np.vstack([y - hdi_lower, hdi_upper - y ]),
                fmt="none",
                ecolor="#FB2E46",
                alpha=0.25,
                elinewidth=0.6,
                capsize=0,
                zorder=1
            )
        
        ax.set_xlim(xmin=lower-0.1*(lower+upper),
                    xmax=upper+0.1*(lower+upper))
        ax.set_ylim(ymin=lower-0.1*(lower+upper),
                    ymax=upper+0.1*(lower+upper))
        
        ax.plot(np.arange(lower, upper, (upper-lower)/20),
                np.arange(lower, upper, (upper-lower)/20),
                color="#6A6A6A",
                linestyle="--")

        ax.grid(color='#E2E2E2', linewidth=0.5, zorder=10000)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("MAP estimates", fontsize=15)
        ax.set_ylabel("Posterior means", fontsize=15)

        ax.set_facecolor("#F9F9F9")

        title = par

    fig.suptitle(t=rf"${title}$", x=0.12, fontsize=30)
    fig.tight_layout()

    fig.savefig(f"{fig_dir}/{par}_vs_map.png", dpi=300, bbox_inches="tight")
    plt.close()

def generate_model_comp_plot(model_comp, fig_dir):
    
    model_names = {
    "UCB_beta_epsilon_eta": r"UCB ($\beta,\ \eta,\ \epsilon$)",
    "UCB_beta_eta": r"UCB ($\beta, \eta$)",
    "UCB_beta_epsilon": r"UCB ($\beta,\ \epsilon$)",
    "UCB_beta": r"UCB ($\beta$)",
    "Thompson_epsilon_eta": r"TS ($\eta,\ \epsilon$)",
    "Thompson_eta": r"TS ($\eta$)",
    "Thompson_epsilon": r"TS ($\epsilon$)",
    "Thompson": r"TS"
    }

    n_models = len(model_names)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)
    ax_loo = axes[0]
    ax_loo.bar(x=range(n_models), height = model_comp["looic"], yerr=model_comp["se"], color="#1393FD")
    ax_loo.set_ylim(ymin = model_comp["looic"].min()-0.5*model_comp["looic"].std(), 
                    ymax = model_comp["looic"].max()+0.5*model_comp["looic"].std())
    
    ax_loo.set_xticks(range(n_models))
    ax_loo.set_xticklabels([model_names[model] for model in model_comp["model"]], rotation=30)
    ax_loo.set_ylabel("LOOIC", fontsize=12)
    ax_loo.set_title("HBA", fontsize=15)

    ax_loo.spines["top"].set_visible(False)
    ax_loo.spines["right"].set_visible(False)
    ax_loo.set_facecolor(color='#F9F9F9')
    ax_loo.set_axisbelow(True)
    ax_loo.grid(axis="y", color='#E2E2E2', linewidth=0.5)

    ax_bic = axes[1]
    ax_bic.bar(x=range(n_models), height = model_comp["map_BIC"], color="#FB2E46")
    ax_bic.set_ylim(ymin = model_comp["map_BIC"].min()-0.5*model_comp["map_BIC"].std(), 
                    ymax = model_comp["map_BIC"].max()+0.5*model_comp["map_BIC"].std())
    ax_bic.set_xticks(range(n_models))
    ax_bic.set_xticklabels([model_names[model] for model in model_comp["model"]], rotation=30)
    ax_bic.set_ylabel("BIC", fontsize=12)
    ax_bic.set_title("MAP", fontsize=15)
    
    ax_bic.spines["top"].set_visible(False)
    ax_bic.spines["right"].set_visible(False)
    ax_bic.set_facecolor(color='#F9F9F9')
    ax_bic.set_axisbelow(True)
    ax_bic.grid(axis="y", color='#E2E2E2', linewidth=0.5)

    fig.suptitle("Model Comparison", fontsize=20, x=0.2, y=0.99)
    fig.tight_layout()
    fig.savefig(fig_dir, dpi=300)
    plt.close()

    return fig, axes

params_lim = {"Q_0": [1, 8],
              "sigma_0": [[0.01, 0.01], [3, 3]],
              "gamma": [[0, 0], [1, 3]],
              "beta": [[0.5, 0.5], [5, 5]],
              "eta": [[0, 0], [5, 5]],
              "epsilon": [[0, 0], [0.5, 0.5]]}

if __name__ == "__main__":
    
    model_dict = dict(
                  UCB_beta_epsilon_eta = ["Q_0", "gamma", "beta", "eta", "epsilon"],
                  UCB_beta_eta = ["Q_0", "gamma", "beta", "eta"],
                  UCB_beta_epsilon = ["Q_0", "gamma", "beta", "epsilon"],
                  UCB_beta = ["Q_0", "gamma", "beta"],
                  Thompson_epsilon_eta = ["Q_0", "sigma_0", "eta", "epsilon"],
                  Thompson_eta = ["Q_0", "sigma_0", "eta"],
                  Thompson_epsilon = ["Q_0", "sigma_0", "epsilon"],
                  Thompson =["Q_0", "sigma_0"]
                  )
    model_list = list(model_dict.keys())

    run_name = "100"
    n_subject = 100
    
    model_comp_path = Path("./HBA_results/model_comparison.csv")
    if model_comp_path.exists():
        model_comp = pd.read_csv(model_comp_path, index_col=0)
    else:
        model_comp = pd.DataFrame(columns=["model", "elpd", "se", "p", "looic", "map_BIC"])

    map_BIC = pd.read_csv("./MAP_results/model_comparison/BIC_all.csv").iloc[:n_subject, :].mean(axis=0)
    map_BIC_std = pd.read_csv("./MAP_results/model_comparison/BIC_all.csv").iloc[:n_subject, :].std(axis=0)
    
    for model_name, params in model_dict.items():
        MODEL_DIR = f"HBA_model/{model_name}/{run_name}"
        SAVE_DIR = f"./HBA_results/{model_name}_{run_name}"
        os.makedirs(SAVE_DIR, exist_ok=True)

        idata = generate_idata(MODEL_DIR)

        generate_group_trace_plot(idata, params, save_fig = f"{SAVE_DIR}/trace.png")

        diag = get_diagnostics(idata)
        with open(f"{SAVE_DIR}/diagnostics.json", "w") as f:
                json.dump(diag, f, indent=4)
            
        if model_name not in model_comp["model"].values:
            loo = az.loo(idata)
            
            model_comp.loc[len(model_comp)] = {
                                        "model": model_name,
                                        "elpd": np.round(loo.elpd,2),
                                        "se": np.round(loo.se,3),
                                        "p": np.round(loo.p,2),
                                        "looic": np.round(-2 * loo.elpd,2),
                                        "map_BIC": np.round(map_BIC[model_name], 2)
                                        }

        ## compare posterior mean for each parameters with MAP estimates
        map_estimates = pd.read_csv(f"MAP_results/parameter_estimates/{model_name}.csv").iloc[:n_subject, :]

        for par in params:
            generate_mu_KDE(par, idata, SAVE_DIR)

            lower = params_lim[par][0]
            upper = params_lim[par][1]

            generate_scatter_vs_MAP(par, lower, upper, idata, map_estimates, SAVE_DIR)
        
    model_comp.to_csv("./HBA_results/model_comparison.csv")
    generate_model_comp_plot(model_comp = model_comp, fig_dir=f"./HBA_results//model_comparison.png")
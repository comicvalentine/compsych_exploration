# Generate result of parameter recovery.
# idata (arviz) object is saved in "HBA_model/{model_name}/{run_name}/parameter_recovery/idata"
# summary of model diagnostics are saved as json file ("diagnostics_recovery.json") in "HBA_results/{model_name}_{run_name}"
# Traceplots and scatter plots comparing recovered parameter with true parameters are saved in "HBA_results/{model_name}_{run_name}" 

import arviz as az
import numpy as np
import os
from pathlib import Path
from cmdstanpy import from_csv
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import json
from shared import generate_idata, generate_group_trace_plot, get_diagnostics

def generate_scatter_vs_true_params(par, lower, upper, idata, true_params, fig_dir, prob=0.95):

    posterior = idata.posterior[par]

    if par != "Q_0":
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,4.5))

        for h_idx, horizon in enumerate(["short", "long"]):
            x = true_params[par][h_idx, :]
            
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
            
            slope = linregress(x=x, y=y).slope
            cor = pearsonr(x=x, y=y)[0]
            in_hdi = np.mean((x < hdi_upper)&(x > hdi_lower)).values*100
            
            ax.set_title(f"{horizon} \nslope = {slope:.2f}, r = {cor:.2f}\nHDI coverage: {in_hdi}%", fontsize=15)

            ax.set_xlabel("True parameter", fontsize=15)
            ax.set_ylabel("Recovered parameter", fontsize=15)

            ax.set_facecolor("#F9F9F9")
        
        title = rf"\{par}"

    else:
        x = true_params[par]
        y = posterior.mean(dim=["chain", "draw"]).values
        hdi = az.hdi(posterior, prob=prob, var_names=par)
        hdi_lower = hdi[:, 0]
        hdi_upper = hdi[:, 1]

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
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

        ax.grid(color='#E2E2E2', linewidth=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("True parameter", fontsize=15)
        ax.set_ylabel("Recovered parameter", fontsize=15)

        ax.set_facecolor("#F9F9F9")

        slope = linregress(x=x, y=y).slope
        cor = pearsonr(x=x, y=y)[0]

        in_hdi = np.mean((x < hdi_upper)&(x > hdi_lower)).values*100
        
        ax.set_title(f"slope = {slope:.2f}, r = {cor:.2f}\nHDI coverage: {in_hdi}%", fontsize=15)

        title = par

    fig.suptitle(t=rf"${title}$", x=0.12, fontsize=30)
    fig.tight_layout()

    fig.savefig(f"{fig_dir}/{par}_recovery.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)

    run_name = "100"
    
    model_dict = dict(
                # NOTE: The order of parameter list should be matched with that of stan for "mu_p index"
                UCB_beta_epsilon_eta = ["Q_0", "gamma", "beta", "eta", "epsilon"],
                Thompson_epsilon_eta = ["Q_0", "sigma_0", "eta", "epsilon"],
                # UCB_beta_epsilon = ["Q_0", "gamma", "beta", "epsilon"],
                # UCB_beta = ["Q_0", "gamma", "beta"],
                # Thompson_epsilon = ["Q_0", "sigma_0", "epsilon"],
                # Thompson =["Q_0", "sigma_0"]
                )
    
    params_lim = {"Q_0": [1, 8],
              "sigma_0": [[0.01, 0.01], [3, 3]],
              "gamma": [[0, 0], [0.3, 3]],
              "beta": [[0.5, 0.5], [5, 5]],
              "eta": [[0, 0], [5, 5]],
              "epsilon": [[0, 0], [0.5, 0.5]]}

    
    diag_path = Path("./HBA_results/model_diagnostics_recovery.json")
    if diag_path.exists():
        with open(diag_path, "r") as f:
            diag_dict = json.load(f)
    else:
        diag_dict = {}
    
    for model_name, params in model_dict.items():
        SAVE_DIR = f"./HBA_results/{model_name}_{run_name}"
        MODEL_DIR = f"./HBA_model/{model_name}/{run_name}/parameter_recovery"
        idata = generate_idata(MODEL_DIR)

        diag = get_diagnostics(idata)
        with open(f"{SAVE_DIR}/diagnostics_recovery.json", "w") as f:
                json.dump(diag, f, indent=4)

        generate_group_trace_plot(idata, params, save_fig = f"{SAVE_DIR}/trace_recovery.png")

        true_params = dict(np.load(f"./HBA_model/{model_name}/{run_name}/parameter_recovery/true_params.npz"))

        for par in true_params:
            generate_scatter_vs_true_params(par, 
                                            lower=params_lim[par][0], 
                                            upper=params_lim[par][1], 
                                            idata=idata, 
                                            true_params=true_params, 
                                            fig_dir = SAVE_DIR)

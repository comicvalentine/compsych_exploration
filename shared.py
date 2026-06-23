import arviz as az
import numpy as np
from cmdstanpy import from_csv
import os
from pathlib import Path
import matplotlib.pyplot as plt
import re

def load_data(n_subject):
    r_sum = np.load("./data/concat_data/r_sum.npy")
    n_obs = np.load("./data/concat_data/n_obs.npy")
    novelty = np.load("./data/concat_data/novelty.npy")
    fst_chos_bandit = np.load("./data/concat_data/fst_chos_bandit.npy")

    if n_subject != "full":
        r_sum = r_sum[:, :n_subject, :, :]
        n_obs = n_obs[:, :n_subject, :, :]
        novelty = novelty[:, :n_subject, :, :]
        fst_chos_bandit = fst_chos_bandit[:, :n_subject, :] 

    fst_chos_bandit = fst_chos_bandit[:, :n_subject, :] 

    nH, nS, nBT = fst_chos_bandit.shape[:]

    data = {"nH":nH, "nS":nS, "nBT":nBT, 
        "r_sum": r_sum, "n_obs": n_obs, "novelty": novelty, "fst_chos": fst_chos_bandit}
    return data

def generate_idata(MODEL_DIR):

    idata_path = Path(f"{MODEL_DIR}/idata/model_fit.nc")
    
    if idata_path.exists():
        idata = az.from_netcdf(idata_path)
        return idata
    
    fit = from_csv(f"{MODEL_DIR}/fit")

    save_dir = f"{MODEL_DIR}/idata"
    os.makedirs(save_dir, exist_ok=True)

    idata = az.from_cmdstanpy(
        fit,
        log_likelihood="log_lik"
    )

    idata.to_netcdf(f"{save_dir}/model_fit.nc")
    
    return idata


def generate_group_trace_plot(idata, params, save_fig=None):

    trace_vars = [f"mu_{par}" for par in params]
    trace_vars.append("sigma_Q_0")
    trace_vars.append("sigma")

    az.plot_trace(idata, var_names=trace_vars)

    for ax in plt.gcf().axes:

        title = ax.get_title()

        m = re.match(r"^(mu_[^\n]+)\n([01])$", title)

        if m:

            par_name = m.group(1)
            horizon = {"0": "short", "1": "long"}[m.group(2)]

            ax.set_title(f"{par_name}\n{horizon}")
            continue

        m = re.match(r"^sigma\n([01]),\s*([0-9]+)$", title)

        if m:

            horizon = {"0": "short", "1": "long"}[m.group(1)]

            param_idx = int(m.group(2))+1
            param_name = params[param_idx]

            ax.set_title(f"sigma_raw_{param_name}\n{horizon}")

    plt.tight_layout()

    if save_fig is not None:
        plt.savefig(save_fig)
        plt.close()

def get_diagnostics(idata, rhat_thresh=1.01):
    """
    Function computing n_divergence and rhat.
    Much faster than az.diagnose but ESS and BFMI are not computed. 
    Produce a dictionary that can be saved as json format.
    """

    diag = {}

    n_divergent = int(idata.sample_stats["diverging"].sum().values)
    diag["n_divergent"] = n_divergent

    rhat_ds = az.rhat(idata)

    diag["max_rhat"] = float(
        np.round(np.nanmax([np.nanmax(rhat_ds[var].values) for var in rhat_ds.data_vars]), 4)
    )

    bad_rhat = {}

    for var in rhat_ds.data_vars:

        values = rhat_ds[var].values

        # scalar parameter
        if np.ndim(values) == 0:
            if values > rhat_thresh:
                bad_rhat[var] = {"rhat": float(np.round(values, 4))}
            continue

        idxs = np.argwhere(values > rhat_thresh)

        if len(idxs) == 0:
            continue

        bad_rhat[var] = {}
        bad_rhat[var]["list"] = []

        for idx in idxs:
            idx = tuple((int(i) for i in idx))
            bad_rhat[var]["list"].append({"index": idx, "rhat": float(np.round(values[idx], 4))})
        bad_rhat[var]["count"] = len(bad_rhat[var]["list"])

    diag["bad_rhat"] = bad_rhat

    return diag
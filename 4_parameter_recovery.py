# Conduct parameter recovery
# 1. generate participant-level parameters using estimated mu and sigma
# 2. simulate behavioral data using generated parameters 
# => Simulated behavioral data and parameters are saved in "HBA_model/{model_name}/{run_name}/parameter_recovery/" as numpy files.
# 3. re-fit the model to simulated data 
# => the object of re-fit is saved in "HBA_model/{model_name}/{run_name}/parameter_recovery/fit"

import arviz as az
import pandas as pd
import numpy as np
import os
from scipy.special import softmax
from scipy.stats import truncnorm, norm, multivariate_normal
from cmdstanpy import CmdStanModel, from_csv
import matplotlib.pyplot as plt
from shared import load_data

# Phi_approx function matching with stan code
PARAM_TRANSFORMS = {
    "gamma": lambda x: norm.cdf(x) * 3.0,
    "beta": lambda x: norm.cdf(x) * 4.5 + 0.5,
    "eta": lambda x: norm.cdf(x) * 5.0,
    "epsilon": lambda x: norm.cdf(x) * 0.5,
    "sigma_0": lambda x: norm.cdf(x)*5.99 + 0.01,
    "Q_0": lambda x: x
}


model_dict = dict(
            # NOTE: The order of parameter list should be matched with that of stan for "mu_p index"
            UCB_beta_epsilon_eta = ["Q_0", "gamma", "beta", "eta", "epsilon"],
            Thompson_epsilon_eta = ["Q_0", "sigma_0", "eta", "epsilon"],
            # UCB_beta_epsilon = ["Q_0", "gamma", "beta", "epsilon"],
            # UCB_beta = ["Q_0", "gamma", "beta"],
            # Thompson_epsilon = ["Q_0", "sigma_0", "epsilon"],
            # Thompson =["Q_0", "sigma_0"]
            )

def generate_indiv_params(param, group_params, n_subject):
    
    if param == "Q_0":
        lower_bound = 1.0
        upper_bound = 10.0
        
        mu = group_params["mu_Q_0"]
        sigma = group_params["sigma_Q_0"]
        
        # "standardized bound" which is required by scipy.stats.truncnorm
        a = (lower_bound - mu) / sigma
        b = (upper_bound - mu) / sigma
        
        # sampling from truncated normal just like stan code
        actual_samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_subject)
        
        return {"raw": actual_samples, "actual": actual_samples}
    
    else:
        
        raw_samples = np.random.normal(loc=group_params[f"mu_{param}_raw"].reshape((2,1)),
                                        scale=group_params[f"sigma_{param}_raw"].reshape((2,1)),
                                        size=(2, n_subject))
    
        actual_samples = PARAM_TRANSFORMS[param](raw_samples)
        return {"raw": raw_samples, "actual": actual_samples}

def get_estimated_group_params(idata, param_list):
    p_idx = 0
    group_params = {}
    for param in param_list:
        if param == "Q_0":
            group_params[f"mu_{param}"] = idata.posterior["mu_Q_0"].mean(dim=["chain", "draw"]).values
            group_params[f"sigma_{param}"] = idata.posterior["sigma_Q_0"].mean(dim=["chain", "draw"]).values
        else:
            # 1. separate posterior meas for each horizon
            # 2. raw parameter before transformed by Phi was saved as vectorized mu_p
            group_params[f"mu_{param}_raw"] = idata.posterior["mu_p"].mean(dim=["chain", "draw"]).values[:, p_idx]
            group_params[f"sigma_{param}_raw"] = idata.posterior["sigma"].mean(dim=["chain", "draw"]).values[:, p_idx]
            
            p_idx += 1
    return group_params

def kalman_filter(Q_0, tau_0, tau_S, n_obs, r_sum):
    tau_n = tau_0 + n_obs * tau_S
    Q_n = ((tau_0 * Q_0 + tau_S * r_sum)/(tau_0 + tau_S * n_obs))
    sigma_n = 1/np.sqrt(tau_n)
    return Q_n, sigma_n               

def simulate_UCB_model(param_list, n_subject, data, group_params):

    n_blocktrial = data["fst_chos"].shape[2]

    param_dict = {}

    for param in param_list:

        param_dict[param] = generate_indiv_params(param, group_params, n_subject)["actual"]
    
    
    sigma_0 = 1.5
    tau_0 = 1/(sigma_0**2)
    sigma_S = 0.8
    tau_S = 1/(sigma_S**2)

    simulated_chos_bandit = np.zeros(shape=(2, n_subject, n_blocktrial), dtype=int)

    for h_idx in range(2):
        for s_idx in range(n_subject):
            for bt_idx in range(n_blocktrial):
                # compute Q and sigma at the first choice trial using closed-form kalman filter
                Q_n, sigma_n = kalman_filter(param_dict["Q_0"][s_idx], 
                                             tau_0, tau_S, 
                                             data["n_obs"][h_idx, s_idx, bt_idx, :], 
                                             data["r_sum"][h_idx, s_idx, bt_idx, :])

                # compute V
                if "eta" in param_list:
                    V_n = Q_n + param_dict["gamma"][h_idx, s_idx] * sigma_n + param_dict["eta"][h_idx, s_idx] * data["novelty"][h_idx, s_idx, bt_idx, :]
                else:
                    V_n = Q_n + param_dict["gamma"][h_idx, s_idx] * sigma_n

                # Probability: value-based(softmax with inverse temperature) + value-free(epsilon)
                P_softmax = softmax(param_dict["beta"][h_idx, s_idx]*V_n)
                if "epsilon" in param_list:
                    P_final = (1-param_dict["epsilon"][h_idx, s_idx])*P_softmax + (param_dict["epsilon"][h_idx, s_idx]/3.0)
                else:
                    P_final = P_softmax
                choice = np.random.choice(3, p=P_final)
                simulated_chos_bandit[h_idx, s_idx, bt_idx] = choice+1
    
    return simulated_chos_bandit, param_dict

def simulate_Thompson_model(param_list, n_subject, data, group_params):

    n_blocktrial = data["fst_chos"].shape[2]

    param_dict = {}

    for param in param_list:

        param_dict[param] = generate_indiv_params(param, group_params, n_subject)["actual"]
    
    
    sigma_S = 0.8
    tau_S = 1/(sigma_S**2)

    simulated_chos_bandit = np.zeros(shape=(2, n_subject, n_blocktrial), dtype=int)

    for h_idx in range(2):
        for s_idx in range(n_subject):
            for bt_idx in range(n_blocktrial):
                # compute Q and sigma at the first choice trial using closed-form kalman filter
                tau_0 = 1/(param_dict["sigma_0"][h_idx, s_idx]**2)
                Q_n, sigma_n = kalman_filter(param_dict["Q_0"][s_idx], 
                                             tau_0, tau_S, 
                                             data["n_obs"][h_idx, s_idx, bt_idx, :], 
                                             data["r_sum"][h_idx, s_idx, bt_idx, :])

                # compute V
                if "eta" in param_list:
                    V_n = Q_n + param_dict["eta"][h_idx, s_idx] * data["novelty"][h_idx, s_idx, bt_idx, :]
                else:
                    V_n = Q_n

                # Probability:Thompson sampling - directly sample for each bandit!
                choice = np.argmax(
                    np.random.normal(
                        loc=V_n,
                        scale=sigma_n
                    )
                )

                if "epsilon" in param_list:
                    if np.random.rand() < param_dict["epsilon"][h_idx, s_idx]: #value-free random choice for epsilon probability
                        choice = np.random.randint(3)

                simulated_chos_bandit[h_idx, s_idx, bt_idx] = choice + 1
                
    return simulated_chos_bandit, param_dict


def parameter_recovery(model_name, run_name, n_subject, 
                       iter_warmup=1000, iter_sampling=1000, 
                       chains=4, parallel_chains = None, seed=42):
    
    data = load_data(n_subject)

    MODEL_DIR = f"./HBA_model/{model_name}/{run_name}"
    REC_DIR = f"{MODEL_DIR}/parameter_recovery"
 
    idata = az.from_netcdf(f"{MODEL_DIR}/idata/model_fit.nc")
    
    group_params = get_estimated_group_params(idata, param_list = model_dict[model_name])
    
    if "UCB" in model_name:
        simulated_chos_bandit, param_dict = simulate_UCB_model(param_list = model_dict[model_name],
                                                               n_subject = n_subject,
                                                               data = data,
                                                               group_params=group_params)

    elif "Thompson" in model_name:
        simulated_chos_bandit, param_dict = simulate_Thompson_model(param_list = model_dict[model_name],
                                                               n_subject = n_subject,
                                                               data = data,
                                                               group_params=group_params)

    # replace real choice data to simulated data
    data["fst_chos"] = simulated_chos_bandit

    os.makedirs(f"{REC_DIR}", exist_ok=True)
    np.save(f"{REC_DIR}/sim_chos.npy", simulated_chos_bandit)
    np.savez(
        f"{REC_DIR}/true_params.npz",
        **param_dict
    )

    stan_file = f"./HBA_stan/{model_name}.stan"
    model = CmdStanModel(stan_file=stan_file)
    save_dir = f"{REC_DIR}/fit"


    fit = model.sample(
                    data=data,
                    iter_warmup=iter_warmup,
                    iter_sampling=iter_sampling,
                    chains=chains,
                    parallel_chains=parallel_chains,
                    output_dir = save_dir,
                    # show_console = True,
                    show_progress = True,
                    refresh=1,
                    seed=seed
                    )

    return fit

if __name__ == "__main__":
    RANDOM_SEED = 42
    
    np.random.seed(RANDOM_SEED)

    run_name = "100"
    n_subject = 100
    
    data = load_data(n_subject)

    for model_name in model_dict:
        fit, = parameter_recovery(model_name, run_name, n_subject,
                                  iter_warmup=1000, iter_sampling=1000, chains=4, seed=RANDOM_SEED)
        
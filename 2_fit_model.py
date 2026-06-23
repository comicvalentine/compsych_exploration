# Fit hierarchical Bayesian model using stan files in "HBA_stan" folder.
# The output fit object is saved in "./HBA_model/{model_name}/{run_name}/fit".

import numpy as np
import os
from cmdstanpy import CmdStanModel
import arviz as az
from shared import load_data

def fit_HBA_model(model_name, 
                  run_name,
                  adapt_delta = 0.8,
                  n_subject = "full", 
                  iter_warmup=None, 
                  iter_sampling=None, 
                  chains=4, 
                  parallel_chains=None,
                  seed=42):
    
    data = load_data(n_subject)
    stan_file = f"./HBA_stan/{model_name}.stan"
    model = CmdStanModel(stan_file=stan_file)

    save_dir = f"HBA_model/{model_name}/{run_name}/fit"
    os.makedirs(f"{save_dir}", exist_ok=True)

    fit = model.sample(
                    data=data,
                    iter_warmup=iter_warmup,
                    iter_sampling=iter_sampling,
                    adapt_delta=adapt_delta,
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
    model_list = [
                  "UCB_beta_epsilon_eta",
                  "UCB_beta_epsilon",
                  "UCB_beta_eta",
                  "UCB_beta",
                  "Thompson_epsilon_eta",
                  "Thompson_epsilon",
                  "Thompson_eta"
                  "Thompson"
                  ]

    for model_name in model_list:
        fit = fit_HBA_model(model_name=model_name,
                            run_name = "100",
                            n_subject = 100,
                            iter_warmup= 1000, 
                            iter_sampling= 1000,
                            chains = 4,
                            seed=42)

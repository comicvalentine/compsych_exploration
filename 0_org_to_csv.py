# Convert original mat files (stored in /org/MFweb-data) into csv format, which is more python-friendly
# files: raw data file, map parameter estiamtes for all subjects, BIC for all subjects

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

def convert_user_mat_to_csv(user_id):
    file = f"org/MFweb-data/raw/user_{user_id}.mat"
    raw_data = loadmat(file)
    columns = [arr[0] for arr in raw_data["user"][0][0][1][0]]
    data = raw_data["user"][0][0][0]
    user_df = pd.DataFrame(data=data, columns = columns)

    os.makedirs("./data/raw", exist_ok=True)
    user_df.to_csv(f"./data/raw/user_{user_id}.csv")

    return user_df

def convert_MAP_mat_to_csv(model_id, model_name, user_id_list):
    param_dir = f"./org/MFweb-data/data_for_figs/model_parameters{model_id}.mat"
    desc_dir = f"./org/MFweb-data/data_for_figs/model_parameters{model_id}_desc.mat"

    params = loadmat(param_dir)["model_parameters"]
    desc = loadmat(desc_dir)["model_parameters_desc"]
    columns = [str(p[0]) for p in desc[0]]

    params_df = pd.DataFrame(data=params, columns = columns)
    params_df = params_df[params_df["ID"].isin(user_id_list)]
    
    # tau -> beta
    if "UCB" in model_name:

        tau_cols = [c for c in params_df.columns if "tau" in c]

        if len(tau_cols) == 0:
            params_df["tau_long"] = 1.0
            params_df["tau_short"] = 1.0

    for c in [c for c in params_df.columns if "tau" in c]:
        params_df[c] = 1 / params_df[c]

    # rename
    params_df = params_df.rename(
                            columns=lambda x: x.replace("xi", "epsilon")
                                                .replace("tau", "beta")
                                                .replace("Q0", "Q_0")
                                                .replace("sgm0", "sigma_0")
                                )

    os.makedirs("./MAP_results/parameter_estimates", exist_ok=True)
    params_df.to_csv(f"./MAP_results/parameter_estimates/{model_name}.csv")

    return params_df

def convert_BIC_mat_to_csv(user_id_list):
    BIC = loadmat("./org/MFweb-data/data_for_figs/BIC_all.mat")["BIC_all"]
    desc = loadmat("./org/MFweb-data/data_for_figs/BIC_all_desc.mat")["BIC_all_desc"]
    columns = [str(p[0]) for p in desc[0]]
    log = pd.read_csv("org/MFweb-data/data_collection.log", sep="\t", encoding="utf-16")

    BIC_df = pd.DataFrame(data=BIC, columns = columns)
    BIC_df["ID"] = log["user"]
    BIC_df = BIC_df[BIC_df["ID"].isin(user_id_list)]
    BIC_df = BIC_df.rename(
                            columns=lambda x: x.replace("eps", "epsilon")
                                            .replace("BIC_", "")
                                            .replace("UCB", "UCB_beta")
                                            .replace("UCB_beta_b1", "UCB_b1")
                                            .replace("thompson", "Thompson")
                                )

    os.makedirs("./MAP_results/model_comparison", exist_ok=True)
    BIC_df.to_csv(f"./MAP_results/model_comparison/BIC_all.csv")

    return BIC_df


if __name__ == "__main__":
    log = pd.read_csv("org/MFweb-data/data_collection.log", sep="\t", encoding="utf-16")
    user_id_list = [row["user"] for idx, row in log.iterrows() if row["exclude"]==0]
    
    for s_idx, user_id in enumerate(user_id_list): #(user_id(1~792) with "jump" to subject_id (0~580) without "jump")
            subject_df = convert_user_mat_to_csv(user_id)

    log.to_csv("data/data_collection.log")

    # NOTE: 
    # Except for full UCB model (UCB_beta_epsilon_eta), 
    # for other UCB models there are no parameter estimates of the models leaving beta as free-parameter. 
    # In other words, corresponding MAP model for "UCB_beta_eta", "UCB_beta_epsilon", and "UCB_beta" are in fact the model beta is fixed as 1.
    
    # TODO 
    # Run the matlab code for the models with free-beta, and save them as "mod5", "mod6", "mod7"
        
    model_name_to_id = {"UCB_beta_epsilon_eta": "_mod8",
                        "UCB_beta_eta": "_mod7_t1",
                        "UCB_beta_epsilon": "_mod6_t1",
                        "UCB_beta": "_mod5_t1",
                        "Thompson_epsilon_eta": "",
                        "Thompson_eta": "_mod11",
                        "Thompson_epsilon": "_mod10",
                        "Thompson": "_mod9"}
    
    for model_name, model_id in model_name_to_id.items():
         convert_MAP_mat_to_csv(model_id, model_name, user_id_list)
    
    convert_BIC_mat_to_csv(user_id_list)
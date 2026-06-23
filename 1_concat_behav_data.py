# Preprocess raw data into array, which can be directly employed in Stan.
# The output is saved in "./data/concat_data" as npy files.

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

def delete_unshown_option(all_bandit, unused_index):
    return np.delete(all_bandit, int(unused_index), axis=1)

def multiply_reward_to_option(all_shown_bandit, rew):
     return all_shown_bandit * rew[:, None]

def load_user_df(user_id):
    file = f"./data/raw/user_{user_id}.csv"
    user_df = pd.read_csv(file, index_col=0)

    return user_df


if __name__ == "__main__":
    log = pd.read_csv("org/data/data_collection.log", sep="\t", encoding="utf-16")
    user_id_list = [row["user"] for idx, row in log.iterrows() if row["exclude"]==0]

    n_sub = len(user_id_list)
    test_df = load_user_df(user_id_list[0], save_csv=True)
    n_bt_per_horizon = test_df.drop_duplicates("Blocktrial").groupby("Horizon").count().loc[6, "Blocktrial"]

    r_sum_concat = np.full(shape = (2, n_sub, n_bt_per_horizon, 3), fill_value=np.nan) # (horizon, subject, blocktrial, bandit)
    n_concat = np.full(shape = (2, n_sub, n_bt_per_horizon, 3), fill_value=np.nan) # (horizon, subject, blocktrial, bandit) 
    novelty_concat= np.full(shape = (2, n_sub, n_bt_per_horizon, 3), fill_value=np.nan) # (horizon, subject, blocktrial, bandit) 
                                                                    # value: 0 for sampled option, 1 for non-sampled option
    fst_chos_rews_concat = np.full(shape = (2, n_sub, n_bt_per_horizon, 3), fill_value=np.nan) # (horizon, subject, blocktrial, bandit)
    fst_chos_bandit_concat = np.zeros(shape = (2, n_sub, n_bt_per_horizon), dtype=int) # (horizon, subject, blocktrial, bandit)

    long_hrz_rews_concat = np.full(shape = (n_sub, n_bt_per_horizon, 5, 3), fill_value=np.nan) # (subject, blocktrial, choicetrial, bandit) 
                                                                            # saving the outcomes after the first choice trial to analyze the impact of exploration afterwards

    for s_idx, user_id in enumerate(user_id_list): #(user_id(1~792) with "jump" to subject_id (0~580) without "jump")
    
        subject_df = load_user_df(user_id, save_csv=True)
        short_horizon_df = subject_df[subject_df["Horizon"]==6] 
        long_horizon_df = subject_df[subject_df["Horizon"]==11]

        for h_idx, horizon_df in enumerate([short_horizon_df, long_horizon_df]): #h_idx: 0 for short horizon, 1 for long horizon
            for bt_idx, (bt, bt_df) in enumerate(horizon_df.groupby("Blocktrial")): #bt_idx: idx for "inside horizon" #bt: block_trial_idx for the whole task
                            
                bt_df = bt_df.copy().reset_index(drop=True)
                c = np.cumsum(~np.isnan(bt_df["PressedKey"]))
                unused_index = bt_df.loc[0, "UnusedTree"]-1
                all_shown_bandit = delete_unshown_option(np.array(bt_df.loc[:, "TreeA":"TreeD"]), unused_index)
                scr_rew = np.array(bt_df["Size"])
                all_shown_rews = multiply_reward_to_option(all_shown_bandit, scr_rew)
                
                # get previous sampling outcomes
                no_chos_rews = all_shown_rews[c==0] #non chosen (0) or first choice (1)
                # get r_sum, n, novelty info for each bandit
                r_sum = np.nansum(no_chos_rews, axis=0, dtype=float)
                n_obs = np.sum(~np.isnan(no_chos_rews), axis=0, dtype=float)
                novelty = np.array(n_obs==0, dtype=float)
                
                # get the first choice trial outcome
                fst_chos_rews = all_shown_rews[c==1].reshape(-1).astype(float)
                fst_chos_bandit = np.where(~np.isnan(fst_chos_rews))[0][0] + 1 #Index for chosen bandit (add 1 to match stan syntax)
    
                # concatenate
                r_sum_concat[h_idx, s_idx, bt_idx, :] = r_sum
                n_concat[h_idx, s_idx, bt_idx, :] = n_obs
                novelty_concat[h_idx, s_idx, bt_idx, :] = novelty
                fst_chos_rews_concat[h_idx, s_idx, bt_idx, :] = fst_chos_rews
                fst_chos_bandit_concat[h_idx, s_idx, bt_idx] = fst_chos_bandit

                # save the outcome after the first choice trial for long horizon
                if h_idx == 1:
                    long_hrz_rews_concat[s_idx, bt_idx, :, :] = all_shown_rews[c>1]
    
    os.makedirs("./data/concat_data", exist_ok=True)
    np.save("./data/concat_data/user_id.npy", np.array(user_id_list))
    np.save("./data/concat_data/r_sum.npy", r_sum_concat)
    np.save("./data/concat_data/n_obs.npy", n_concat)
    np.save("./data/concat_data/novelty.npy", novelty_concat)
    np.save("./data/concat_data/fst_chos_rews.npy", fst_chos_rews_concat)
    np.save("./data/concat_data/fst_chos_bandit.npy", fst_chos_bandit_concat)
    np.save("./data/concat_data/long_hrz_rews.npy", long_hrz_rews_concat)
data {

    int<lower=1> nH; // number of horizons
    int<lower=1> nS; // number of subjects
    int<lower=1> nBT; // number of blocktirals per horizon

    array[nH, nS, nBT] vector[3] r_sum; // sum of observed reward for each bandit
    array[nH, nS, nBT] vector[3] n_obs; // number of observation for each bandit
    array[nH, nS, nBT] vector[3] novelty; // indicator for non-sampled bandit
    array[nH, nS, nBT] int <lower=1, upper=3> fst_chos; // indicator for bandit chosen by participant at the first choice trial

}

transformed data{
    vector[3] tau_S;
    vector[3] tau_0;
    tau_S = rep_vector(inv_square(0.8), 3); // sampling precision
    tau_0 = rep_vector(inv_square(1.5), 3); // prior precision (fixed in UCB model)
}

parameters{
    
    // Group parameters (supposed to phi-transformed)
    real mu_Q_0; //[prior mean] (fixed across horizon)
    real<lower=0> sigma_Q_0; //[prior mean] (fixed across horizon)
    
    // vectorizing group parameters which varies across horizon
    array[nH] vector[3] mu_p; //[information bonus, inverse temperature, novelty bonus, value-free random]
    array[nH] vector<lower=0>[3] sigma;  //[information bonus, inverse temperature, novelty bonus, value-free random]

    // Individual parameters
    array[nS] real<lower=1, upper=10> Q_0; //prior mean

    // (Non-centered)
    array[nH, nS] real gamma_raw; //information bonus
    
    array[nH, nS] real beta_raw; //inverse temperature

    array[nH, nS] real epsilon_raw; //value-free random exploration

}

//Matt's Trick
transformed parameters{

    array[nH, nS] real gamma; //information bonus
    
    array[nH, nS] real beta; //inverse temperature

    array[nH, nS] real epsilon; //value-free random exploration

    for (h_idx in 1:nH){
        for (s_idx in 1:nS){

            gamma[h_idx, s_idx] = Phi_approx(mu_p[h_idx][1] + sigma[h_idx][1]*gamma_raw[h_idx, s_idx])*3; //boundary: (0, 3)
            beta[h_idx, s_idx] = Phi_approx(mu_p[h_idx][2] + sigma[h_idx][2]*beta_raw[h_idx, s_idx])*4.5+0.5; //boundary: (0.5, 5)
            epsilon[h_idx, s_idx] = Phi_approx(mu_p[h_idx][3] + sigma[h_idx][3]*epsilon_raw[h_idx, s_idx])*0.5; //boundary: (0, 0.5) 

        }
    }
    
}

model{
    
    mu_Q_0 ~ normal(5, 0.5);
    sigma_Q_0 ~ normal(0, 2);

    for (s_idx in 1:nS){

        Q_0[s_idx] ~ normal(mu_Q_0,sigma_Q_0);
    }

    for (h_idx in 1:nH) {

        mu_p[h_idx] ~ normal(0, 1);
        sigma[h_idx] ~ normal(0, 1);

        for (s_idx in 1:nS) {

            gamma_raw[h_idx, s_idx] ~ normal(0,1);

            beta_raw[h_idx, s_idx] ~ normal(0,1);
            
            epsilon_raw[h_idx, s_idx] ~ normal(0,1);

            for (bt_idx in 1:nBT) {
                vector[3] tau_n;
                vector[3] sigma_n;
                vector[3] Q_n;
                vector[3] V_n;
                vector[3] P_softmax;
                vector[3] P_final;

                //compute Q and sigma at the first choice trial using closed-form kalman filter
                tau_n = tau_0 + n_obs[h_idx, s_idx, bt_idx] .* tau_S;
                Q_n = (tau_0.*Q_0[s_idx] + tau_S.* r_sum[h_idx, s_idx, bt_idx])./tau_n;
                sigma_n = inv_sqrt(tau_n);
                
                //compute V by adding information bonus and novelty bonus
                V_n = Q_n + gamma[h_idx, s_idx] .* sigma_n;

                //Probability: value-based(softmax with inverse temperature) + value-free(epsilon)
                P_softmax = softmax(beta[h_idx, s_idx]*V_n);
                P_final = (1-epsilon[h_idx, s_idx])*P_softmax + rep_vector(epsilon[h_idx, s_idx]/3.0, 3);
                fst_chos[h_idx, s_idx, bt_idx] ~ categorical(P_final);
            }

        }

    }

}

generated quantities{
    array[nH] real mu_gamma;
    array[nH] real mu_beta;
    array[nH] real mu_eta;
    array[nH] real mu_epsilon;

    for (h_idx in 1:nH){

        mu_gamma[h_idx] = Phi_approx(mu_p[h_idx][1])*3;
        mu_beta[h_idx] = Phi_approx(mu_p[h_idx][2])*4.5+0.5;
        mu_epsilon[h_idx] = Phi_approx(mu_p[h_idx][3])*0.5;
    }

   // For log likelihood calculation
    array[nH, nS, nBT] real log_lik;
    
    // For posterior predictive check
    array[nH, nS, nBT] real y_pred;

    for (h_idx in 1:nH) {

        for (s_idx in 1:nS) {

            for (bt_idx in 1:nBT) {
                vector[3] tau_n;
                vector[3] sigma_n;
                vector[3] Q_n;
                vector[3] V_n;
                vector[3] P_softmax;
                vector[3] P_final;

                //compute Q and sigma at the first choice trial using closed-form kalman filter
                tau_n = tau_0 + n_obs[h_idx, s_idx, bt_idx] .* tau_S;
                Q_n = (tau_0.*Q_0[s_idx] + tau_S.* r_sum[h_idx, s_idx, bt_idx])./tau_n;
                sigma_n = inv_sqrt(tau_n);
                
                //compute V by adding information bonus and novelty bonus
                V_n = Q_n + gamma[h_idx, s_idx] .* sigma_n;
                
                //Probability: value-based(softmax with inverse temperature) + value-free(epsilon)
                P_softmax = softmax(beta[h_idx, s_idx]*V_n);
                P_final = (1-epsilon[h_idx, s_idx])*P_softmax + rep_vector(epsilon[h_idx, s_idx]/3.0, 3);
                log_lik[h_idx, s_idx, bt_idx] = categorical_lpmf(fst_chos[h_idx, s_idx, bt_idx] | P_final);
                y_pred[h_idx, s_idx, bt_idx] = categorical_rng(P_final);

            }

        }

    }
}
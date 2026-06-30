
functions{
    // bivariate normal cdf implemented on "Stan User's Guide".
    // https://mc-stan.org/docs/stan-users-guide/custom-probability.html
    // This computes p(x1 > z1 & x2 > z2) when x1 and x2 follows normal distribution, with zero mean, unit variance, correlation of "rho"

    real binormal_cdf(tuple(real, real) z, real rho) {
    real z1 = z.1;
    real z2 = z.2;
    if (z1 == 0 && z2 == 0) {
        return 0.25 + asin(rho) / (2 * pi());
    }
    real denom = sqrt((1 + rho) * (1 - rho));
    real term1 = z1 == 0
        ? (z2 > 0 ? 0.25 : -0.25)
        :  owens_t(z1, (z2 / z1 - rho) / denom);
    real term2 = z2 == 0
        ? (z1 > 0 ? 0.25 : -0.25)
        : owens_t(z2, (z1 / z2 - rho) / denom);
    real z1z2 = z1 * z2;
    real delta = z1z2 < 0 || (z1z2 == 0 && (z1 + z2) < 0);
    return 0.5 * (Phi(z1) + Phi(z2) - delta) - term1 - term2;
    }
}

data {

    int<lower=1> nH; // number of horizons
    int<lower=1> nS; // number of subjects
    int<lower=1> nBT; // number of blocktirals per horizon

    array[nH, nS, nBT] vector[3] r_sum; // sum of observed reward for each bandit
    array[nH, nS, nBT] vector[3] n_obs; // number of observation for each bandit
    array[nH, nS, nBT] vector[3] novelty; // indicator for non-sampled bandit
    array[nH, nS, nBT] int <lower=1, upper=3> fst_chos; // indicator for bandit chosen by participant at the first choice trial

}

transformed data {

   vector[3] tau_S;
   tau_S = rep_vector(inv_square(0.8), 3);

}

parameters{
    
    // Group parameters (supposed to phi-transformed)
    real mu_Q_0; //[prior mean] (fixed across horizon)
    real<lower=0> sigma_Q_0; //[prior mean] (fixed across horizon)
    
    // vectorizing group parameters which varies across horizon
    array[nH] vector[2] mu_p; //[prior uncertainty, novelty bonus, value-free random]
    array[nH] vector<lower=0>[2] sigma;  //[prior uncertainty, novelty bonus, value-free random]

    // Individual parameters
    array[nS] real<lower=1, upper=10> Q_0; //prior mean

    // (Non-centered)
    array[nH, nS] real sigma_0_raw; //prior uncertainty

    array[nH, nS] real eta_raw; //novelty bonus
    
}

//Matt's Trick
transformed parameters{

    array[nH, nS] real sigma_0; //prior uncertainty
        
    array[nH, nS] real eta; //novelty bonus

    for (h_idx in 1:nH){
        for (s_idx in 1:nS){
            sigma_0[h_idx, s_idx] = Phi_approx(mu_p[h_idx][1] + sigma[h_idx][1] *sigma_0_raw[h_idx, s_idx])*5.99 + 0.01; //boundary: (0.01, 6)
            eta[h_idx, s_idx] = Phi_approx(mu_p[h_idx][2] + sigma[h_idx][2]*eta_raw[h_idx, s_idx])*5; //boundary: (0, 5)
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
            
            sigma_0_raw[h_idx, s_idx] ~ normal(0,1);
            
            eta_raw[h_idx, s_idx] ~ normal(0,1);
            
            for (bt_idx in 1:nBT) {
                vector[3] tau_0;
                vector[3] tau_n;
                vector[3] sigma_n;
                vector[3] Q_n;
                vector[3] V_n;
                vector[3] P_n_raw;
                vector[3] P_n;

                //convert variance to precision for readability (more simple expression)
                tau_0 = rep_vector(inv_square(sigma_0[h_idx, s_idx]), 3); 
                
                //compute Q and sigma at the first choice trial using closed-form kalman filter
                tau_n = tau_0 + n_obs[h_idx, s_idx, bt_idx] .* tau_S;
                Q_n = (tau_0.*Q_0[s_idx] + tau_S.* r_sum[h_idx, s_idx, bt_idx])./(tau_0 + tau_S.* n_obs[h_idx, s_idx, bt_idx]);
                sigma_n = inv_sqrt(tau_n);
                
                //compute V by adding novelty bonus
                V_n = Q_n + eta[h_idx, s_idx] .* novelty[h_idx, s_idx, bt_idx];

                // Thompson sampling
                // Although the original code from Dubois & Hauser (2022) used "A matrix", repeating matrix computation for all bandit makes HMC really slow.
                // Instead, manually compute choice probability for each bandit
                vector[3] sq_sigma = square(sigma_n);

                real m1_1 = V_n[1] - V_n[2];
                real m1_2 = V_n[1] - V_n[3];
                real c1_11 = sq_sigma[1]+sq_sigma[2];
                real c1_22 = sq_sigma[1]+sq_sigma[3];
                real c1_12 = sq_sigma[1];
                
                P_n_raw[1] = binormal_cdf( (m1_1 / sqrt(c1_11), m1_2 / sqrt(c1_22)) | c1_12 / sqrt(c1_11*c1_22));
                P_n_raw[1] = fmax(1e-12, P_n_raw[1]);

                real m2_1 = V_n[2] - V_n[1];
                real m2_2 = V_n[2] - V_n[3];
                real c2_11 = sq_sigma[2]+sq_sigma[1];
                real c2_22 = sq_sigma[2]+sq_sigma[3];
                real c2_12 = sq_sigma[2];
                
                P_n_raw[2] = binormal_cdf( (m2_1 / sqrt(c2_11), m2_2 / sqrt(c2_22)) | c2_12 / sqrt(c2_11*c2_22));
                P_n_raw[2] = fmax(1e-12, P_n_raw[2]);

                real m3_1 = V_n[3] - V_n[1];
                real m3_2 = V_n[3] - V_n[2];
                real c3_11 = sq_sigma[3] + sq_sigma[1];
                real c3_22 = sq_sigma[3] + sq_sigma[2];
                real c3_12 = sq_sigma[3];
                
                P_n_raw[3] = binormal_cdf( (m3_1 / sqrt(c3_11), m3_2 / sqrt(c3_22)) | c3_12 / sqrt(c3_11*c3_22));
                P_n_raw[3] = fmax(1e-12, P_n_raw[3]);

                P_n = P_n_raw/sum(P_n_raw);
                fst_chos[h_idx, s_idx, bt_idx] ~ categorical(P_n);
            }

        }

    }

}

generated quantities{
    array[nH] real mu_sigma_0;
    array[nH] real mu_eta;

    for (h_idx in 1:nH){

        mu_sigma_0[h_idx] = Phi_approx(mu_p[h_idx][1])*5.99+0.01;
        mu_eta[h_idx] = Phi_approx(mu_p[h_idx][2])*5;
    }

   // For log likelihood calculation
    array[nH, nS, nBT] real log_lik;
    
    // For posterior predictive check
    array[nH, nS, nBT] real y_pred;
   
    for (h_idx in 1:nH) {

        for (s_idx in 1:nS) {
            
            for (bt_idx in 1:nBT) {
                vector[3] tau_0;
                vector[3] tau_n;
                vector[3] sigma_n;
                vector[3] Q_n;
                vector[3] V_n;
                vector[3] P_n_raw;
                vector[3] P_n;

                //convert sigma to precision for readability (more simple expression)
                tau_0 = rep_vector(inv_square(sigma_0[h_idx, s_idx]), 3); 
                
                //compute Q and sigma at the first choice trial using closed-form kalman filter
                tau_n = tau_0 + n_obs[h_idx, s_idx, bt_idx] .* tau_S;
                Q_n = (tau_0.*Q_0[s_idx] + tau_S.* r_sum[h_idx, s_idx, bt_idx])./tau_n;
                sigma_n = inv_sqrt(tau_n);
                
                //compute V by adding novelty bonus
                V_n = Q_n + eta[h_idx, s_idx] .* novelty[h_idx, s_idx, bt_idx];

                // Thompson sampling
                // Although the original code from Dubois & Hauser (2022) used "A matrix", repeating matrix computation for all bandit makes HMC really slow.
                // Instead, manually compute choice probability for each bandit
                vector[3] sq_sigma = square(sigma_n);

                real m1_1 = V_n[1] - V_n[2];
                real m1_2 = V_n[1] - V_n[3];
                real c1_11 = sq_sigma[1]+sq_sigma[2];
                real c1_22 = sq_sigma[1]+sq_sigma[3];
                real c1_12 = sq_sigma[1];
                
                P_n_raw[1] = binormal_cdf( (m1_1 / sqrt(c1_11), m1_2 / sqrt(c1_22)) | c1_12 / sqrt(c1_11*c1_22));
                P_n_raw[1] = fmax(1e-12, P_n_raw[1]);
                
                real m2_1 = V_n[2] - V_n[1];
                real m2_2 = V_n[2] - V_n[3];
                real c2_11 = sq_sigma[2]+sq_sigma[1];
                real c2_22 = sq_sigma[2]+sq_sigma[3];
                real c2_12 = sq_sigma[2];
                
                P_n_raw[2] = binormal_cdf( (m2_1 / sqrt(c2_11), m2_2 / sqrt(c2_22)) | c2_12 / sqrt(c2_11*c2_22));
                P_n_raw[2] = fmax(1e-12, P_n_raw[2]);

                real m3_1 = V_n[3] - V_n[1];
                real m3_2 = V_n[3] - V_n[2];
                real c3_11 = sq_sigma[3] + sq_sigma[1];
                real c3_22 = sq_sigma[3] + sq_sigma[2];
                real c3_12 = sq_sigma[3];
                
                P_n_raw[3] = binormal_cdf( (m3_1 / sqrt(c3_11), m3_2 / sqrt(c3_22)) | c3_12 / sqrt(c3_11*c3_22));
                P_n_raw[3] = fmax(1e-12, P_n_raw[3]);

                P_n = P_n_raw/sum(P_n_raw);
                log_lik[h_idx, s_idx, bt_idx] = categorical_lpmf(fst_chos[h_idx, s_idx, bt_idx] | P_n);
                y_pred[h_idx, s_idx, bt_idx] = categorical_rng(P_n);

            }

        }

    }

}
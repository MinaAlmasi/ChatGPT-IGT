#install.packages("pacman")
pacman::p_load(hesim, extraDistr, R2jags, parallel)

# set seed
set.seed(2502)

# read json
root_path <- "~/Desktop/dm-code" # personal comp
#root_path <- "dm-code" # UCloud
file <- file.path(root_path, "ChatGPT-IGT", "src", "recovery", "simulated_data", "simulated_group_data.json")
data <- jsonlite::fromJSON(file)

MPD <- function(x) {density(x)$x[which(density(x)$y==max(density(x)$y))]}

# define n groups (iterations to run)
n_groups <- 30

# subset data to n_groups (n groups)
data <- lapply(data, function(x) x[1:n_groups])

# generate combinations
combinations_groups <- combn(n_groups, 2, simplify = FALSE)

# define n_comparisons
n_comparisons <- n_groups

# sample from combinations (to reduce amount of combinations due to computational time)
sampled_combinations <- sample(combinations_groups, n_comparisons, replace = FALSE)

# define arrays to store true and inferred parameters
true_alpha_a_rew <- array(NA,c(n_comparisons))
true_alpha_a_pun <- array(NA,c(n_comparisons))
true_alpha_K <- array(NA,c(n_comparisons))
true_alpha_theta <- array(NA,c(n_comparisons))
true_alpha_omega_f <- array(NA,c(n_comparisons))
true_alpha_omega_p <- array(NA,c(n_comparisons))

infer_alpha_a_rew <- array(NA,c(n_comparisons))
infer_alpha_a_pun <- array(NA,c(n_comparisons))
infer_alpha_K <- array(NA,c(n_comparisons))
infer_alpha_theta <- array(NA,c(n_comparisons))
infer_alpha_omega_f <- array(NA,c(n_comparisons))
infer_alpha_omega_p <- array(NA,c(n_comparisons))

start_time = Sys.time()
for (i in 1:length(sampled_combinations)) {
    start_iteration = Sys.time()

    # extract groups for comparison
    combination <- sampled_combinations[[i]]
    grp1_number <- combination[1]
    grp2_number <- combination[2]
    
    # extract data from each group
    grp1_data <- lapply(data, function(x) x[grp1_number])
    grp2_data <- lapply(data, function(x) x[grp2_number])

    # extract x and X from each group
    x_grp1 <- grp1_data$x[[1]]
    X_grp1 <- grp1_data$X[[1]]
    x_grp2 <- grp2_data$x[[1]]
    X_grp2 <- grp2_data$X[[1]]

    # change x from values 0, 1, 2, 3 to 1, 2, 3, 4
    x_grp1 <- x_grp1 + 1
    x_grp2 <- x_grp2 + 1

    # define ntrials and nsubs
    ntrials <- rep(dim(x_grp1)[2], dim(x_grp1)[1]) # trials, subjects (defined by the dimensions of the choices)
    nsubs <- dim(x_grp1)[1]

    # setup jags 
    print("Intializing JAGS ...")
    jags_data <- list("x_grp1", "X_grp1", "x_grp2", "X_grp2", "ntrials", "nsubs")
    params<-c("alpha_a_rew","alpha_a_pun","alpha_K","alpha_omega_f","alpha_omega_p", "alpha_theta")

    # run jags
    model_file <- file.path(root_path, "ChatGPT-IGT", "models", "hier_ORL_compare.txt")
    samples <- jags.parallel(jags_data, inits=NULL, params,
                model.file =model_file,
                n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4)
    
    print(samples$BUGSoutput)

    # true parameters
    true_alpha_a_rew[i] <- as.numeric(grp2_data$mu_a_rew[[1]] - grp1_data$mu_a_rew[[1]])
    true_alpha_a_pun[i] <- as.numeric(grp2_data$mu_a_pun[[1]] - grp1_data$mu_a_pun[[1]])
    true_alpha_K[i] <- as.numeric(grp2_data$mu_K[[1]] - grp1_data$mu_K[[1]])
    true_alpha_theta[i] <- as.numeric(grp2_data$mu_theta[[1]] - grp1_data$mu_theta[[1]])
    true_alpha_omega_f[i] <- as.numeric(grp2_data$mu_omega_f[[1]] - grp1_data$mu_omega_f[[1]])
    true_alpha_omega_p[i] <- as.numeric(grp2_data$mu_omega_p[[1]] - grp1_data$mu_omega_p[[1]])
    
    # extract samples to inferred
    Y <- samples$BUGSoutput$sims.list
    infer_alpha_a_rew[i] <- MPD(Y$alpha_a_rew)
    infer_alpha_a_pun[i] <- MPD(Y$alpha_a_pun)
    infer_alpha_K[i] <- MPD(Y$alpha_K)
    infer_alpha_theta[i] <- MPD(Y$alpha_theta)
    infer_alpha_omega_f[i] <- MPD(Y$alpha_omega_f)
    infer_alpha_omega_p[i] <- MPD(Y$alpha_omega_p)

    # time 
    end_iteration <- Sys.time()
    run_iteration <- round(end_iteration - start_iteration, 2)
    print(paste0(i, " Iteration time: ", run_iteration, " minutes"))
}

# save df with true and inferred parameters
df <- data.frame(true_alpha_a_rew, true_alpha_a_pun, true_alpha_K, true_alpha_theta, true_alpha_omega_f, true_alpha_omega_p,
                infer_alpha_a_rew, infer_alpha_a_pun, infer_alpha_K, infer_alpha_theta, infer_alpha_omega_f, infer_alpha_omega_p)

# save df
write.csv(df, file.path(root_path, "ChatGPT-IGT", "src", "recovery", "results", "param_recovery_group_comparisons.csv"))

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)
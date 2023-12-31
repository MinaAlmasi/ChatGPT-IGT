#install.packages("pacman")
pacman::p_load(hesim, extraDistr, R2jags, parallel)

# set seed
set.seed(2502)

# define whether theta is fixed or not
fixed_theta <- TRUE

# read json
root_path <- "~/Desktop/dm-code" # personal comp
#root_path <- "dm-code" # UCloud

# define filename based on whether theta is fixed or not
filename <- paste0("simulated_group_data", if(fixed_theta) "_fixed_theta", ".json")
file <- file.path(root_path, "ChatGPT-IGT", "src", "recovery", "simulated_data", filename)
data <- jsonlite::fromJSON(file)

MPD <- function(x) {density(x)$x[which(density(x)$y==max(density(x)$y))]}

# define n groups (iterations to run)
n_groups <- 30

# subset data to n_groups (n groups)
data <- lapply(data, function(x) x[1:n_groups])

## DEFINE PARAMS ## 
# mu
true_mu_a_rew <- array(NA,c(n_groups))
true_mu_a_pun <- array(NA,c(n_groups))
true_mu_K <- array(NA,c(n_groups))
true_mu_omega_f <- array(NA,c(n_groups))
true_mu_omega_p <- array(NA,c(n_groups))

infer_mu_a_rew <- array(NA,c(n_groups))
infer_mu_a_pun <- array(NA,c(n_groups))
infer_mu_K <- array(NA,c(n_groups))
infer_mu_omega_f <- array(NA,c(n_groups))
infer_mu_omega_p <- array(NA,c(n_groups))

# sigma (SD for R) / lambda (precision for JAGS)
true_lambda_a_rew <- array(NA,c(n_groups))
true_lambda_a_pun <- array(NA,c(n_groups))
true_lambda_K <- array(NA,c(n_groups))
true_lambda_omega_f <- array(NA,c(n_groups))
true_lambda_omega_p <- array(NA,c(n_groups))

infer_lambda_a_rew <- array(NA,c(n_groups))
infer_lambda_a_pun <- array(NA,c(n_groups))
infer_lambda_K <- array(NA,c(n_groups))
infer_lambda_omega_f <- array(NA,c(n_groups))
infer_lambda_omega_p <- array(NA,c(n_groups))

if(!fixed_theta) {
    true_mu_theta <- array(NA,c(n_groups))
    infer_mu_theta <- array(NA,c(n_groups))
    true_lambda_theta <- array(NA,c(n_groups))
    infer_lambda_theta <- array(NA,c(n_groups))
}

print(paste0("Working on file: ", basename(file)))
start_time = Sys.time()
for (i in 1:n_groups) {
    start_iteration = Sys.time()

    # get group choices and rewards
    x <- data$x[i][[1]]
    X <- data$X[i][[1]]

    # nsubs
    nsubs <- dim(x)[1]

    # change x from values 0, 1, 2, 3 to 1, 2, 3, 4
    x <- x + 1

    # get the group parameters
    mu_a_rew <- as.numeric(data$mu_a_rew[i])
    mu_a_pun <- as.numeric(data$mu_a_pun[i])
    mu_K <- as.numeric(data$mu_K[i])
    mu_omega_f <- as.numeric(data$mu_omega_f[i])
    mu_omega_p <- as.numeric(data$mu_omega_p[i])

    sigma_a_rew <- as.numeric(data$sigma_a_rew[i])
    sigma_a_pun <- as.numeric(data$sigma_a_pun[i])
    sigma_K <- as.numeric(data$sigma_K[i])

    sigma_omega_f <- as.numeric(data$sigma_omega_f[i])
    sigma_omega_p <- as.numeric(data$sigma_omega_p[i])

    if (!fixed_theta) {
        mu_theta <- as.numeric(data$mu_theta[i])
        sigma_theta <- as.numeric(data$sigma_theta[i])
    }

    ntrials <- rep(dim(x)[2], dim(x)[1]) # trials, subjects (defined by the dimensions of the choices)
    
    # setup jags
    jags_data <- list("x", "X", "ntrials", "nsubs")

    # change setup depending on whether theta is fixed or not
    if (!fixed_theta) {
        params <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_theta", "mu_omega_f", "mu_omega_p",
                "lambda_a_rew", "lambda_a_pun", "lambda_K", "lambda_theta", "lambda_omega_f", "lambda_omega_p")
        model_file <- file.path(root_path, "ChatGPT-IGT", "models", "hier_ORL.txt")
    } else {
        params <- c("mu_a_rew", "mu_a_pun", "mu_K", "mu_omega_f", "mu_omega_p",
                "lambda_a_rew", "lambda_a_pun", "lambda_K", "lambda_omega_f", "lambda_omega_p")
        model_file <- file.path(root_path, "ChatGPT-IGT", "models", "hier_ORL_fixed_theta.txt")
    }

    print("Intializing JAGS ...")
    samples <- jags.parallel(jags_data, inits = NULL, params,
                model.file = model_file, n.chains = 3, 
                n.iter = 3000, n.burnin = 1000, n.thin = 1, n.cluster = 4)

    print(samples$BUGSoutput)

    # mu
    true_mu_a_rew[i] <- mu_a_rew
    true_mu_a_pun[i] <- mu_a_pun
    true_mu_K[i] <- mu_K
    true_mu_omega_f[i] <- mu_omega_f
    true_mu_omega_p[i] <- mu_omega_p
    
    # find maximum a posteriori
    Y <- samples$BUGSoutput$sims.list
    infer_mu_a_rew[i] <- MPD(Y$mu_a_rew)
    infer_mu_a_pun[i] <- MPD(Y$mu_a_pun)
    infer_mu_K[i] <- MPD(Y$mu_K)
    infer_mu_omega_f[i] <- MPD(Y$mu_omega_f)
    infer_mu_omega_p[i] <- MPD(Y$mu_omega_p)
    
    # lambda (converting sigma)
    true_lambda_a_rew[i] <- 1/sigma_a_rew
    true_lambda_a_pun[i] <- 1/sigma_a_pun
    true_lambda_K[i] <- 1/sigma_K
    true_lambda_omega_f[i] <- 1/sigma_omega_f
    true_lambda_omega_p[i] <- 1/sigma_omega_p
    
    # find maximum a posteriori
    infer_lambda_a_rew[i] <- MPD(Y$lambda_a_rew)
    infer_lambda_a_pun[i] <- MPD(Y$lambda_a_pun)
    infer_lambda_K[i] <- MPD(Y$lambda_K)
    infer_lambda_omega_f[i] <- MPD(Y$lambda_omega_f)
    infer_lambda_omega_p[i] <- MPD(Y$lambda_omega_p)

    if(!fixed_theta) {
        true_mu_theta[i] <- mu_theta
        infer_mu_theta[i] <- MPD(Y$mu_theta)
        true_lambda_theta[i] <- 1/sigma_theta
        infer_lambda_theta[i] <- MPD(Y$lambda_theta)
    }
    
    # time 
    end_iteration <- Sys.time()
    run_iteration <- round(end_iteration - start_iteration, 2)
    print(paste0(i, " Iteration time: ", run_iteration, " minutes"))
}   

# save to df with n_groups rows and 12 columns (6 true, 6 infer)
if(!fixed_theta){
    df <- data.frame(true_mu_a_rew, true_mu_a_pun, true_mu_K, true_mu_theta, true_mu_omega_f, true_mu_omega_p, infer_mu_a_rew, infer_mu_a_pun, infer_mu_K, infer_mu_theta, infer_mu_omega_f, infer_mu_omega_p,
                 true_lambda_a_rew, true_lambda_a_pun, true_lambda_K, true_lambda_theta, true_lambda_omega_f, true_lambda_omega_p, infer_lambda_a_rew, infer_lambda_a_pun, infer_lambda_K, infer_lambda_theta, infer_lambda_omega_f, infer_lambda_omega_p)
} else {
    df <- data.frame(true_mu_a_rew, true_mu_a_pun, true_mu_K, true_mu_omega_f, true_mu_omega_p, infer_mu_a_rew, infer_mu_a_pun, infer_mu_K, infer_mu_omega_f, infer_mu_omega_p,
                 true_lambda_a_rew, true_lambda_a_pun, true_lambda_K, true_lambda_omega_f, true_lambda_omega_p, infer_lambda_a_rew, infer_lambda_a_pun, infer_lambda_K, infer_lambda_omega_f, infer_lambda_omega_p)
}

# save to csv
savefile_name = paste0("param_recovery_group_ALL_", if(fixed_theta) "fixed_theta", ".csv")
write.csv(df, file.path(root_path, "ChatGPT-IGT", "src", "recovery", "results", savefile_name), row.names=FALSE)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)
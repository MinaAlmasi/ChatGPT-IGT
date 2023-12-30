#install.packages("pacman")
pacman::p_load(hesim, extraDistr, R2jags, parallel)

# set seed
set.seed(2502)

# define whether theta is fixed or not
fixed_theta <- FALSE

# define root path depending on whether it is run locally or on Ucloud
root_path <- "~/Desktop/dm-code" # personal comp
#root_path <- "dm-code" # UCloud

# define filename based on whether theta is fixed or not
filename <- paste0("simulated_single_subject_data", if(fixed_theta) "_fixed_theta", ".json")

file <- file.path(root_path, "ChatGPT-IGT", "src", "recovery", "simulated_data", filename)
data <- jsonlite::fromJSON(file)

MPD <- function(x) {density(x)$x[which(density(x)$y==max(density(x)$y))]}

n_iterations <- 100
ntrials <- 100

true_a_rew <- array(NA,c(n_iterations))
true_a_pun <- array(NA,c(n_iterations))
true_K <- array(NA,c(n_iterations))
true_theta <- array(NA,c(n_iterations))
true_omega_f <- array(NA,c(n_iterations))
true_omega_p <- array(NA,c(n_iterations))

infer_a_rew <- array(NA,c(n_iterations))
infer_a_pun <- array(NA,c(n_iterations))
infer_K <- array(NA,c(n_iterations))
infer_theta <- array(NA,c(n_iterations))
infer_omega_f <- array(NA,c(n_iterations))
infer_omega_p <- array(NA,c(n_iterations))

x_pred <- array(NA,c(n_iterations, ntrials))

start_time = Sys.time()

for (i in 1:n_iterations) {
    start_iteration = Sys.time()
    # get true parameter values
    a_rew <- as.numeric(data$a_rew[i])
    a_pun <- as.numeric(data$a_pun[i])
    K <- as.numeric(data$K[i])
    theta <- as.numeric(data$theta[i])
    omega_f <- as.numeric(data$omega_f[i])
    omega_p <- as.numeric(data$omega_p[i])

    # define x, X
    x <- data$x[[i]]
    X <- data$X[[i]]

    # change x from values 0, 1, 2, 3 to 1, 2, 3, 4
    x <- x + 1

    # setup jags
    jags_data <- list("x", "X", "ntrials")
    params <- c("a_rew", "a_pun", "K", "theta", "omega_f", "omega_p", "p")
    
    # change setup depending on whether theta is fixed or not
    params <- if(fixed_theta) setdiff(params, "theta") else params # rm theta from params to recover if fixed
    model_filename <- paste0("ORL", if(fixed_theta) "_fixed_theta", ".txt")
    model_file <- file.path(root_path, "ChatGPT-IGT", "models", model_filename)

    # initialize JAGS
    samples <- jags.parallel(jags_data, inits = NULL, params,
                model.file = model_file, n.chains = 3, 
                n.iter = 3000, n.burnin = 1000, n.thin = 1, n.cluster = 3)

    print(samples$BUGSoutput)

    true_a_rew[i] <- a_rew
    true_a_pun[i] <- a_pun
    true_K[i] <- K
    true_theta[i] <- theta
    true_omega_f[i] <- omega_f
    true_omega_p[i] <- omega_p
    
    # find maximum a posteriori
    Y <- samples$BUGSoutput$sims.list
    infer_a_rew[i] <- MPD(Y$a_rew)
    infer_a_pun[i] <- MPD(Y$a_pun)
    infer_K[i] <- MPD(Y$K)
    infer_theta[i] <- MPD(Y$theta)
    infer_omega_f[i] <- MPD(Y$omega_f)
    infer_omega_p[i] <- MPD(Y$omega_p)

    # inferred choice based on probabilities
    # set up x_pred for the subject
    x_pred_subj <- array(NA,c(ntrials))

    # get probabilities for each response per trials and take max
    for (j in 1:ntrials) {
        p_predict <- c(
            MPD(Y$p[,j,1]),
            MPD(Y$p[,j,2]),
            MPD(Y$p[,j,3]),
            MPD(Y$p[,j,4])
        )
        x_pred_subj[j] <- which.max(p_predict)
    }

    # assign to general array
    x_pred[i,] <- x_pred_subj

    # time
    end_iteration <- Sys.time()
    run_iteration <- round(end_iteration - start_iteration, 2)
    print(paste0(i, " Iteration time: ", run_iteration, " secs"))
}

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)

par(mfrow=c(3,2))
plot(true_a_rew,infer_a_rew)
plot(true_a_pun,infer_a_pun)
plot(true_K,infer_K)
plot(true_theta,infer_theta)
plot(true_omega_f,infer_omega_f)
plot(true_omega_p,infer_omega_p)

# save to df with n_iterations rows and 12 columns (6 true, 6 infer)
df <- data.frame(true_a_rew, true_a_pun, true_K, true_theta, true_omega_f, true_omega_p, infer_a_rew, infer_a_pun, infer_K, infer_theta, infer_omega_f, infer_omega_p, x_pred)

# save to csv
savefile_name = paste0("param_recovery_single_subject_", if(fixed_theta) "fixed_theta", ".csv")
write.csv(df, file.path(root_path, "ChatGPT-IGT", "src", "recovery", "results", savefile_name), row.names=FALSE)
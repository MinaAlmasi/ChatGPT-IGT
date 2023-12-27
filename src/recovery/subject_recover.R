#install.packages("pacman")
pacman::p_load(hesim, extraDistr, R2jags, parallel, ggpubr)

# read json
file <- file.path("~", "Desktop", "dm-code", "ChatGPT-IGT", "src", "recovery", "simulated_single_subject_data.json")
data <- jsonlite::fromJSON(file)

n_iterations <- 10
for (i in 1:n_iterations) {
    
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

    ntrials <- 100

    # setup jags 
    jags_data <- list("x", "X", "ntrials")
    params <- c("a_rew", "a_pun", "K", "theta", "omega_f", "omega_p")
    model_file <- file.path("~", "Desktop", "dm-code", "ChatGPT-IGT", "models", "ORL.txt")

    samples <- jags.parallel(jags_data, inits = NULL, params,
                model.file = model_file, n.chains = 2, 
                n.iter = 100, n.burnin = 10, n.thin = 1, n.cluster = 3)

    print(samples$BUGSoutput) 
}

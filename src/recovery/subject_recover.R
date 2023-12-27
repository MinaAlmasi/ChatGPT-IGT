install.packages("pacman")
pacman::p_load(hesim, extraDistr, R2jags, parallel, ggpubr)

# read json
file <- file.path("~", "Desktop", "dm-code", "ChatGPT-IGT", "src", "recovery", "simulated_single_subject_data.json")
data <- jsonlite::fromJSON(file)


n_iterations <- 10
for (i in 1:n_iterations) {
    a_rew <- data$a_rew[i]
    a_pun <- data$a_pun[i]
    K <- data$K[i]
    theta <- data$theta[i]
    omega_f <- data$omega_f[i]
    omega_p <- data$omega_p[i]

    
    

}

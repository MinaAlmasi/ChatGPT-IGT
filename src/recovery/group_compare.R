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

n_comparisons

length(sampled_combinations)

for (i in 1:length(sampled_combinations)) {
    # extract groups for comparison
    combination <- sampled_combinations[[i]]
    grp1_number <- combination[1]
    grp2_number <- combination[2]
    
    # extract data from each group
    grp1_data <- lapply(data, function(x) x[grp1_number])
    grp2_data <- lapply(data, function(x) x[grp2_number])

    # extract x and X from each group
    grp1_x <- grp1_data$x[[1]]
    grp1_X <- grp1_data$X[[1]]
    grp2_x <- grp2_data$x[[1]]
    grp2_X <- grp2_data$X[[1]]

    # change x from values 0, 1, 2, 3 to 1, 2, 3, 4
    grp1_x <- grp1_x + 1
    grp2_x <- grp2_x + 1

    # define ntrials and nsubs
    ntrials <- 100
    nsubs <- length(grp1_x)

    # setup jags 
    jags_data <- list("x_grp1", "X_grp1", "x_grp2", "X_grp2", "ntrials", "nsubs")

    params<-c("alpha_a_rew","alpha_a_pun","alpha_K","alpha_omega_f","alpha_omega_p", "")

}
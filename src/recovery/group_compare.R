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


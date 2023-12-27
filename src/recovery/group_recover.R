#install.packages("pacman")
pacman::p_load(hesim, extraDistr, R2jags, parallel, ggpubr)

# read json
file <- file.path("~", "Desktop", "dm-code", "ChatGPT-IGT", "src", "recovery", "simulated_data", "simulated_group_data.json")
data <- jsonlite::fromJSON(file)

MPD <- function(x) {density(x)$x[which(density(x)$y==max(density(x)$y))]}

# print data x for the first group for the first subject
print(data$x[3])

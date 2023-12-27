install.packages("pacman")
pacman::p_load(hesim, extraDistr, R2jags, parallel, ggpubr)

# load payoff
file <- file.path("~", "Desktop", "dm-code", "ChatGPT-IGT", "utils", "payoff_scheme_3.csv")
payoff <- read.csv(file, header = TRUE)

print(payoff)

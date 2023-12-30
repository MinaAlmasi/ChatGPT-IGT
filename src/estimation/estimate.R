#install.packages("pacman")
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

# read data
root_path <- "~/Desktop/dm-code" # personal comp
#root_path <- "dm-code" # UCloud
group_name = "gpt"
file <- file.path(root_path, "ChatGPT-IGT", "data", "final_data", paste0("clean_", group_name, ".csv"))
data <- read.csv(file)

# extract relevant variables
subIDs <- unique(data$subjID)
nsubs <- length(subIDs)
x_raw <- data$x
X_raw <- data$X

# initialize empty arrays
ntrials_max <- 100
x <- array(0,c(nsubs,ntrials_max))
X <- array(0,c(nsubs,ntrials_max))
ntrials <- array(0,c(nsubs))

# turn data from long format into arrays with (nsubs x ntrials_max) dimensions
for (s in 1:nsubs) {
  
  #record n trials for subject s
  ntrials[s] <- length(x_raw[data$subjID==subIDs[s]])
  
  # extract x and X for subject s
  X_sub <- X_raw[data$subjID==subIDs[s]] 
  x_sub <- x_raw[data$subjID==subIDs[s]] 

  # assign arrays
  x[s,] <- x_sub
  X[s,] <- X_sub
}

# scale the payoffs
X <- X/100

# set up jags and run jags model
jags_data <- list("x","X","ntrials","nsubs") 
params<-c("mu_a_rew","mu_a_pun","mu_K","mu_theta","mu_omega_f","mu_omega_p") 
model_file <- file.path(root_path, "ChatGPT-IGT", "models", "hier_ORL.txt")

# set timer
start_time = Sys.time()

# run jags model
print("Intializing JAGS ...")
samples <- jags.parallel(jags_data, inits=NULL, params,
                         model.file = model_file,
                         n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4)

print(samples$BUGSoutput)

# extact mu parameters
Y <- samples$BUGSoutput$sims.list
mu_a_rew <- Y$mu_a_rew
mu_a_pun <- Y$mu_a_pun
mu_K <- Y$mu_K
mu_theta <- Y$mu_theta
mu_omega_f <- Y$mu_omega_f
mu_omega_p <- Y$mu_omega_p

# save to df 
df <- data.frame(mu_a_rew, mu_a_pun, mu_K, mu_theta, mu_omega_f, mu_omega_p)

# save df
savefile <- file.path(root_path, "ChatGPT-IGT", "src", "estimation", "estimated_parameters", paste0("param_estimated_", group_name, ".csv"))
write.csv(df, savefile)

# print time run
end_time = Sys.time()
print(end_time - start_time)



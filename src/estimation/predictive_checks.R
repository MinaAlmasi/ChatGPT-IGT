#install.packages("pacman")
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

# defining a function for calculating the maximum of the posterior density (not exactly the same as the mode)
MPD <- function(x) {density(x)$x[which(density(x)$y==max(density(x)$y))]}

root_path <- "~/Desktop/dm-code" # personal comp
#root_path <- "dm-code" # UCloud
group_name = "gpt"
file <- file.path(root_path, "ChatGPT-IGT", "data", "final_data", paste0("clean_", group_name, ".csv"))
data <- read.csv(file)

# get vars
source(file.path(root_path, "ChatGPT-IGT", "src", "data_util.R"))
data_vars <- processData(data)

# unpack vars
x_all <- data_vars$x
X_all <- data_vars$X
ntrials <- data_vars$ntrials
nsubs <- data_vars$nsubs

pred_success <- array(nsubs)

start_time = Sys.time()
for (s in 1:nsubs) {
  
  x <- x_all[s, ]
  X <- X_all[s, ]
  ntrials <- ntrials[s]
  
  # set up jags and run jags model on one subject
  data <- list("x","X","ntrials") 
  params<-c("a_rew","a_pun","K","theta","omega_f","omega_p","p")
  model_file <- file.path(root_path, "ChatGPT-IGT", "models", "ORL.txt")
  temp_samples <- jags.parallel(data, inits=NULL, params,
                                model.file =model_file,
                                n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=3)
  
  p_post <- temp_samples$BUGSoutput$sims.list$p
  
  x_predict <- array(ntrials)
  
  for (t in 1:ntrials) {
    p_predict <- c(
      MPD(p_post[,t,1]),
      MPD(p_post[,t,2]),
      MPD(p_post[,t,3]),
      MPD(p_post[,t,4])
    )
    
    x_predict[t] <- which.max(p_predict)
    
  }
  
  pred_success[s] <- sum(x_predict==x[1:ntrials]) # only comparing with trials for which we have choices
  print(s)
}

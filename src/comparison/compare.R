# script inspired by ORL_compare (UCloud: /Module 4/group_comparison/ORL_compare.R) 
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

# read data
root_path <- "~/Desktop/dm-code" # personal comp
#root_path <- "dm-code" # UCloud
gpt_file <- file.path(root_path, "ChatGPT-IGT", "data", "final_data", "clean_gpt.csv")
gpt_data <- read.csv(file)

hc_file <- file.path(root_path, "ChatGPT-IGT", "data", "final_data", "clean_ahn_hc.csv")
hc_data <- read.csv(file)

# get vars
source(file.path(root_path, "ChatGPT-IGT", "src", "comparison", "prepare_data.R"))
gpt_vars <- processData(gpt_data)
hc_vars <- processData(hc_data)

# unpack vars (group 1 = hc, group 2 = gpt)
x_grp1 <- hc_vars$x
X_grp1 <- hc_vars$X

x_grp2 <- gpt_vars$x
X_grp2 <- gpt_vars$X

# common vars (should be the same for both groups, therefore just using gpt vars)
ntrials <- gpt_vars$ntrials
nsubs <- gpt_vars$nsubs

# setup jags 
print("Intializing JAGS ...")
jags_data <- list("x_grp1", "X_grp1", "x_grp2", "X_grp2", "ntrials", "nsubs")
params<-c("alpha_a_rew","alpha_a_pun","alpha_K","alpha_omega_f","alpha_omega_p", "alpha_theta")

# run jags
model_file <- file.path(root_path, "ChatGPT-IGT", "models", "hier_ORL_compare.txt")
samples <- jags.parallel(jags_data, inits=NULL, params,
                model.file =model_file,
                n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4)
    
print(samples$BUGSoutput)


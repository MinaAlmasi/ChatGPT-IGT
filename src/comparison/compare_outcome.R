# script inspired by ORL_outcome_compare (UCloud: /Module 4/group_comparison/ORL_compare.R) 
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

# read data
root_path <- "~/Desktop/dm-code" # personal comp
#root_path <- "dm-code" # UCloud
gpt_file <- file.path(root_path, "ChatGPT-IGT", "data", "final_data", "clean_gpt.csv")
gpt_data <- read.csv(gpt_file)

hc_file <- file.path(root_path, "ChatGPT-IGT", "data", "final_data", "clean_ahn_hc.csv")
hc_data <- read.csv(hc_file)

# get vars
source(file.path(root_path, "ChatGPT-IGT", "src", "data_util.R")) # function processData() already scales payoffs for both groups
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
jags_data <- list("x_grp1", "X_grp1", "x_grp2", "X_grp2", "ntrials", "nsubs")
params<-c("mu", "alpha", "Smu_grp1", "Smu_grp2")

# run jags
print("Intializing JAGS ...")
model_file <- file.path(root_path, "ChatGPT-IGT", "models", "outcome_compare.txt")
samples <- jags.parallel(jags_data, inits=NULL, params,
                model.file =model_file,
                n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4)
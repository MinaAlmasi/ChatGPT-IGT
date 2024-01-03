# script inspired by ORL_compare (UCloud: /Module 4/group_comparison/ORL_compare.R) 
pacman::p_load(R2jags, parallel)

# set seed
set.seed(2502)

# set theta
fixed_theta <- TRUE # if FALSE then theta is estimated (included in params and a different model file is used)

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

# set timer
start_iteration = Sys.time()

# setup jags 
jags_data <- list("x_grp1", "X_grp1", "x_grp2", "X_grp2", "ntrials", "nsubs")

if (fixed_theta) {
    params<-c("alpha_a_rew","alpha_a_pun","alpha_K","alpha_omega_f","alpha_omega_p")
    model_file <- file.path(root_path, "ChatGPT-IGT", "models", "hier_ORL_compare_fixed_theta.txt")
} else {
    params<-c("alpha_a_rew","alpha_a_pun","alpha_K","alpha_omega_f","alpha_omega_p", "alpha_theta")
    model_file <- file.path(root_path, "ChatGPT-IGT", "models", "hier_ORL_compare.txt")
}


# run jags
print("Intializing JAGS ...")
samples <- jags.parallel(jags_data, inits=NULL, params,
                model.file =model_file,
                n.chains=3, n.iter=3000, n.burnin=1000, n.thin=1, n.cluster=4)
    
print(samples$BUGSoutput)

# save bugs output to txt
write.table(samples$BUGSoutput$summary, file.path(root_path, "ChatGPT-IGT", "src", "comparison", "results", "alpha_params_comparison_summary.txt"))

# extract alpha parameters
Y <- samples$BUGSoutput$sims.list
alpha_a_rew <- Y$alpha_a_rew
alpha_a_pun <- Y$alpha_a_pun
alpha_K <- Y$alpha_K
alpha_theta <- Y$alpha_theta
alpha_omega_f <- Y$alpha_omega_f
alpha_omega_p <- Y$alpha_omega_p

# time 
end_iteration <- Sys.time()
run_iteration <- round(end_iteration - start_iteration, 2)
print(paste0("Iteration time: ", run_iteration, " minutes"))

# make dataframe
if (fixed_theta) {
   df = data.frame(alpha_a_rew, alpha_a_pun, alpha_K, alpha_omega_f, alpha_omega_p)
} else {
    df = data.frame(alpha_a_rew, alpha_a_pun, alpha_K, alpha_theta, alpha_omega_f, alpha_omega_p)
}
# save df
write.csv(df, file.path(root_path, "ChatGPT-IGT", "src", "comparison", "results", "alpha_params_comparison.csv"))
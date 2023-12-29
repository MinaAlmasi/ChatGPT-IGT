# utility script for processing real data for compare scripts 
processData <- function(data) {
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
        # record n trials for subject s
        ntrials[s] <- length(x_raw[data$subjID==subIDs[s]])
  
        # extract x and X for subject s
        X_sub <- X_raw[data$subjID==subIDs[s]] 
        x_sub <- x_raw[data$subjID==subIDs[s]] 

        # assign arrays
        x[s,] <- x_sub
        X[s,] <- X_sub
    }

    # scale the payoffs
    X <- X / 100

    # return a list of variables
    return(list(x = x, X = X, ntrials = ntrials, nsubs = nsubs))
}
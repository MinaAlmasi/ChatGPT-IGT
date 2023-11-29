'''
Parameter recovery for the ORL 
'''
import numpy as np

from payoff import create_payoff_dict, create_payoff_structure

def intialize_empty_arrays(niterations:int=100, variables:list = ['mu_a_rew', 'mu_a_pun', 'mu_K', 'mu_theta', 
                                                                'mu_omega_f', 'mu_omega_p','lambda_a_rew', 
                                                                'lambda_a_pun', 'lambda_K', 'lambda_theta', 
                                                                'lambda_omega_f', 'lambda_omega_p']):

    '''
    Intialize empty arrays for parameter recovery 

    Args
        niterations: number of iterations to run
        variables: list of variables to recover

    Returns
        data: dictionary of empty arrays for parameter recovery (true and inferred values)
    
    (based on ORLRecovery.R)
    '''
    data = {}

    for var in variables:
        data['true_' + var] = np.empty(niterations)
        data['infer_' + var] = np.empty(niterations)

    return data

def simulate_ORL(payoff, n_trials=100, a_rew=0.3, a_pun=0.3, K=3, theta=3, omega_f=0.7, omega_p=0.7):
    '''
    Simulate ORL 

    from (ORL.R)
    '''
    # arrays to store values
    x = np.zeros(n_trials) # choices
    X = np.zeros(n_trials) # outcomes 
    signX = np.zeros(n_trials)

    Ev_update = np.zeros((n_trials, 4)) # expected value update
    Ev = np.zeros((n_trials, 4)) # expected value

    Ef_cho = np.zeros((n_trials, 4)) # expected frequency update (chosen)
    Ef_not = np.zeros((n_trials, 4)) # expected frequency update (not chosen)
    Ef = np.zeros((n_trials, 4)) # expected frequency

    PS = np.zeros((n_trials, 4)) # perseverance

    V = np.zeros((n_trials, 4)) # valence

    exp_p = np.zeros((n_trials, 4)) # softmax part 1
    p = np.zeros((n_trials, 4)) # softmax part 2

    # initial values
    x[0] = np.random.choice(4, p=np.ones(4) / 4)
    X[0] = payoff[0, int(x[0])]
    Ev[0, :] = np.repeat(0, 4)
    Ef[0, :] = np.repeat(0, 4)
    PS[0, :] = np.repeat(1, 4)

    # loop over trials
    for t in range(1, n_trials):
        signX[t] = np.where(X[t-1] < 0, -1, 1) # if X[t-1] is less than 0, sign is set to -1. Else set to 1. 

        for d in range(4):
            if X[t-1] >= 0:
                Ev_update[t, d] = Ev[t-1, d] + a_rew * (X[t-1] - Ev[t-1, d])

            else:
                Ev_update[t, d] = Ev[t-1, d] + a_pun * (X[t-1] - Ev[t-1, d])

            # update expected frequencies 
            Ef_cho[t, d] = Ef[t-1, d] + a_rew * (signX[t] - Ef[t-1, d])
            Ef_not[t, d] = Ef[t-1, d] + a_pun * (-(signX[t] / 3) - Ef[t-1, d])

            # copy appropriate values to ef variable
            Ef[t, d] = Ef_cho[t, d] if d == x[t-1] else Ef_not[t, d]

            # update perseverance
            PS[t, d] = 1 / (1 + K) if x[t-1] == d else PS[t-1, d] / (1 + K)

            # update valence
            V[t, d] = Ev[t, d] + Ef[t, d] * omega_f + PS[t, d] * omega_p

            # softmax part 1
            exp_p[t, d] = np.exp(theta * V[t, d])

            # softmax part 2
            for d in range(4):
                p[t, d] = exp_p[t, d] / sum(exp_p[t, :])

            x[t] = np.random.choice([0, 1, 2, 3], p=p[t, :])
            X[t] = payoff[t, int(x[t])]

    # store values in dictionary
    data = {'x': x,
            'X': X,
            'Ev': Ev,
            'Ef': Ef,
            'PS': PS}

    return data

def main():
    data = intialize_empty_arrays()

    # define payoff dict with default values
    payoff_dict = create_payoff_dict()

    # create payoff structure
    payoff = create_payoff_structure(payoff_dict)

    # simulate
    data = simulate_ORL(payoff)

    print(data['x'])

if __name__ == '__main__':
    main()
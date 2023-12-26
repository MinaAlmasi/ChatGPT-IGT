'''
functions for simulating ORL data for param recovery
'''

import numpy as np 
import pathlib
import pandas as pd

def translate_payoff(payoff_df):
    '''
    Add outcome col to payoff structure
    '''
    # add outcome col for each
    for letter in ['A', 'B', 'C', 'D']:
        payoff_df[f'{letter}_outcome'] = payoff_df[f'win_{letter}'] - payoff_df[f'loss_{letter}']

    # select only filter only relevant cols
    payoff_df = payoff_df[['A_outcome', 'B_outcome', 'C_outcome', 'D_outcome']]
    
    # make into numpy array
    payoff_array = payoff_df.to_numpy()

    return payoff_array


def simulate_ORL(payoff_array, n_trials, a_rew, a_pun, K, theta, omega_f, omega_p):
    '''
    Simulate ORL data (subject level) to use for parameter recovery
    '''

    # define empty arrays
    x = np.zeros(n_trials).astype(int) # choice
    X = np.zeros(n_trials).astype(int) # outcome / reward
    signX = np.zeros(n_trials) # sign of outcome
    
    EV_update = np.zeros((n_trials, 4))
    EV = np.zeros((n_trials, 4)) # expected value
    EF_chosen = np.zeros((n_trials, 4)) # expected frequency of chosen option
    EF_not = np.zeros((n_trials, 4)) # ... of not chosen option
    EF = np.zeros((n_trials, 4)) # expected frequency
    PS = np.zeros((n_trials, 4)) # probability of switching
    V = np.zeros((n_trials, 4)) # value

    exp_p = np.zeros((n_trials, 4)) # expected probability of choosing each option
    p = np.zeros((n_trials, 4)) # probability of choosing each option

    # set initial choice and payoff 
    x[0] = np.random.choice([0,1,2,3], p=[0.25, 0.25, 0.25, 0.25]) # random choice between 1 and 4 with equal probabilities 
    X[0] = payoff_array[0, x[0]] # set first payoff

    for t in range(1, n_trials): # skip first trial
        # update the sign based on previous outcome
        signX[t] = -1 if X[t-1] < 0 else 1

        for deck in range(4):
            ### EV ### 
            # update expected values using previous outcome and appropriate learning rate
            EV_update[t, deck] = EV[t-1, deck] + a_rew*((X[t-1] - EV[t-1, deck])) if X[t-1] >= 0 else EV[t-1, deck] + a_pun*((X[t-1] - EV[t-1, deck]))

            # update ev for the chosen deck
            EV[t, deck] = EV_update[t, deck] if deck == x[t-1] else EV[t-1, deck]

            ### EF ###
            # update expected frequency of chosen deck
            EF_chosen[t, deck] = EF[t-1, deck] + a_rew * (signX[t] - EF[t-1, deck]) if X[t-1] >= 0 else EF_chosen[t-1, deck]
            
            


def main(): 
    # set random seed
    np.random.seed(2502)

    # define paths
    path = pathlib.Path(__file__)
    payoff_path = path.parents[1] / "utils" / "payoff_scheme_3.csv"

    # load payoff_df
    payoff_df = pd.read_csv(payoff_path)
    
    # translate payoff structure
    payoff_structure = translate_payoff(payoff_df)

    # simulate ORL data
    n_trials = 10
    simulate_ORL(payoff_structure, n_trials, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

if __name__ == "__main__":
    main()
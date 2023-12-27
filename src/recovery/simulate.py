'''
functions for simulating ORL data for param recovery
'''

import numpy as np 
import pathlib
import pandas as pd
from tqdm import tqdm

def translate_payoff(payoff_df, scale=True):
    '''
    Add outcome col to payoff structure

    Args
        payoff_df: payoff structure dataframe
        scale: if True, scale payoff structure by 100
    
    Returns
        payoff_array: payoff structure as numpy array
    '''
    # add outcome col for each
    for letter in ['A', 'B', 'C', 'D']:
        payoff_df[f'{letter}_outcome'] = payoff_df[f'win_{letter}'] - payoff_df[f'loss_{letter}']

    # select only filter only relevant cols
    payoff_df = payoff_df[['A_outcome', 'B_outcome', 'C_outcome', 'D_outcome']]
    
    # make into numpy array
    payoff_array = payoff_df.to_numpy()

    if scale:
        payoff_array = payoff_array/100

    return payoff_array

def simulate_ORL(payoff_array, n_trials, a_rew, a_pun, K, theta, omega_f, omega_p):
    '''
    Simulate ORL data (subject level) to use for parameter recovery

    Function largely based on ORL.R from UCloud module 3. 
    '''
    # define empty arrays
    x = np.full(n_trials, np.nan).astype(int) # choice
    X = np.zeros(n_trials) # outcome / reward
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

    deck_counts = {0: 0, 1: 0, 2: 0, 3: 0} 

    for t in range(1, n_trials): # skip first trial
        # identify which deck was chosen last trial
        last_chosen_deck = x[t-1]
        deck_counts[last_chosen_deck] += 1

        # get available decks (those that have been chosen less than 60 times)
        available_decks = [deck for deck, count in deck_counts.items() if count < 60]
        
        # update the sign based on previous outcome
        signX[t] = -1 if X[t-1] < 0 else 1
        
        for deck in available_decks:
            ### EV ### 
            # update expected values using previous outcome and appropriate learning rate
            EV_update[t, deck] = EV[t-1, deck] + a_rew*((X[t-1] - EV[t-1, deck])) if X[t-1] >= 0 else EV[t-1, deck] + a_pun*((X[t-1] - EV[t-1, deck]))

            # copy appropriate values
            EV[t, deck] = EV_update[t, deck] if deck == x[t-1] else EV[t-1, deck]

            ### EF ###
            # update expected frequency of chosen deck
            EF_chosen[t, deck] = EF[t-1, deck] + a_rew * (signX[t] - EF[t-1, deck]) if X[t-1] >= 0 else EF[t-1, deck] + a_pun * (signX[t] - EF[t-1, deck])

            # update expected frequency of NOT chosen deck
            EF_not[t, deck] = EF[t-1, deck] + a_pun*(-(signX[t]/3) - EF[t-1, deck]) if X[t-1] >= 0 else EF[t-1, deck] + a_rew*(-(signX[t]/3) - EF[t-1, deck])
            
            # copy appropriate values
            EF[t, deck] = EF_chosen[t, deck] if deck == x[t-1] else EF_not[t, deck]

            ### PS ###
            # update probability of switching (ifelse discriminates between chosen and unchosen decks)
            PS[t, deck] = 1/(1+K) if x[t-1]==deck else PS[t-1, deck]/(1+K)

            ### V ###
            # update value of each deck
            V[t, deck] = EV[t, deck] + EF[t, deck]*omega_f + PS[t, deck]*omega_p

            ### SOFTMAX ###
            # calculate expected probability of choosing each deck
            exp_p[t, deck] = np.exp(theta*V[t, deck])

        # calculate probability of choosing each deck
        for deck in available_decks:
            p[t, deck] = exp_p[t, deck]/np.sum(exp_p[t, available_decks])

        # choose deck based on probabilities
        x[t] = np.random.choice(available_decks, p=p[t, available_decks])

        # get index by subtracting 1 from deck count as index starts at 0 but deck count starts at 1
        card_number = deck_counts[x[t]]

        # get payoff corresponding to the choice and which card we are on in that deck
        X[t] = payoff_array[card_number, x[t]]
    
    results = {'x': x, 'X': X, 'n_trials': n_trials} 

    return results

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

def simulate_subject_data(n_iterations, data, payoff_structure, fixed_theta:float=None, save_path:pathlib.Path=None):
    '''
    Run parameter recovery

    Args
        n_iterations: number of iterations to run
        data: dictionary of empty arrays for parameter recovery (true and inferred values)
        payoff_structure: payoff array (translated from payoff_df to include only outcomes)
        fixed_theta: if not None, fix theta to the given value
    '''
    results_list = []

    for i in tqdm(range(n_iterations)):
        # randomly sample parameters
        a_rew = np.random.uniform(0, 1)
        a_pun = np.random.uniform(0, 1)
        K = np.random.uniform(0, 5)
        omega_f = np.random.uniform(0, 1)
        omega_p = np.random.uniform(0, 1)

        # randomly sample theta if not fixed is specified
        if fixed_theta is None:
            theta = np.random.uniform(0, 5)
        else:
            theta = fixed_theta
        
        # run simulation with sampled parameters
        results = simulate_ORL(payoff_structure, 100, a_rew, a_pun, K, theta, omega_f, omega_p)

        # combine results with the sampled parameters
        results['a_rew'] = a_rew
        results['a_pun'] = a_pun
        results['K'] = K
        results['theta'] = theta
        results['omega_f'] = omega_f
        results['omega_p'] = omega_p

        # add results as new row to dataframe
        results_list.append(results)
    
    # convert to df
    results_df = pd.DataFrame(results_list)

    # add 1 to index to match R indexing
    results_df.index += 1

    # write to csv
    if save_path is not None:
        results_df.to_json(save_path)

def main(): 
    # set random seed
    np.random.seed(2502)

    # define paths
    path = pathlib.Path(__file__)
    payoff_path = path.parents[2] / "utils" / "payoff_scheme_3.csv"

    # load payoff_df
    payoff_df = pd.read_csv(payoff_path)
    
    # translate payoff structure
    payoff_structure = translate_payoff(payoff_df)

    # initialize empty arrays
    data = intialize_empty_arrays()

    # simulate ORL data
    simulate_subject_data(100, data, payoff_structure, fixed_theta=None, save_path= path.parents[2] / "src" / "recovery" / "simulated_single_subject_data.json")


if __name__ == "__main__":
    main()
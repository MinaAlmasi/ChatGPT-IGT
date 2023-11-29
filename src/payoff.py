'''
Create payoff structure for ORL task
'''
import numpy as np

def create_payoff_dict(
        n_trials = 100,
        n_struct = 10, 
        freq = 0.5, 
        infreq = 0.1, 
        bad_r = 100,
        bad_freq_l = -250,
        bad_infreq_l = -1250,
        good_r = 50,
        good_freq_l = -50,
        good_infreq_l = -250,
    ):
    '''
    Return a dictionary of payoff values for the ORL task

    Args
        n_trials: number of trials in payoff structure
        n_struct: size of subdivsions for pseudo-randomization
        freq: probability of frequent losses (defaults to losses half of the time)
        infreq: probability of infrequent losses (defaults to losses 10% of the time)
        bad_r: "bad" winnings
        bad_freq_l: "bad" frequent losses
        bad_infreq_l: "bad" infrequent losses
        good_r: "good" winnings
        good_freq_l: "good" frequent
        good_infreq_l: "good" infrequent losses

    Returns 
        payoff_dict: dictionary of payoff values for the ORL task

    (based on ORLRecovery.R)
    '''
    if n_trials % n_struct != 0:
        raise ValueError('n_trials must be divisible by n_struct')

    payoff_dict = {
        'n_trials': n_trials,
        'n_struct': n_struct,
        'freq': freq,
        'infreq': infreq,
        'bad_r': bad_r,
        'bad_freq_l': bad_freq_l,
        'bad_infreq_l': bad_infreq_l,
        'good_r': good_r,
        'good_freq_l': good_freq_l,
        'good_infreq_l': good_infreq_l,
    }

    return payoff_dict

def deck_vals(deck_name:str, n_struct:int, type_r_val:int, type_l_val:int, freq_or_infreq_val:float):
    '''
    Create deck values for R and L for a given deck name

    Args
        deck_name: name of deck (e.g. 'A', 'B', 'C', 'D')
        n_struct: size of subdivsions for pseudo-randomization
        freq: probability of frequent losses 
        infreq: probability of infrequent losses
        type_r_val: value of R for a given deck type (i.e., good vs bad, in the current framework it would either be bad_r or good_r)
        type_l_val: value of L for a given deck type (i.e., good vs bad, in the current framework it would either be bad_freq_l or good_freq_l)
        freq_val: probability of frequent losses f

    (based on ORLRecovery.R)
    '''
    valid_deck_names = ['A', 'B', 'C', 'D']

    if deck_name not in valid_deck_names:
        raise ValueError(f'deck_name must be one of {valid_deck_names}')

    deck_dict = {}

    # Create R and L values for each deck
    deck_dict[f'{deck_name}_R'] = np.repeat(type_r_val, n_struct)
    deck_dict[f'{deck_name}_L'] = np.concatenate(
        (
            np.repeat(type_l_val, int(n_struct * freq_or_infreq_val)),
            np.repeat(0, int(n_struct * (1 - freq_or_infreq_val)))
        )
    )
    
    return deck_dict

def define_deck_values(payoff_dict:dict):
    '''
    Create deck values for all decks

    (based on ORLRecovery.R)
    '''
    # Extract general properties 
    freq = payoff_dict['freq']
    infreq = payoff_dict['infreq']
    n_struct = payoff_dict['n_struct']

    # Bad frequent
    A_vals = deck_vals("A", n_struct, payoff_dict['bad_r'], payoff_dict['bad_freq_l'], freq)

    # Bad infrequent
    B_vals = deck_vals("B", n_struct, payoff_dict['bad_r'], payoff_dict['bad_infreq_l'], infreq)

    # Good frequent
    C_vals = deck_vals("C", n_struct, payoff_dict['good_r'], payoff_dict['good_freq_l'], freq)

    # Good infrequent
    D_vals = deck_vals("D", n_struct, payoff_dict['good_r'], payoff_dict['good_infreq_l'], infreq)

    return A_vals, B_vals, C_vals, D_vals

def create_payoff_structure(payoff_dict:dict):
    '''
    Create payoff structure for ORL 

    Args
        payoff_dict: dictionary of payoff values for the ORL task
    
    Returns
        payoff: payoff structure for ORL task

    (based on ORLRecovery.R)
    '''
    # extract general properties
    n_trials = payoff_dict['n_trials']
    n_struct = payoff_dict['n_struct']

    # setup empty arrays
    A = np.repeat(np.nan, n_trials)
    B = np.repeat(np.nan, n_trials)
    C = np.repeat(np.nan, n_trials)
    D = np.repeat(np.nan, n_trials)

    # define deck values
    A_vals, B_vals, C_vals, D_vals = define_deck_values(payoff_dict)

    # fill arrays
    for i in range(n_trials // n_struct):
        start_idx = i * n_struct
        end_idx = (i + 1) * n_struct

        A[start_idx:end_idx] = A_vals['A_R'] + np.random.choice(A_vals['A_L'], size=n_struct, replace=False)
        B[start_idx:end_idx] = B_vals['B_R'] + np.random.choice(B_vals['B_L'], size=n_struct, replace=False)
        C[start_idx:end_idx] = C_vals['C_R'] + np.random.choice(C_vals['C_L'], size=n_struct, replace=False)
        D[start_idx:end_idx] = D_vals['D_R'] + np.random.choice(D_vals['D_L'], size=n_struct, replace=False)

    # combine for payoff (payoff <- cbind(A,B,C,D))
    payoff = np.column_stack((A, B, C, D))

    # divide by 100 to make it easier to work with
    payoff = payoff / 100

    return payoff 

def main(): 
    # define payoff dict with default values
    payoff_dict = create_payoff_dict()

    print(payoff_dict)

    # create payoff structure
    payoff = create_payoff_structure(payoff_dict)

    # check payoff structure
    print(payoff)
    print(payoff.shape)
    print(payoff.sum(axis=0))

if __name__ == '__main__':
    main()
    
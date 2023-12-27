'''
Subject level simulation
'''
import numpy as np
from tqdm import tqdm
import pandas as pd
import pathlib, sys

# custom imports
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from utils.simulate import simulate_ORL, translate_payoff

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

def parameter_recovery(n_iterations, data, payoff_structure, fixed_theta:float=None):
    '''
    Run parameter recovery

    Args
        n_iterations: number of iterations to run
        data: dictionary of empty arrays for parameter recovery (true and inferred values)
        payoff_structure: payoff array (translated from payoff_df to include only outcomes)
        fixed_theta: if not None, fix theta to the given value
    '''

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

        # print results
        print(results["x"])
    


def main(): 
    # define paths 
    path = pathlib.Path(__file__)
    payoff_path = path.parents[2] / "utils" / "payoff_scheme_3.csv"
    
    # load payoff_df
    payoff_df = pd.read_csv(payoff_path)
    
    # translate payoff structure
    payoff_structure = translate_payoff(payoff_df, scale=True)

    # initliaze empty arrays
    data = intialize_empty_arrays()
    
    # simulate ORL data
    parameter_recovery(100, data, payoff_structure)
    
    
    
if __name__ == "__main__":
    main()
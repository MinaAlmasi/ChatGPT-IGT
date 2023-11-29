'''
Parameter recovery for the ORL 
'''
import numpy as np

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

def main():
    data = intialize_empty_arrays()

    print(data)
    print(data.keys())
    print(data['true_mu_a_rew'].shape)

if __name__ == '__main__':
    main()
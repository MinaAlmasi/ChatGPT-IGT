import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm, gamma, binom

def chance_level(n, alpha = 0.001, p = 0.5):
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level

def sample_truncated_normal(mean, sd, lower, upper, size=1000):
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

def savage_dickey_plot(data, parameters=[("alpha_a_rew", "$\\alpha A_{rew}$"), ("alpha_a_pun", "$\\alpha A_{pun}$"), ("alpha_K", "$\\alpha K$"), ("alpha_theta","$\\alpha \\theta$"), ("alpha_omega_f","$\\alpha \omega_F$"), ("alpha_omega_p", "$\\alpha \omega_P$")], save_path = None):
    # initialize plot
    fig, ax = plt.subplots(2,3, figsize=(15,10))

    # define priors
    priors = {
            "alpha_a_rew": lambda size: sample_truncated_normal(0, 1, -1, 1, size),
            "alpha_a_pun": lambda size: sample_truncated_normal(0, 1, -1, 1, size),
            "alpha_K": lambda size: np.random.normal(0, 1, size),
            "alpha_theta": lambda size: np.random.normal(0, 1, size),
            "alpha_omega_f": lambda size: np.random.normal(0, 1, size),
            "alpha_omega_p": lambda size: np.random.normal(0, 1, size)
            # Add other priors if necessary
        }
    
    # loop over parameters and plot
    for i, (param, name) in enumerate(parameters):
        # define axes
        x = ax[i//3, i%3]

        # plot prior
        sns.kdeplot(priors[param](1000), ax=x, color="black", linestyle="--", label="Prior")

        # plot posterior
        sns.kdeplot(data[param], ax=x, color="red", linestyle="-", label="Posterior")

        # set title
        x.set_title(name)

        # set legend
        x.legend()
    
    if save_path is not None:
        plt.savefig(save_path)



def main():
    # define paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "src" / "comparison" / "results"

    # load data
    comparison_data = pd.read_csv(data_path / "alpha_params_comparison.csv")

    # plot
    savage_dickey_plot(comparison_data, save_path=path.parents[2] / "src" / "comparison" / "plots" / "alpha_comparison.png")

if __name__ == "__main__":
    main()
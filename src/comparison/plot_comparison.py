import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm, gamma, binom, gaussian_kde
from matplotlib.font_manager import FontManager

def chance_level(n, alpha = 0.001, p = 0.5):
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level

def set_font():
    available_fonts = set(f.name for f in FontManager().ttflist)
    if 'Times New Roman' in available_fonts:
        # use times new roman if available
        plt.rcParams['font.family'] = 'Times New Roman'
        # set font size to 13
        plt.rcParams.update({'font.size': 13})

def sample_truncated_normal(mean, sd, lower, upper, size=1000):
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

def savage_dickey_plot_orl(data, parameters=[("alpha_a_rew", "$\\alpha A_{rew}$"), ("alpha_a_pun", "$\\alpha A_{pun}$"), ("alpha_K", "$\\alpha K$"), ("alpha_omega_f","$\\alpha \omega_F$"), ("alpha_omega_p", "$\\alpha \omega_P$")], save_path=None):
    '''
    Plot the prior and posterior distributions of the parameters of the ORL model. Note that theta is not included as it is not estimated.
    '''
    # determine num of rows and cols 
    num_params = len(parameters)
    num_cols = 3
    num_rows = num_params // num_cols + (num_params % num_cols > 0)

    # define plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    if num_rows > 1:  # If more than one row, flatten the axes array
        axes = axes.flatten()

    # define priors
    priors = {
        "alpha_a_rew": lambda size: sample_truncated_normal(0, 1, -1, 1, size),
        "alpha_a_pun": lambda size: sample_truncated_normal(0, 1, -1, 1, size),
        "alpha_K": lambda size: np.random.normal(0, 1, size),
        "alpha_omega_f": lambda size: np.random.normal(0, 1, size),
        "alpha_omega_p": lambda size: np.random.normal(0, 1, size)
    }

    # loop over params
    for i, (param, name) in enumerate(parameters):
        ax = axes[i] if num_rows > 1 else axes

        # generate prior and posterior samples
        prior_samples = priors[param](1000)
        posterior_samples = data[param]

        # create KDE for prior and posterior
        prior_kde = gaussian_kde(prior_samples)
        posterior_kde = gaussian_kde(posterior_samples)

        # compute densities at the critical value 0
        prior_density_at_0 = prior_kde(0)
        posterior_density_at_0 = posterior_kde(0)

        # compute Savage-Dickey density ratio
        BF = prior_density_at_0 / posterior_density_at_0

        # plot prior and posterior
        sns.kdeplot(prior_samples, ax=ax, color="black", linestyle="--", label="Prior")
        sns.kdeplot(posterior_samples, ax=ax, color="red", linestyle="-", label="Posterior")

        # set title and labels
        ax.set_title(name)
        ax.set_xlabel("")

        # write Savage-Dickey in plot
        ax.text(0.05, 0.95, f"BF: {BF[0]:.2f}", transform=ax.transAxes, verticalalignment='top')

        # only add legend to the first plot
        if i == 0:
            ax.legend()

    # rm empty subplots if the number of parameters is less than the number of subplots
    for j in range(num_params, num_rows * num_cols):
        if num_rows > 1:
            fig.delaxes(axes[j])
        else:
            fig.delaxes(axes)

    # adjust layout and save plot if a path is provided
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

def savage_dickey_plot_outcome(data, parameters=[("alpha", "$\\alpha X$")], save_path=None):
    # Define a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

    # Define priors
    priors = {
        "alpha": lambda size: np.random.normal(0, 1/np.sqrt(0.1), size)
        # Add other priors if necessary
    }

    # Loop over parameters and plot
    for i, (param, name) in enumerate(parameters):
        # generate prior and posterior samples
        prior_samples = priors[param](1000)
        posterior_samples = data[param]

        # create KDE for prior and posterior
        prior_kde = gaussian_kde(prior_samples)
        posterior_kde = gaussian_kde(posterior_samples)

        # calculate densities at the critical value 0
        prior_density_at_0 = prior_kde(0)
        posterior_density_at_0 = posterior_kde(0)

        # calculate Savage-Dickey density ratio
        BF = posterior_density_at_0 / prior_density_at_0

        # prior
        sns.kdeplot(prior_samples, ax=axs[0], color="black", linestyle="--")
        axs[0].set_title(f"Prior {name}")
        axs[0].axvline(x=0, color="black", linestyle="-")

        # posterior
        sns.kdeplot(posterior_samples, ax=axs[1], color="red", linestyle="-")
        axs[1].set_title(f"Posterior {name}")
        axs[1].axvline(x=0, color="black", linestyle="-")

        # combined
        sns.kdeplot(prior_samples, ax=axs[2], color="black", linestyle="--", label="Prior")
        sns.kdeplot(posterior_samples, ax=axs[2], color="red", linestyle="-", label="Posterior")
        axs[2].set_title(f"Combined {name}")

        # Annotate Savage-Dickey density ratio on combined plot
        axs[2].text(0.05, 0.95, f"BF: {BF[0]:.2f}", transform=axs[2].transAxes, verticalalignment='top')

    # rm x-labels if needed
    for ax in axs:
        ax.set_xlabel("")

    # save plot if a path is provided
    if save_path is not None:
        plt.savefig(save_path)


def main():
    # define paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "src" / "comparison" / "results"

    # load data
    orl_comparison_data = pd.read_csv(data_path / "alpha_params_comparison.csv")

    # load outcome data
    outcome_comparison_data = pd.read_csv(data_path / "outcome_params.csv")
    
    # set font
    set_font()

    # plot
    savage_dickey_plot_orl(orl_comparison_data, save_path=path.parents[2] / "src" / "comparison" / "plots" / "alpha_comparison_orl.png")
    savage_dickey_plot_outcome(outcome_comparison_data, save_path=path.parents[2] / "src" / "comparison" / "plots" / "alpha_comparison_outcome.png")

if __name__ == "__main__":
    main()
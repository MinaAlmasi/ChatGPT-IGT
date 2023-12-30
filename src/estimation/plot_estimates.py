import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm, gamma

def sample_truncated_normal(mean, sd, lower, upper, size=1000):
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

def plot_posteriors(hc_data, gpt_data, parameters=[("mu_a_rew", "$\mu A_{rew}$"), ("mu_a_pun", "$\mu A_{pun}$"), ("mu_K", "$\mu K$"), ("mu_theta","$\mu \\theta$"), ("mu_omega_f","$\mu \omega_F$"), ("mu_omega_p", "$\mu \omega_P$")], save_path=None, show_priors=False):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    # Define prior distributions only if show_priors is True
    priors = None
    if show_priors:
        priors = {
            "mu_a_rew": lambda size: sample_truncated_normal(0, 1, 0, 1, size),
            "mu_a_pun": lambda size: sample_truncated_normal(0, 1, 0, 1, size),
            "mu_K": lambda size: sample_truncated_normal(0, 1, 0, np.inf, size),
            "mu_theta": lambda size: sample_truncated_normal(0, 1, 0, np.inf, size),
            "mu_omega_f": lambda size: np.random.normal(0, 0.1, size),
            "mu_omega_p": lambda size: np.random.normal(0, 0.1, size),
            # Add other priors if necessary
        }

    for i, (param_name, param_title) in enumerate(parameters):
        ax = axs[i // 2, i % 2]
        sns.kdeplot(hc_data[param_name], ax=ax, fill=True, alpha=0.5, label="HC")
        sns.kdeplot(gpt_data[param_name], ax=ax, fill=True, alpha=0.5, label="GPT")

        # Plot prior if show_priors is True
        if show_priors and priors:
            prior_samples = priors[param_name](1000)
            sns.kdeplot(prior_samples, ax=ax, color="k", linestyle="--", alpha=0.5, label="Prior")

        ax.set_title(param_title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()






def main(): 
    # define paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "src" / "estimation" / "estimated_parameters"
    
    # load data
    hc_data = pd.read_csv(data_path / "param_estimated_ahn_hc.csv")
    gpt_data = pd.read_csv(data_path / "param_estimated_gpt.csv")

    # plot posteriors
    plot_posteriors(hc_data, gpt_data, save_path=path.parents[2] / "src" / "estimation" / "plots", show_priors=True)

if __name__ == "__main__":
    main()
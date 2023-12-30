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
            "mu_omega_f": lambda size: np.random.normal(0, 1, size),
            "mu_omega_p": lambda size: np.random.normal(0, 1, size),
            # Add other priors if necessary
        }

    for i, (param_name, param_title) in enumerate(parameters):
        ax = axs[i // 2, i % 2]
        sns.kdeplot(hc_data[param_name], ax=ax, fill=True, alpha=0.5, label="HC", color="#23AB2D")
        sns.kdeplot(gpt_data[param_name], ax=ax, fill=True, alpha=0.5, label="GPT", color="#5dc5e3")

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

def plot_multiple_descriptive_adequacies(df, save_path=None):
    '''
    Plot the descriptive adequacy of the model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the accuracies in choice estimations for each subject
    
    save_path : str
        Path to save the plot.
    '''
    # divide pred success by 100 to get percentage
    df['pred_success'] = df['pred_success'] / 100

    # Set subplot
    fig, ax = plt.subplots(figsize=(10, 10))

    # sort bar chart by accuracy descending
    df = df.sort_values(by=['pred_success'], ascending=False).reset_index(drop=True)

    # Create bar plot with each subject (row) on the x-axis and the accuracy on the y-axis
    ax.bar(df.index, df['pred_success'], color='darkgrey')  # Set bars to dark grey

    # Add a line for average accuracy across subjects, set to black
    ax.axhline(y=df['pred_success'].mean(), linestyle='-', color='black')

    # Add a dotted line for chance level at 25%, set to black
    ax.axhline(y=chance_level(n=60, p=0.25, alpha=0.05), linestyle='--', color='black')

    # add legend for the two lines in the top right corner
    ax.legend(['Average Accuracy', 'Chance Level'], loc='upper right')

    # Set x-ticks to start at 0 and then at every 10th subject
    ax.set_xticks(range(0, len(df.index), 10))
    ax.set_xticklabels(range(0, len(df.index), 10))

    ax.set_xlabel("Subject")
    ax.set_ylabel("Choice Estimation Accuracy")

    # Limit x-axis to just cover the range of subjects, eliminating extra space on sides
    ax.set_xlim(-0.5, len(df.index) - 0.5)

    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()  # Add this to display the plot when not saving



def main(): 
    # define paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "src" / "estimation" / "results"
    
    # load parameter estimation data
    hc_data_params = pd.read_csv(data_path / "param_estimated_ahn_hc.csv")
    gpt_data_params = pd.read_csv(data_path / "param_estimated_gpt.csv")

    # plot posteriors
    plot_posteriors(hc_data_params, gpt_data_params, save_path=path.parents[2] / "src" / "estimation" / "plots" / "posterior_compare_w_priors", show_priors=True)

    # load posterior predictions data
    #hc_data_pred = pd.read_csv(data_path / "pred_success_ahn_hc.csv")
    gpt_data_pred = pd.read_csv(data_path / "pred_success_gpt.csv")

    # plot descriptive adequacy
    #plot_multiple_descriptive_adequacies(hc_data_pred, save_path=path.parents[2] / "src" / "estimation" / "plots" / "descriptive_adequacy_hc")
    plot_multiple_descriptive_adequacies(gpt_data_pred)



if __name__ == "__main__":
    main()
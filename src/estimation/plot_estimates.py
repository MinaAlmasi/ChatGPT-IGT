import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import truncnorm, gamma, binom
import matplotlib.lines as mlines
from matplotlib.font_manager import FontManager

def set_font():
    available_fonts = set(f.name for f in FontManager().ttflist)
    if 'Times New Roman' in available_fonts:
        # use times new roman if available
        plt.rcParams['font.family'] = 'Times New Roman'
        # set font size to 13
        plt.rcParams.update({'font.size': 13})

def chance_level(n, alpha = 0.001, p = 0.5):
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level

def sample_truncated_normal(mean, sd, lower, upper, size=1000):
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)


def plot_posteriors(hc_data, gpt_data, colors=["#398A20", "#20398A"], 
                    parameters=[("mu_a_rew", "$\mu A_{rew}$"), ("mu_a_pun", "$\mu A_{pun}$"), 
                    ("mu_K", "$\mu K$"), ("mu_omega_f","$\mu \omega_F$"), ("mu_omega_p", "$\mu \omega_P$")], 
                    save_path=None, show_priors=False):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    # Define prior distributions only if show_priors is True
    priors = None
    if show_priors:
        # Assume sample_truncated_normal and other necessary functions are defined
        priors = {
            "mu_a_rew": lambda size: sample_truncated_normal(0, 1, 0, 1, size),
            "mu_a_pun": lambda size: sample_truncated_normal(0, 1, 0, 1, size),
            "mu_K": lambda size: sample_truncated_normal(0, 1, 0, np.inf, size),
            "mu_omega_f": lambda size: np.random.normal(0, 1/np.sqrt(0.1), size),
            "mu_omega_p": lambda size: np.random.normal(0, 1/np.sqrt(0.1), size)
        }

    total_subplots = len(axs.flatten())
    for i, (param_name, param_title) in enumerate(parameters):
        ax = axs[i // 2, i % 2]
        sns.kdeplot(hc_data[param_name], ax=ax, fill=True, alpha=0.5, color=colors[0])
        sns.kdeplot(gpt_data[param_name], ax=ax, fill=True, alpha=0.5, color=colors[1])

        if show_priors and priors:
            prior_samples = priors[param_name](1000)
            sns.kdeplot(prior_samples, ax=ax, color="k", linestyle="--", alpha=0.5)

        ax.set_title(param_title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    # Create a single legend at the top of the figure
    labels = ["Humans", "ChatGPT"]
    if show_priors:
        labels.append("Prior")
    fig.legend(labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 0.98), fancybox=True)

    # Hide the last subplot if the number of parameters is less than total subplots
    if len(parameters) < total_subplots:
        axs[-1, -1].axis('off')


def plot_hc_posteriors(main_hc_sample, other_hc_samples, 
                       parameters=[("mu_a_rew", "$\mu A_{rew}$"), ("mu_a_pun", "$\mu A_{pun}$"), 
                                   ("mu_K", "$\mu K$"), ("mu_omega_f","$\mu \omega_F$"), ("mu_omega_p", "$\mu \omega_P$")], 
                       save_path=None, show_priors=False):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    main_color = "#398A20"  # Green for the main dataframe
    other_color = "#D3D3D3"  # Light grey for other dataframes

    priors = None
    if show_priors:
        # Assume sample_truncated_normal and other necessary functions are defined
        priors = {
            "mu_a_rew": lambda size: sample_truncated_normal(0, 1, 0, 1, size),
            "mu_a_pun": lambda size: sample_truncated_normal(0, 1, 0, 1, size),
            "mu_K": lambda size: sample_truncated_normal(0, 1, 0, np.inf, size),
            "mu_omega_f": lambda size: np.random.normal(0, 1/np.sqrt(0.1), size),
            "mu_omega_p": lambda size: np.random.normal(0, 1/np.sqrt(0.1), size)
        }

    total_parameters = len(parameters)
    for i, (param_name, param_title) in enumerate(parameters):
        ax = axs[i // 2, i % 2]

        for other_df in other_hc_samples:
            sns.kdeplot(other_df[param_name], ax=ax, fill=True, alpha=0.5, color=other_color)

        if show_priors and priors:
            prior_samples = priors[param_name](1000)
            sns.kdeplot(prior_samples, ax=ax, color="k", linestyle="--", alpha=0.5)

        sns.kdeplot(main_hc_sample[param_name], ax=ax, fill=True, alpha=0.5, color=main_color)

        ax.set_title(param_title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    # Hide the last subplot if the number of parameters is odd
    if total_parameters % 2 != 0:
        axs[-1, -1].axis('off')

    # Manually create legend entries
    selected_sample_line = mlines.Line2D([], [], color=main_color, label='Selected Sample', linestyle='-', linewidth=2)
    other_samples_line = mlines.Line2D([], [], color=other_color, label='Other Samples', linestyle='-', linewidth=2)
    prior_line = mlines.Line2D([], [], color='k', label='Prior', linestyle='--', linewidth=2)

    legend_handles = [selected_sample_line, other_samples_line]
    if show_priors:
        legend_handles.append(prior_line)

    # Create a single legend for the entire figure
    fig.legend(handles=legend_handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98), fancybox=True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)


def plot_multiple_descriptive_adequacies(hc_data, gpt_data, colors = ["#52993C", "#3C5299"], save_path=None):
    '''
    Plot the descriptive adequacy of the model.

    Parameters
    ----------
    hc_data, gpt_data : pd.DataFrame
        Dataframes containing the accuracies in choice estimations for each subject
    
    save_path : str
        Path to save the plot.
    '''
    # combine the two dataframes and create an indicator variable for group
    df = pd.concat([hc_data, gpt_data], ignore_index=True)
    df['group'] = ['Humans'] * len(hc_data) + ['ChatGPT'] * len(gpt_data)

    # divide pred success by 100 to get percentage
    df['pred_success'] = df['pred_success'] / 100

    # Set subplot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create bar plot with each subject (row) on the x-axis and the accuracy on the y-axis and colored by group
    sns.barplot(x=df.index, y=df['pred_success'], hue=df['group'], ax=ax, palette=[colors[0], colors[1]])

    # Calculate mean accuracies
    hc_mean = df[df['group'] == 'Humans']['pred_success'].mean()
    gpt_mean = df[df['group'] == 'ChatGPT']['pred_success'].mean()

    # Calculate the mid-point between the two groups on the x-axis
    mid_point = len(hc_data) - 0.5

    # set a darker version of the colors for the mean spans
    colors_dark = ["#1a6403", "#011554"]

    # Add mean accuracy spans for each group
    ax.axhspan(hc_mean-0.002, hc_mean+0.002, xmin=0, xmax=mid_point/(len(df.index) - 1.2), color=colors_dark[0], alpha=1)
    ax.axhspan(gpt_mean-0.002, gpt_mean+0.002, xmin=mid_point/(len(df.index) - 1.2), xmax=1, color=colors_dark[1], alpha=1)

    # Add a dotted line for chance level at 25%
    ax.axhline(y=0.25, linestyle='--', color='black')

    # Set x-ticks
    ax.set_xticks(range(0, len(df.index), 10))
    ax.set_xticklabels(range(0, len(df.index), 10))

    ax.set_xlabel("Subject")
    ax.set_ylabel("Choice Estimation Accuracy")

    ax.set_xlim(-0.5, len(df.index) - 0.5)

    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([plt.Line2D([], [], color=colors_dark[0], linestyle='-'),
                    plt.Line2D([], [], color=colors_dark[1], linestyle='-'),
                    plt.Line2D([], [], color='black', linestyle='--')])
    labels.extend(['Humans (Mean)', 'ChatGPT (Mean)', 'Chance Level'])
    ax.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.1), fancybox=True)

    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



def main(): 
    # define paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "src" / "estimation" / "results"
    
    # load parameter estimation data
    hc_data_params = pd.read_csv(data_path / "param_estimated_ahn_hc.csv")
    gpt_data_params = pd.read_csv(data_path / "param_estimated_gpt.csv")

    # set global font to times if available
    set_font()

    # plot posteriors
    plot_posteriors(hc_data_params, gpt_data_params, save_path=path.parents[2] / "src" / "estimation" / "plots" / "posterior_compare_w_priors", show_priors=True)

     # load all csv files in the extra_samples subfolder in results
    hc_samples = []
    for file in (data_path / "extra_samples").glob("*.csv"):
        hc_samples.append(pd.read_csv(file))
    
    # plot posteriors
    plot_hc_posteriors(hc_data_params, hc_samples, save_path=path.parents[2] / "src" / "estimation" / "plots" / "posterior_compare_hc")

    # load posterior predictions data
    hc_data_pred = pd.read_csv(data_path / "pred_success_ahn_hc.csv")
    gpt_data_pred = pd.read_csv(data_path / "pred_success_gpt.csv")

    # plot descriptive adequacy
    plot_multiple_descriptive_adequacies(hc_data_pred, gpt_data_pred, save_path=path.parents[2] / "src" / "estimation" / "plots" / "descriptive_adequacy")



if __name__ == "__main__":
    main()
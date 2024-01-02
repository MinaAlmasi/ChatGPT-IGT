'''
Plot the recovery 
'''
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.stats import binom

def chance_level(n, alpha = 0.001, p = 0.5):
    k = binom.ppf(1-alpha, n, p)
    chance_level = k/n
    
    return chance_level

def plot_parameter(ax, df, true_col, infer_col, title, col="#7C7D7C"):
    '''
    Helper function to plot parameter recovery.
    '''
    ax.scatter(df[true_col], df[infer_col], c=col)
    ax.set_title(title)
    ax.set_xlabel("True Value")
    ax.set_ylabel("Estimated Value")
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="black")  # Add diagonal line
    m, b = np.polyfit(df[true_col], df[infer_col], 1)
    ax.plot(df[true_col], m*df[true_col] + b, c=col)  # Add regression line

def plot_recovery(df, parameters:list, subplot_dims:tuple=(3, 2), save_path=None):
    '''
    Plot the recovery of parameters.

    Args
        df: Dataframe containing the recovered parameters.
        subplot_dims: Dimensions of the subplot grid.
        parameters: List of tuples containing the column names of the (1) true and (2) inferred parameters and the (3) title of the subplot.
        save_path: Full path to save the plot (defaults to None, i.e., no saving)

    Returns
        plt
    '''

    # define subplot
    fig, axs = plt.subplots(subplot_dims[0], subplot_dims[1], figsize=(10, 10))

    # adjust spacing
    plt.subplots_adjust(hspace=0.5)

    # plot each parameter
    for i, (true_col, infer_col, title) in enumerate(parameters):
        ax = axs[i // subplot_dims[1], i % subplot_dims[1]]
        plot_parameter(ax, df, true_col, infer_col, title)

    # check if there are any empty subplots
    if subplot_dims[0] * subplot_dims[1] > len(parameters):
        for i in range(len(parameters), subplot_dims[0] * subplot_dims[1]):
            ax = axs[i // subplot_dims[1], i % subplot_dims[1]]
            ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return plt

def preprocess_descriptive_adaquacy(true_parameter_data, subject_data):
    # fix formatting of true_parameter_data to fit with subject_data
    expanded_true_parameter_data = pd.concat([pd.DataFrame(row.x).T for index, row in true_parameter_data.iterrows()], ignore_index=True)
    expanded_true_parameter_data.columns = [f'X{i+1}' for i in range(expanded_true_parameter_data.shape[1])]
    true_parameter_data = pd.concat([true_parameter_data.reset_index(drop=True), expanded_true_parameter_data], axis=1)

    # assuming df1 and df2 are your two DataFrames
    accuracies = []

    # filter columns that start with 'X' followed by a number
    x_columns = [col for col in true_parameter_data.columns if re.match(r'X\d+', col)]

    # iterate over each row
    for i in range(len(true_parameter_data)):
        # select only the relevant 'X' columns for the current row in both DataFrames
        row_df1 = true_parameter_data[x_columns].iloc[i]
        row_df2 = subject_data[x_columns].iloc[i]

        # plus one to all in row_df1 to account for the fact that the parameters are not 1-4 but 0-3
        row_df1 = row_df1 + 1

        # count number of relevant 'X' columns with the same value
        same_count = sum(row_df1 == row_df2)

        # compute accuracy
        accuracy = same_count / len(x_columns)
        accuracies.append(accuracy)

    # add the accuracy list as a new column to one of the DataFrames
    true_parameter_data['Accuracy'] = accuracies

    return true_parameter_data

def plot_descriptive_adequacy(df, save_path=None):
    '''
    Plot the descriptive adequacy of the model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the accuracies in choice estimations for each subject
    
    save_path : str
        Path to save the plot.
    '''
    # Set subplot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create bar plot with each subject (row) on the x-axis and the accuracy on the y-axis
    ax.bar(df.index, df['Accuracy'], color='#7C7D7C')  # Set bars to dark grey

    # Add a line for average accuracy across subjects, set to black
    ax.axhline(y=df['Accuracy'].mean(), linestyle='-', color='black')

    # Add a dotted line for chance level at 25%, set to black
    ax.axhline(y=chance_level(n=100, p=0.25, alpha=0.05), linestyle='--', color='black')

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

    return plt


def main(): 
    # define paths
    path = pathlib.Path(__file__)
    single_subj_path = path.parents[2] / "src" / "recovery" / "results" / "param_recovery_single_subject.csv"
    single_subj_fixed_theta_path = path.parents[2] / "src" / "recovery" / "results" / "param_recovery_single_subject_fixed_theta.csv"
    group_path = path.parents[2] / "src" / "recovery" / "results" / "param_recovery_group_ALL.csv"
    group_fixed_theta_path = path.parents[2] / "src" / "recovery" / "results" / "param_recovery_group_ALL_fixed_theta.csv"
    comparison_path = path.parents[2] / "src" / "recovery" / "results" / "param_recovery_group_comparisons.csv"
    comparison_fixed_theta_path = path.parents[2] / "src" / "recovery" / "results" / "param_recovery_group_comparisons_fixed_theta.csv"

    # load recovered parameters
    subject_data = pd.read_csv(single_subj_path)
    subject_data_fixed_theta = pd.read_csv(single_subj_fixed_theta_path)
    group_data = pd.read_csv(group_path)
    group_data_fixed_theta = pd.read_csv(group_fixed_theta_path)
    comparison_data = pd.read_csv(comparison_path)
    comparison_data_fixed_theta = pd.read_csv(comparison_fixed_theta_path)

    # plot subject recovery
    subject_parameters = [
        ("true_a_rew", "infer_a_rew", "$A_{rew}$"),
        ("true_a_pun", "infer_a_pun", "$A_{pun}$"),
        ("true_omega_f", "infer_omega_f", "$\omega_F$"),
        ("true_omega_p", "infer_omega_p", "$\omega_P$"),
        ("true_theta", "infer_theta", "$\\theta$"),
        ("true_K", "infer_K", "$K$")
    ]
    
    plot_recovery(subject_data, parameters = subject_parameters, save_path = path.parents[2] / "src" / "recovery" / "plots" / "param_recovery_single_subject.png")

    # plot subject recovery but for recovered parameters without theta
    subject_parameters_no_theta = [x for x in subject_parameters if x[0] != "true_theta"]
    plot_recovery(subject_data_fixed_theta, parameters = subject_parameters_no_theta, save_path = path.parents[2] / "src" / "recovery" / "plots" / "param_recovery_single_subject_fixed_theta.png")

    # plot parameters for group recovery
    group_parameters = [
        ("true_mu_a_rew", "infer_mu_a_rew", "$\mu A_{rew}$"),
        ("true_mu_a_pun", "infer_mu_a_pun", "$\mu A_{pun}$"),
        ("true_mu_omega_f", "infer_mu_omega_f", "$\mu \omega_F$"),
        ("true_mu_omega_p", "infer_mu_omega_p", "$\mu \omega_P$"),
        ("true_mu_theta", "infer_mu_theta", "$\mu \\theta$"),
        ("true_mu_K", "infer_mu_K", "$\mu K$")
    ]

    plot_recovery(group_data, parameters = group_parameters, save_path = path.parents[2] / "src" / "recovery" / "plots" / "param_recovery_group.png")

    # plot parameters for group recovery but for recovered parameters without theta
    group_parameters_no_theta = [x for x in group_parameters if x[0] != "true_mu_theta"]

    plot_recovery(group_data_fixed_theta, parameters = group_parameters_no_theta, save_path = path.parents[2] / "src" / "recovery" / "plots" / "param_recovery_group_fixed_theta.png")


    # plot recovery for group comparisons
    comparison_parameters = [
        ("true_alpha_a_rew", "infer_alpha_a_rew", "$\\alpha A_{rew}$"),
        ("true_alpha_a_pun", "infer_alpha_a_pun", "$\\alpha A_{pun}$"),
        ("true_alpha_omega_f", "infer_alpha_omega_f", "$\\alpha \omega_F$"),
        ("true_alpha_omega_p", "infer_alpha_omega_p", "$\\alpha \omega_P$"),
        ("true_alpha_theta", "infer_alpha_theta", "$\\alpha \\theta$"),
        ("true_alpha_K", "infer_alpha_K", "$\\alpha K$")
    ]

    plot_recovery(comparison_data, parameters = comparison_parameters, save_path = path.parents[2] / "src" / "recovery" / "plots" / "param_recovery_group_comparison.png")

    # plot recovery for group comparisons but for recovered parameters without theta
    comparison_parameters_no_theta = [x for x in comparison_parameters if x[0] != "true_alpha_theta"]

    plot_recovery(comparison_data_fixed_theta, parameters = comparison_parameters_no_theta, save_path = path.parents[2] / "src" / "recovery" / "plots" / "param_recovery_group_comparison_fixed_theta.png")

    # load actual parameters
    true_parameter_data = pd.read_json(path.parents[2] / "src" / "recovery" / "simulated_data" / "simulated_single_subject_data.json") 

    # run descriptive adequacy
    desc_ada_data = preprocess_descriptive_adaquacy(true_parameter_data, subject_data)
    plot_descriptive_adequacy(desc_ada_data, save_path = path.parents[2] / "src" / "recovery" / "plots" / "descriptive_adequacy.png")


if __name__ == "__main__":
    main()
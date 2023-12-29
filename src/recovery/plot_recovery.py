'''
Plot the recovery 
'''
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_parameter(ax, df, true_col, infer_col, title, col="#343795"):
    """
    Helper function to plot parameter recovery.
    """
    ax.scatter(df[true_col], df[infer_col], c=col)
    ax.set_title(title)
    ax.set_xlabel("True Value")
    ax.set_ylabel("Estimated Value")
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="black")  # Add diagonal line
    m, b = np.polyfit(df[true_col], df[infer_col], 1)
    ax.plot(df[true_col], m*df[true_col] + b, c=col)  # Add regression line

def plot_recovery(df, parameters, subplot_dims=(3, 2), save_path=None):
    '''
    Plot the recovery of parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the recovered parameters.
    
    subplot_dims : tuple
        Dimensions of the subplot grid.
    
    parameters : list
        List of tuples containing the column names of the (1) true and (2) inferred parameters and the (3) title of the subplot.
    
    save_path : str
        Path to save the plot.
    '''
    fig, axs = plt.subplots(subplot_dims[0], subplot_dims[1], figsize=(10, 10))

    plt.subplots_adjust(hspace=0.5)

    for i, (true_col, infer_col, title) in enumerate(parameters):
        ax = axs[i // subplot_dims[1], i % subplot_dims[1]]
        plot_parameter(ax, df, true_col, infer_col, title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

def main(): 
    # define paths
    path = pathlib.Path(__file__)
    single_subj_path = path.parents[2] / "src" / "recovery" / "recovered_parameters" / "param_recovery_single_subject.csv"
    group_path = path.parents[2] / "src" / "recovery" / "recovered_parameters" / "param_recovery_group_ALL.csv"

    # load recovered parameters
    subject_data = pd.read_csv(single_subj_path)
    group_data = pd.read_csv(group_path)
    

    # set parameters for subject recovery
    subject_parameters = [
        ("true_a_rew", "infer_a_rew", "$A_{rew}$"),
        ("true_a_pun", "infer_a_pun", "$A_{pun}$"),
        ("true_omega_f", "infer_omega_f", "$\omega_F$"),
        ("true_omega_p", "infer_omega_p", "$\omega_P$"),
        ("true_theta", "infer_theta", "$\\theta$"),
        ("true_K", "infer_K", "$K$")
    ]
    
    # run subject recovery
    plot_recovery(subject_data, parameters = subject_parameters, save_path = path.parents[2] / "src" / "recovery" / "plots" / "param_recovery_single_subject.png")

    # set parameters for group recovery
    group_parameters = [
        ("true_mu_a_rew", "infer_mu_a_rew", "$\mu A_{rew}$"),
        ("true_mu_a_pun", "infer_mu_a_pun", "$\mu A_{pun}$"),
        ("true_mu_omega_f", "infer_mu_omega_f", "$\mu \omega_F$"),
        ("true_mu_omega_p", "infer_mu_omega_p", "$\mu \omega_P$"),
        ("true_mu_theta", "infer_mu_theta", "$\mu \\theta$"),
        ("true_mu_K", "infer_mu_K", "$\mu K$")
    ]

    plot_recovery(group_data, parameters = group_parameters, save_path = path.parents[2] / "src" / "recovery" / "plots" / "param_recovery_group.png")


if __name__ == "__main__":
    main()
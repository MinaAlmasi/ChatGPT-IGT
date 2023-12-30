import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def plot_posteriors(hc_data, gpt_data, parameters=["mu_a_rew", "mu_a_pun", "mu_K", "mu_theta", "mu_omega_f", "mu_omega_p"], save_path=None):
    # initialize subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    # plot posteriors as densities using KDE
    for i, parameter in enumerate(parameters):
        ax = axs[i // 2, i % 2]
        sns.kdeplot(hc_data[parameter], ax=ax, fill=True, alpha=0.5, label="HC")
        sns.kdeplot(gpt_data[parameter], ax=ax, fill=True, alpha=0.5, label="GPT")
        ax.set_title(parameter)
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
    plot_posteriors(hc_data, gpt_data)

if __name__ == "__main__":
    main()
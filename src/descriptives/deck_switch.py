import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.stats import binom

def calculate_deck_switches(data):
    # create a new column that indicates the trial number
    data['trial'] = data.groupby('subjID').cumcount() + 1

    # create a new column that indicates the deck chosen on the previous trial
    data['x_last_trial'] = data.groupby('subjID')['x'].shift()

    # ensure new column is int (unless it is NaN)
    data['x_last_trial'] = data['x_last_trial'].fillna(0).astype(int)

    # create a column that indicates whether the deck was switched
    data['deck_switch'] = data['x'] != data['x_last_trial']

    # create a new dataframe that indicates the proportion of subjects that switched decks on each trial
    deck_switch_prop = data.groupby('trial')['deck_switch'].mean().reset_index()

    # disregard trial one
    deck_switch_prop = deck_switch_prop[deck_switch_prop['trial'] != 1]

    return deck_switch_prop

def plot_deck_switch_with_confidence(data1, data2, labels, window_size=10, save_path=None):
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors
    colors = ["#52993C", "#3C5299"]
    alpha = 0.3  # Transparency for confidence interval

    # set x-axis ticks to every 10th trial
    ax.set_xticks(np.arange(0, 101, 10))

    # set y-axis ticks to every 0.1
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # add very light grid
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    lines = []  # To store line objects for legend
    for i, data in enumerate([data1, data2]):
        # Apply smoothing
        smooth_data = data['deck_switch'].rolling(window=window_size, min_periods=1).mean()
        std_error = data['deck_switch'].rolling(window=window_size, min_periods=1).std()
        
        # Calculate confidence interval (95%)
        ci = 1.96 * std_error / np.sqrt(window_size)  # 95% CI

        # Plotting line and shaded confidence interval
        line, = ax.plot(data['trial'], smooth_data, label=labels[i], color=colors[i])
        ax.fill_between(data['trial'], (smooth_data - ci), (smooth_data + ci), color=colors[i], alpha=alpha)

        lines.append(line)

    # expand y axis
    ax.set_ylim(0, 1)

    # Add labels and custom legend
    ax.set_xlabel("Trial")
    ax.set_ylabel("Proportion of Deck Switches at Trial")
    ax.legend(handles=lines, labels=labels)

    # save the plot
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def main():
    # define paths
    path = pathlib.Path(__file__)
    data_path = path.parents[2] / "data" / "final_data"

    # load both final datasets
    hc_data = pd.read_csv(data_path / "clean_gpt.csv")
    gpt_data = pd.read_csv(data_path / "clean_ahn_hc.csv")

    # calculate deck switches for both datasets
    hc_deck_switch_prop = calculate_deck_switches(hc_data)
    gpt_deck_switch_prop = calculate_deck_switches(gpt_data)

    # plot deck switches for both datasets
    plot_deck_switch_with_confidence(hc_deck_switch_prop, gpt_deck_switch_prop, ["Humans", "ChatGPT"], save_path=path.parents[2] / "src" / "descriptives" / "plots" / "deck_switches.png")


    print(hc_deck_switch_prop)



if __name__ == "__main__":
    main()
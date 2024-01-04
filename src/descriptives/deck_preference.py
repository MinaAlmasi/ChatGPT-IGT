import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.stats import binom
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontManager

def set_font():
    available_fonts = set(f.name for f in FontManager().ttflist)
    if 'Times New Roman' in available_fonts:
        # use times new roman if available
        plt.rcParams['font.family'] = 'Times New Roman'
        # set font size to 13
        plt.rcParams.update({'font.size': 12})

def calculate_deck_preferences(data):
    # create a new column that indicates the trial number
    data['trial'] = data.groupby('subjID').cumcount() + 1

    # replace 1,2,3,4 with A,B,C,D for 'x' column
    data['x'] = data['x'].replace([1,2,3,4], ['A','B','C','D'])

    grouped = data.groupby(['trial', 'x']).size().unstack(fill_value=0)
    # Calculate the proportion of each 'x' value in each trial
    proportions = grouped.div(grouped.sum(axis=1), axis=0)

    # Reset the index to have 'trial' as a column
    proportions.reset_index(inplace=True)

    return proportions


def plot_deck_preferences(data1, data2, labels, window_size=10, save_path=None):
    # Create a plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # add space between subplots
    fig.subplots_adjust(hspace=0.3)

    # Define colors for all four decks
    colors = ["#F59B8C", "#B43622", "#A1E780", "#3CA50A"]

    # loop over both subplots and datasets
    for i, (ax, data) in enumerate(zip([ax1, ax2], [data1, data2])):
        # Apply smoothing
        smooth_data = data.iloc[:, 1:].rolling(window=window_size, min_periods=1).mean()
        std_error = data.iloc[:, 1:].rolling(window=window_size, min_periods=1).std()
        
        # Calculate confidence interval (95%)
        ci = 1.96 * std_error / np.sqrt(window_size)

        alpha = 0.1

        # Plotting line and shaded confidence interval on the correct axis
        for j, deck in enumerate(smooth_data.columns):
            ax.plot(data['trial'], smooth_data[deck], label=deck, color=colors[j])
            ax.fill_between(data['trial'], (smooth_data[deck] - ci[deck]), (smooth_data[deck] + ci[deck]), color=colors[j], alpha=alpha)
        
        # set x-axis ticks to every 10th trial
        ax.set_xticks(np.arange(0, 101, 10))

        # set y-axis ticks to every 0.1
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        # add very light grid
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        # set title
        ax.set_title(labels[i])

    # add custom legend
    custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
                    Line2D([0], [0], color=colors[1], lw=2),
                    Line2D([0], [0], color=colors[2], lw=2),
                    Line2D([0], [0], color=colors[3], lw=2)]

    # add legend
    ax1.legend(custom_lines, ['Deck A', 'Deck B', 'Deck C', 'Deck D'], loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=4)

    # add y-axis label
    ax1.set_ylabel('Choice Proportions')
    ax2.set_ylabel('Choice Proportions')

    # add x-axis label
    ax2.set_xlabel('Trial number')
    ax2.set_xlabel('Trial number')

    # expand y axis
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0, 0.5)

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

    # calculate deck preferences
    hc_deck_preferences = calculate_deck_preferences(hc_data)
    gpt_deck_preferences = calculate_deck_preferences(gpt_data)

    # set font
    set_font()

    # plot deck preferences
    plot_deck_preferences(hc_deck_preferences, gpt_deck_preferences, labels=['Humans', 'ChatGPT'], save_path=path.parents[2] / "src" / "descriptives" / "plots" / "deck_preferences.png")





if __name__ == "__main__":
    main()
'''
Create table for summary BUGS output
'''
import pathlib
import pandas as pd


def main():
    # define paths
    path = pathlib.Path(__file__)

    # load ahn data
    ahn_path = path.parents[2] / "src" / "estimation" / "results" / "summary_param_estimated_ahn_hc.txt"
    ahn_df = pd.read_csv(ahn_path, sep=" ")

    # load gpt data 
    gpt_path = path.parents[2] / "src" / "estimation" / "results" / "summary_param_estimated_gpt.txt"
    gpt_df = pd.read_csv(gpt_path, sep=" ")

    # round and convert to string
    ahn_df = ahn_df.round(3).astype(str)
    gpt_df = gpt_df.round(3).astype(str)

    # make into latex
    ahn_latex = ahn_df.to_latex(index=True)
    print(ahn_latex)

    gpt_latex = gpt_df.to_latex(index=True)
    print(gpt_latex)
    
    # nb print latex saves a \midrule which is not working in Overleaf. Replace manually with \hline

if __name__ == "__main__":
    main()
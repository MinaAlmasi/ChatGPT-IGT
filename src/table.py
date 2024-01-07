'''
Create table for summary BUGS output
'''
import pathlib
import pandas as pd

def make_table(path, round_int=2, sep=" "): 
    '''
    Make latex table from BUGS output

    Args
        path: path to BUGS output (txt)
        round_int: number of decimals to round to

    Returns
        latex_table: latex table
    '''
    # load data
    df = pd.read_csv(path, sep=sep)

    # round and convert to string
    df = df.round(round_int).astype(str)

    # make into latex
    latex_table = df.to_latex(index=True)

    return latex_table

def main():
    # define paths
    path = pathlib.Path(__file__)

    # individual group level ORL parameters 
    ahn_path = path.parents[1] / "src" / "estimation" / "results" / "summary_param_estimated_ahn_hc.txt"
    gpt_path = path.parents[1] / "src" / "estimation" / "results" / "summary_param_estimated_gpt.txt"

    # group differences in ORL parameters 
    compORL_path = path.parents[1] / "src" / "comparison" / "results" / "alpha_params_comparison_summary.txt"
    compOUTCOME_path = path.parents[1] / "src" / "comparison" / "results" / "outcome_comparison_summary.txt"

    # payoff structure
    payoff_path = path.parents[1] / "utils" / "payoff_scheme_3.csv"

    # make latex tables
    ahn_table = make_table(ahn_path)
    gpt_table = make_table(gpt_path)
    
    comp_table = make_table(compORL_path)
    compOUTCOME_table = make_table(compOUTCOME_path)

    payoff_table = make_table(payoff_path, round_int=0, sep=",")

    print(payoff_table)

    # nb print latex saves a \midrule which is not working in Overleaf. Replace manually with \hline
    

if __name__ == "__main__":
    main()
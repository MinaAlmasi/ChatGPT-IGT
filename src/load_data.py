'''
Check Steingrover data
'''
import pathlib
import pandas as pd

def load_steingroever(data_path, study:str=None):
    # load index 
    index = pd.read_csv(data_path / "index_100.csv").reset_index(drop=True)

    # load choice 
    choice = pd.read_csv(data_path / "choice_100.csv").reset_index(drop=True)

    # load win (wi_100)
    win = pd.read_csv(data_path / "wi_100.csv").reset_index(drop=True)

    # load loss (lo_100)
    loss = pd.read_csv(data_path / "lo_100.csv").reset_index(drop=True)

    # bind files together as they are index in the same order
    data = pd.concat([index, choice], axis=1)

    # filter data to only include Study = "Horstmann"
    if study:
        data = data[data['Study'] == study]

    return data

def load_ahn(data_path, filename:str="healthy_control"):
    '''
    Load Ahn data
    '''
    # check if filename is valid
    valid_filenames = ["healthy_control", "amphetamine", "heroin"]

    if filename not in valid_filenames:
        raise ValueError(f"filename must be one of {valid_filenames}")

    # load data
    df = pd.read_csv(data_path / f"IGTdata_{filename}.txt", sep="\t")

    return df 


def main(): 
    # define path
    path = pathlib.Path(__file__).parents[1]

    # path to data 
    data_root = path / "data" 

    # load data
    #df_stein = load_steingroever(data_root /  "IGTdataSteingroever2014")

    # load ahn data
    df_ahn = load_ahn(data_path = data_root / "Ahn2014", filename="healthy_control")

    print(df_ahn)


if __name__ == "__main__":
    main()
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

def load_gpt(datapath):
    '''
    Load GPT data
    '''
    # get all files with iterdir 
    files = [file for file in datapath.iterdir() if file.is_file()]

    # load all files, add subject id and then concat in for loop
    dfs = []

    for i, file in enumerate(files):
        # load data
        data = pd.read_csv(datapath / file)

        # add subject id
        data['subject_id'] = i+1

        # append to list
        dfs.append(data)

    # concat all dfs
    df = pd.concat(dfs, axis=0)

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

    #print(df_ahn)

    # load gpt data
    df_gpt = load_gpt(data_root / "gpt_data")

    print(df_gpt)


if __name__ == "__main__":
    main()
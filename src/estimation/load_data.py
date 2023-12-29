'''
Check Steingrover data
'''
import pathlib
import pandas as pd

def prepare_ahn(data_path, filename:str="healthy_control"):
    '''
    Load and Prepare Ahn data 
    '''
    # check if filename is valid
    valid_filenames = ["healthy_control", "amphetamine", "heroin"]

    if filename not in valid_filenames:
        raise ValueError(f"filename must be one of {valid_filenames}")

    # load data
    df = pd.read_csv(data_path / f"IGTdata_{filename}.txt", sep="\t")

    # create outcoome column (note the sign: plus because losses are stored as negative)
    df['X'] = df['gain'] + df['loss']

    # drop gain and loss
    df = df.drop(columns=['gain', 'loss'])

    return df

def load_gpt(datapath):
    '''
    Load GPT data
    '''
    # get all files with iterdir 
    files = [file for file in datapath.iterdir() if file.is_file() and file.suffix == ".csv"]

    # load all files, add subject id and then concat in for loop
    dfs = []

    for i, file in enumerate(files):
        # load data
        data = pd.read_csv(file)

        # add subject id from filename
        data['subjID'] = file.stem

        # append to list
        dfs.append(data)

    # concat all dfs
    df = pd.concat(dfs, axis=0)

    return df

def clean_gpt(gpt_df):
    '''
    Cleans GPT data by translating decks, adding outcomes, dropping unnecessary columns and renaming columns
    '''
    ## translate the decks
    # translate A and E to 1
    gpt_df['deck'] = gpt_df['deck'].replace({"A":1, "E":1})

    # translate B and F to 2
    gpt_df['deck'] = gpt_df['deck'].replace({"B":2, "F":2})

    # translate C and G to 3
    gpt_df['deck'] = gpt_df['deck'].replace({"C":3, "G":3})

    # translate D and H to 4
    gpt_df['deck'] = gpt_df['deck'].replace({"D":4, "H":4})

    # add outcomes
    gpt_df['X'] = gpt_df['gain'] - gpt_df['loss']

    # drop unnecessary columns
    gpt_df = gpt_df.drop(columns=['gain', 'loss', 'trial', 'probabilities'])

    # rename columns
    gpt_df = gpt_df.rename(columns={'deck':'x'})

    return gpt_df
    

def main(): 
    # define path
    path = pathlib.Path(__file__).parents[2]

    # path to data 
    data_root = path / "data" 

    # save path 
    save_path = path / "data"

    # load ahn data
    df_ahn = load_ahn(data_path = data_root / "Ahn2014", filename="healthy_control")

    print(df_ahn)

    # load gpt data
    df_gpt = load_gpt(data_root / "GPTdata")

    # translate gpt data
    df_gpt = clean_gpt(df_gpt)

    print(df_gpt)


if __name__ == "__main__":
    main()
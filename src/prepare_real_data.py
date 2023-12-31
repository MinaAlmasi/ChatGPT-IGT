'''
Prepare AHN data
'''
import pathlib
import pandas as pd
import numpy as np

def prepare_ahn(data_path, filename:str="healthy_control", save_path=None):
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

    # group by subject id to see how many observatiions each subject has
    count = df.groupby('subjID').count()

    # get the last 30 subject ids in count
    last_30 = count.index[-30:]

    # filter df to only include last 30 subjects
    df = df[df['subjID'].isin(last_30)].reset_index(drop=True)

    # drop gain and loss
    df = df.drop(columns=['gain', 'loss', 'trial'])

    # rename columns
    df = df.rename(columns={'deck':'x'})

    # save data
    if save_path is not None:
        df.to_csv(save_path / f"clean_ahn_hc.csv", index=False)

    return df

def create_ahn_extra_samples(data_path, n_samples=5, filename:str="healthy_control", save_path=None):
    '''
    Create n_samples extra samples for Ahn data with 30 subjects
    '''
    # check if filename is valid
    valid_filenames = ["healthy_control", "amphetamine", "heroin"]

    if filename not in valid_filenames:
        raise ValueError(f"filename must be one of {valid_filenames}")

    # load data
    df = pd.read_csv(data_path / f"IGTdata_{filename}.txt", sep="\t")

    # create outcoome column (note the sign: plus because losses are stored as negative)
    df['X'] = df['gain'] + df['loss']

    # group by subject id to see how many observatiions each subject has
    count = df.groupby('subjID').count()

    # rm all subjects that does not have 100 observations
    count = count[count['trial'] == 100]

    # filter with count
    df = df[df['subjID'].isin(count.index)].reset_index(drop=True)

    # randomly sample 30 subjects
    subjects = df['subjID'].unique()

    # create empty list to store dfs
    dfs = []

    # loop over subjects
    for i in range(n_samples):
        # sample 30 subjects
        sample = np.random.choice(subjects, 30, replace=False)

        # filter df with sample
        df_sample = df[df['subjID'].isin(sample)].reset_index(drop=True)

        # drop gain and loss
        df_sample = df_sample.drop(columns=['gain', 'loss', 'trial'])

        # rename columns
        df_sample = df_sample.rename(columns={'deck':'x'})

        # append to list
        dfs.append(df_sample)
    
    # define path
    final_path = save_path / "extra_samples"
    final_path.mkdir(parents=True, exist_ok=True)

    # save each dataframe 
    for i, df in enumerate(dfs):
        df.to_csv(final_path / f"clean_ahn_hc_{i+1}.csv", index=False)

    return dfs

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

def clean_gpt(gpt_df, save_path=None):
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

    # save data
    gpt_df.to_csv(save_path / f"clean_gpt.csv", index=False)

    return gpt_df
    

def main(): 
    # set seed
    np.random.seed(2502)

    # define path
    path = pathlib.Path(__file__).parents[1]

    # path to data 
    data_root = path / "data" / "raw_data"

    # save path 
    save_path = path / "data" / "final_data"

    # load ahn data
    df_ahn = prepare_ahn(data_path = data_root / "Ahn2014", filename="healthy_control", save_path=save_path)

    # load gpt data
    df_gpt = load_gpt(data_root / "GPTdata")

    # translate gpt data
    df_gpt = clean_gpt(df_gpt, save_path=save_path)

    # create extra samples for ahn data
    create_ahn_extra_samples(data_path = data_root / "Ahn2014", n_samples=4, filename="healthy_control", save_path=save_path)

if __name__ == "__main__":
    main()
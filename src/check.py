'''
Check Steingrover data
'''
import pathlib
import pandas as pd


def main(): 
    # define path
    path = pathlib.Path(__file__).parents[1]

    # path to data 
    data_path = path / "data" /  "IGTdataSteingroever2014"

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
    data = data[data['Study'] == "Wood"]

    count_of_ones = data.apply(lambda row: (row == 4).sum(), axis=1)

    # sort from highest to lowest
    count_of_ones = count_of_ones.sort_values(ascending=False)

    print(count_of_ones.head(20))

    #print(data)


if __name__ == "__main__":
    main()
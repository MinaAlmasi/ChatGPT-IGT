from openai import OpenAI
import pathlib
import pandas as pd
import numpy as np
import datetime

def create_deck_dict(payoff_df, deck_name): 
    '''
    Create a deck dictionary for each deck 
    '''
    deck_dict = {}

    for i, row in payoff_df.iterrows(): 
        deck_dict[row["card_number"]] = {f"win": row[f"win_{deck_name}"], f"loss": row[f"loss_{deck_name}"]}

    return deck_dict

def create_all_decks():
    '''
    Create all decks from the payoff scheme as seperate dicts
    '''
    # define paths
    path = pathlib.Path(__file__)
    filepath = path.parents[1] / "utils" / "payoff_scheme_3.csv"

    # load df
    df = pd.read_csv(filepath)

    # create decks
    decks = {}

    for deck_name in ["A", "B", "C", "D"]:
        decks[deck_name] = create_deck_dict(df, deck_name)

    return decks


def initialize(task_desc_path):
    # load the task description from txt
    with open(task_desc_path / 'modified_task_desc.txt', 'r') as f:
        task_desc = f.read() 

    # load the api key
    with open(pathlib.Path(__file__).parents[1] / "keys" / 'api_key.txt', 'r') as f:
        key = f.read()

    # load organization id
    with open(pathlib.Path(__file__).parents[1] / "keys" / 'org_id.txt', 'r') as f:
        org_id = f.read()

    # initilize the OpenAI class
    client = OpenAI(
        organization=org_id,
        api_key=key
    )

    return task_desc, client


def select_deck(client, task_desc, model_endpoint = 'gpt-3.5-turbo-1106', updated_messages=None):
    # update messages (with previous conversation)
    if updated_messages is None:
        messages = [
            {"role": "system", "content": task_desc},
            {"role": "user", "content": task_desc + "\n" + "A, B, C or D?"}
        ]
    else: 
        messages = updated_messages

    # create completion
    completion = client.chat.completions.create(
        model=model_endpoint,
        frequency_penalty=-2,
        presence_penalty=-2,
        temperature=0.8,
        messages=messages,
        max_tokens=15
        )

    # get how many tokens the completion used
    #print(completion.usage)

    # get text response
    card_selection = completion.choices[0].message.content

    # print response
    print(card_selection)

    return card_selection, messages


def get_payoff(card_selection, decks, selected):
    # check card selection
    if card_selection not in ['A', 'B', 'C', 'D']:
        raise ValueError("An invalid deck was chosen!")

    # select deck from decks dict
    deck = decks[card_selection]

    # update the selected dict
    selected[card_selection] += 1

    # define new value as cardnumber
    card_number = selected[card_selection]

    # get win, loss from deck for the selected card
    win = deck[card_number]["win"]       
    loss = deck[card_number]["loss"]

    return selected, win, loss

def check_for_empty_deck(selected):
    '''
    Checks if any of the four decks have been selected 60 times
    '''
    # check if any of them are 60, if so return that deck
    for deck in selected:
        if selected[deck] >= 60:
            return deck
    
    # if none of them are 60, return None
    return None

def update_messages(messages, card_selection, win, loss, total_earnings, empty_deck):
    # update messages with card selection
    messages.append({"role": "assistant", "content": card_selection})

    # create win message
    payoff_message = f" You WON ${win}!"
    
    # create loss message
    if loss > 0:
        loss_message = f"But you LOSE ${loss}!"
        # add to payoff message
        payoff_message += "\n" + loss_message

    # add total earnings
    payoff_message += f"\nYour TOTAL EARNINGS are ${total_earnings}."

    # add amount borrowed
    payoff_message += f"\nYour BORROWED MONEY is $2000."

    # append win/loss message
    messages.append({"role": "user", "content": payoff_message})

    # initialize next round message based on empty deck also
    if empty_deck is None:
        messages.append({"role": "user", "content": "A, B, C or D?"})
    
    elif empty_deck == "A":
        messages.append({"role": "user", "content": "Deck A is empty. B, C or D?"})
    
    elif empty_deck == "B":
        messages.append({"role": "user", "content": "Deck B is empty. A, C or D?"})
    
    elif empty_deck == "C":
        messages.append({"role": "user", "content": "Deck C is empty. A, B or D?"})
    
    elif empty_deck == "D":
        messages.append({"role": "user", "content": "Deck D is empty. A, B or C?"})

    return messages

def save_data(data:dict, updated_messages, data_path):
    # ensure data_path exists
    data_path.mkdir(parents=True, exist_ok=True)

    # make a log path 
    log_path = data_path / "message_logs"
    log_path.mkdir(parents=True, exist_ok=True)

    # make data into a df
    df = pd.DataFrame.from_dict(data, orient="index")

    # set first column as 'trial'
    df.index.name = "trial"
    df.reset_index(inplace=True)
    
    # create filename with a date and time (by getting time, formatting it and then adding it to the filename)
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    filename = f"gpt_{formatted_time}"

    # save df into data 
    df.to_csv(data_path/ f"{filename}.csv", index=False)

    # save messages (logfile-ish)
    with open(log_path / f"{filename}_messages.txt", 'w') as f:
        f.write(str(updated_messages))
    
def main():
    # get paths
    path = pathlib.Path(__file__).parents[1]
    task_desc_path = path / "utils" / "task_descriptions"
    
    ## SETUP ##
    # create all decks (A, B, C, D)
    decks = create_all_decks()

    task_desc, client = initialize(task_desc_path)

    # define things for playing
    trials_to_play = 100
    trials_played = 0
    selected = {"A":0, "B":0, "C":0, "D":0}
    total_earnings = 2000
    
    data = {}

    # start with no message history 
    updated_messages = None
    empty_deck = None

    while trials_played < trials_to_play:
        ## TEMP CHUNK START ##
        # get chatgpt to select a card 
        messages = [
            {"role": "system", "content": task_desc},
            {"role": "user", "content": task_desc + "\n" + "A, B, C or D?"}
        ]

        # get card selection with numpy, weight A more 
        if empty_deck == None:
            card_selection = np.random.choice(['A', 'B', 'C', 'D'], p=[0.8, 0.05, 0.05, 0.1])
        else:
            card_selection = "B"
        ## TEMP CHUNK END ##
        #card_selection, messages = select_deck(client, task_desc, updated_messages=updated_messages)

        # get payoff
        selected, win, loss = get_payoff(card_selection, decks, selected)

        # check for empty decks
        empty_deck = check_for_empty_deck(selected)

        # update total earnings based on wins and losses
        total_earnings = total_earnings + win - loss

        # add a trial played
        trials_played += 1

        # update data dict 
        data[trials_played] = {"deck": card_selection, "gain": win, "loss": loss}

        # update messages
        updated_messages = update_messages(messages, card_selection, win, loss, total_earnings, empty_deck)

        # save data
        data_path = path / "data" / "gpt_data"
        save_data(data, updated_messages, data_path=data_path)

        print(trials_played, card_selection, win, loss)

if __name__ == '__main__':
    main()

from openai import OpenAI
import pathlib
import pandas as pd
import numpy as np
import datetime
import time
import argparse

def input_parse():
    parser = argparse.ArgumentParser(description='Run the GPT simulation')

    parser.add_argument("-other", "--other_letters", help='If specified, use other deck names (W, Y, X, Z)', action="store_true")

    return parser.parse_args()

def create_deck_dict(payoff_df, deck_name): 
    '''
    Create a deck dictionary for each deck 
    '''
    deck_dict = {}

    for i, row in payoff_df.iterrows(): 
        deck_dict[row["card_number"]] = {f"win": row[f"win_{deck_name}"], f"loss": row[f"loss_{deck_name}"]}

    return deck_dict

def create_all_decks(other=False):
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
    
    if other: 
        # replace keys ["A", "B", "C", "D"] with ["W", "Y", "X", "Z"]
        decks["W"] = decks.pop("A")
        decks["Y"] = decks.pop("B")
        decks["X"] = decks.pop("C")
        decks["Z"] = decks.pop("D")

    return decks

def create_task_description(task_desc_path, other=False):
    # load the task description from txt
    with open(task_desc_path / 'modified_task_desc.txt', 'r') as f:
        task_desc = f.read() 
    
    # if other, replace A, B, C, D with W, Y, X, Z
    if other:
        task_desc = task_desc.replace("A, B, C and D", "W, Y, X and Z")
        task_desc = task_desc.replace("(A, B, C, D)", "(W, Y, X, Z)")

    return task_desc

def initialize_client():
    '''
    Intialize the OpenAI class
    '''
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

    return client

def select_deck(client, task_desc, model_endpoint = 'gpt-3.5-turbo-1106', updated_messages=None, other=False):
    '''
    Select a deck from the four decks with ChatGPT
    '''
    if other:
        letters = ["W", "Y", "X", "Z"]
        logit_bias = {54: 100, 55: 100, 56: 100, 57: 100}
    else:
        letters = ["A", "B", "C", "D"]
        logit_bias = {32: 100, 33: 100, 34: 100, 35: 100}
    
    # define the four decks
    d1 = letters[0]
    d2 = letters[1]
    d3 = letters[2]
    d4 = letters[3]

    # update messages (with previous conversation)
    if updated_messages is None:
        messages = [
            {"role": "system", "content": task_desc},
            {"role": "user", "content": task_desc + "\n" + f"'{d1}, {d2}, {d3} or {d4}?'"}
        ]
    else: 
        messages = updated_messages
    
    # set valid_deck, attempt count and max attempts to not make chatgpt stuck in a loop forever
    valid_deck = False
    attempt_count = 0
    max_attempts = 3 

    while not valid_deck and attempt_count < max_attempts:
        # create completion
        completion = client.chat.completions.create(
            model=model_endpoint,
            frequency_penalty=-2, # disables all frequency penalty
            presence_penalty=-2, # disables all presence penalty
            temperature=1.4,
            messages=messages,
            max_tokens=1,
            logit_bias=logit_bias # force ChatGPT to only use letters
            )

        # get how many tokens the completion used
        #print(completion.usage)

        # get text response
        card_selection = completion.choices[0].message.content

        # check if card selection is valid
        if card_selection in letters:
            valid_deck = True
        else:
            # add message to messages
            messages.append({"role": "assistant", "content": card_selection})
            messages.append({"role": "user", "content": "Please only specify the letter of the deck you want."})
            attempt_count += 1

    if not valid_deck:
        print(messages)
        raise ValueError("Maximum attemtps reached, ChatGPT failed to select a valid deck.")

    return card_selection, messages


def get_payoff(card_selection, decks, selected):
    '''
    Get the payoff for the selected card
    '''
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

def update_messages(messages, card_selection, win, loss, total_earnings, empty_deck, other=False):
    '''
    Update ChatGPT messages with card selection and payoff. Add empty deck message if needed.
    '''
    if other:
        letters = ["W", "Y", "X", "Z"]
    else:
        letters = ["A", "B", "C", "D"]
    
    # define the four decks
    d1 = letters[0]
    d2 = letters[1]
    d3 = letters[2]
    d4 = letters[3]

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
    payoff_message += f"\nTOTAL EARNINGS: ${total_earnings}."

    # add amount borrowed
    payoff_message += f"\nBORROWED MONEY: $2000."

    # append win/loss message
    messages.append({"role": "user", "content": payoff_message})

    # if empty deck is not none, add message to ask for a new deck
    if empty_deck is None:
        messages.append({"role": "user", "content": f"'{d1}, {d2}, {d3} or {d4}?'"})

    # if a deck is empty, identify which and ask for a new deck based on the remaining decks
    else:
        # create list of decks that are not empty
        not_empty = [d for d in letters if d != empty_deck]

        # create message
        new_message = f"Deck {empty_deck} is empty. {not_empty[0]}, {not_empty[1]} or {not_empty[2]}?"

        # append message
        messages.append({"role": "user", "content": new_message})

    return messages

def save_data(data:dict, updated_messages, data_path, other=False):
    '''
    Save data as a csv and save messages as a txt
    '''
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

    if other:
        filename = f"gpt_{formatted_time}_WXYZ"
    else:
        filename = f"gpt_{formatted_time}_ABCD"
    
    # save df into data 
    df.to_csv(data_path/ f"{filename}.csv", index=False)

    # save messages (logfile-ish)
    with open(log_path / f"{filename}.txt", 'w') as f:
        f.write(str(updated_messages))

        
def main():
    args = input_parse()

    # get paths
    path = pathlib.Path(__file__).parents[1]
    task_desc_path = path / "utils" / "task_descriptions"
    
    ## SETUP ##
    # define letters
    if args.other_letters:
        letters = ["W", "Y", "X", "Z"]
    else:
        letters = ["A", "B", "C", "D"]

    # create all decks (A, B, C, D) or (W, Y, X, Z)
    decks = create_all_decks(other=args.other_letters)
    d1, d2, d3, d4 = letters[0], letters[1], letters[2], letters[3] # extract for later use

    # init OpenAI client
    client = initialize_client()

    # load task desc
    task_desc = create_task_description(task_desc_path, other=args.other_letters)

    # define things for playing
    trials_to_play = 10
    trials_played = 0
    selected = {d1:0, d2:0, d3:0, d4:0}
    total_earnings = 2000
    
    data = {}

    # start with no message history 
    updated_messages = None
    empty_deck = None

    while trials_played < trials_to_play:
        # add delay to avoid rate limit
        time.sleep(10)

        # select deck
        card_selection, messages = select_deck(client, task_desc, updated_messages=updated_messages, other=args.other_letters)

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
        updated_messages = update_messages(messages, card_selection, win, loss, total_earnings, empty_deck, other=args.other_letters)

        # get the last four dictionaries in the messages list
        print(trials_played, card_selection, win, loss)
        print(updated_messages[-4:-1])

    # save data
    data_path = path / "data" / "gpt_data"
    save_data(data, updated_messages, data_path=data_path, other=args.other_letters)

if __name__ == '__main__':
    main()
from openai import OpenAI
import pathlib

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


def select_deck(client, task_desc, model_endpoint = 'gpt-3.5-turbo-1106', new_messages=None):
    # create completion
    completion = client.chat.completions.create(
        model=model_endpoint,
        frequency_penalty=-2,
        presence_penalty=-2,
        temperature=0.8,
        messages=[
            {"role": "system", "content": task_desc},
            {"role": "user", "content": task_desc + "\n" + "A, B, C or D?"}
        ]
        )

    # get how many tokens the completion used
    print(completion.usage)

    # get response
    deck_selection = completion.choices[0].message

    # print response
    print(deck_selection)

    return deck_selection


def main():
    # get paths
    path = pathlib.Path(__file__).parents[1]
    task_desc_path = path / "utils" / "task_descriptions"

    task_desc, client = initialize(task_desc_path)

    select_deck(client, task_desc)

if __name__ == '__main__':
    main()

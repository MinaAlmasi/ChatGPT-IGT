<h1 style="font-size: xx-large; font-weight: bold; text-align: center">Investigating ChatGPT’s Decision-Making with The Iowa Gambling Task</h1>


<p align="center">
  Anton Drasbæk Schiønning (<strong><a href="https://github.com/drasbaek">@drasbaek</a></strong>) and
  Mina Almasi (<strong><a href="https://github.com/MinaAlmasi">@MinaAlmasi</a></strong>)<br>
  Aarhus University, Decision-Making Exam (E23)
</p>
<hr>

## About
This repository contains all data and code for used for the exam project *Investigating ChatGPT’s Decision-Making with The Iowa Gambling Task* by Mina Almasi and Anton Drasbæk Schiønning.

## Project Overview
### General Overview
The repository is structured as such:
| Folder              | Description |
|---------------------------|-------------|
| `data/`                   | Contains both the used sample of healthy controls and the ChatGPT data created in this investigation. |
| `keys/`                  | Placeholder folder for storing OpenAI API keys if wishing to use the pipeline for playing IGT with ChatGPT. |
| `models/`                  | Contains all JAGS models used in the investigation. |
| `src/`                    | Contains all Python and R code related to the project. For a detailed overview, refer to the table below (`src` Overview). |
| `utils/`               | Contains IGT utility files such as the modified payoff structure and task descriptions. |

Separate READMEs with further detail are contained within folders where relevant.


### `src` Overview
Each subfolder within `src` contains both code, plots and results (if relevant) for that subsection.

| Folder/File               | Description |
|---------------------------|-------------|
| `comparison/`                   | Code for doing outcome and ORL parameter group comparisions.|
| `descriptives/`                  | Code for plotting the behavioral checks, deck switches and preferences.|
| `estimation/`                  | Code for doing individual group-level estimates and posterior predictive checks. |
| `recovery/`                | Code for doing both subject-level and hierarhical parameter recovery. Also contains code for simulating the data for the recoveries.|
| `gpt_simulate.py`                    | Python script for playing the Iowa Gambling Task with ChatGPT. Note API key + ORG ID should be placed in `keys/`.|
| `prepare_real_data.py`               | Python script for reformatting the human- and GPT data.|
| `table.py`               | Python script for converting tables into LaTeX formats.|


## Technical Requirements
The data creation and analysis was run in part via Ubuntu v22.04.3, Python v3.10.12 (UCloud, Coder Python 1.84.2) and also locally using a Macbook Pro ‘13 (2020, 2 GHz Intel i5, 16GB of ram). 

Python's `venv` needs to be installed for the Python code to run as intended. Note that any code requiring `JAGS` is only tested on UCloud (see its setup below).

## Setup
Prior to running any code, please run the command below to create a virtual environment (`env`) and install necessary packages within it:
```
bash setup.sh
```

Importantly, running the R scripts with JAGS models requires additional setups (only tested on UCloud):
```
bash setup_rjags.sh
```

## Contact
For any questions regarding the project or its reproducibility, please feel free to contact us: 
<ul style="list-style-type: none;">
  <li><a href="mailto:drasbaek@post.au.dk">drasbaek@post.au.dk</a>
(Anton)</li>
    <li><a href="mailto: mina.almasi@post.au.dk"> mina.almasi@post.au.dk</a>
(Mina)</li>
</ul>



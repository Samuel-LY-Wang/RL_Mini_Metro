# Python Mini metro RL
This repo builds on the pygame implementation of Mini Metro, a fun 2D strategic game where you try to optimize the max number of passengers your metro system can handle. For the purpose of RL, a version with no rendering was created, allowing for much faster running.

# Installation
`pip install -r requirements.txt`

# How to run
# Testing
`python -m unittest -v`

# Run RL
If you want to train your own RL agent, run `python -m RL.RL_main --steps [number of steps desired] --updates [number of epochs desired]`. If you wish to plot the progress of the training, use the "--plot" flag. If you want more information on each update, use the "--verbose" flag. \
If you merely wish to run the existing agent, load from any of the checkpoints provided in checkpoints/ into the code in RL/Display_Agent.py
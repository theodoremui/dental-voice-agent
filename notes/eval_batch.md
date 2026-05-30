Ultrathink on a batch evaluation runner file named "batch_eval_runner.py" which generates evaluation results that captures the specific voice agent being run.  The different voice agent is passed in as a command line parameter for "batch_eval_runner.py" such as:

"uv run python batch_eval_runner.py --bots bot0.py bot1.py bot2.py"

Our goal is to plot changes in the accuracy, latency, and other metrics across different implementations of the voice agent.  

The running of evaluations should make use of parallelism as much as possible.  Think carefully how to invoke these bots using different ports to enable parallelism.

Each agent is invoked using "uv run bot.py" where "bot.py" is any specific implemmentation of the bot.  

Ultrathink on how to adjust the judge so that as long as the answers are what the users are looking for, it should pass those test scenarios.

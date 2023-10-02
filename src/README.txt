To set up the Python environment, follow the instructions in the 'setup.txt' file. Then, run 'jupyter notebook' to open any of the notebooks.

See Demo.ipynb for basic functionality. Feel free to explore the other notebooks to see how experiments were set up and run some extra ones yourself!

To train agents, run scripts/opt_multirotorenv.py from the src/ directory. Ex: 'python -m scripts.opt_multirotorenv test --nprocs 10 --ntrials 500 --cardinal True' 
    will run a hyperparameter optimziation with 3 parallel processes for 500 trials with cardinal wind directions during training. Make sure this script is set up properly for your experiments.

We expect that our seeding is set up properly to reproduce our results exactly, it has worked from our testing. However, some hyperparameter trials were stopped early due to computational
    limitations, so it is possible you will find agents with higher reward through more trials.

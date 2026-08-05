# ML Playground repository instructions

This is a repository for various, unrelated, small-scale ML experiments. While the experiments are unrelated, the code
structure is the same: prioritizing production speed and readability rather than long-term maintainability.

For the sake of simplicity, every single experiment has its own dedicated folder in the `experiments/` directory. Hence,
something like `experiments/foo/` will be about an experiment that does foo, whatever that may be. Inside the folder for
some experiment is all the necessary code for running it, idea being that `main.py` is the main file, all others are
auxiliary. It's preferred to have just one file per experiment, unless it really is better to have multiple.

Code needs to be readable first and foremost, even if it's a monolithic file, and since it's being run locally, any kind
or argument parsing is just needless clutter. Variables and hyperparameters need to be recorded at the top of the file
and then used later.

All data used for all experiments needs to be downloaded/saved into `data/`. Not `experiments/data/`, nor
`experiments/foo/data/`, no, it needs to be one shared directory for storing all data for all experiments, since a lot
of the times the data is reused.

Experiments rely on matplotlib for plots. It's vastly preferred to simply run `plt.plot()` rather than saving the plots.
It's expected that there's a loss/accuracy plot plotted for each epoch of training. On top of that, tqdm should show
progress and basic diagnostic print statements. Once training is complete, another final plot needs to be made, showing
some actual examples from the test/eval dataset so that it's actually clear if the experiment works or not, not just by
looking at the loss and whatnot.

Saving the models is not strictly necessary unless explicitly asked for.

<p align='left'>
  <a href="https://github.com/hydra/hydra"><img alt="Numba" src="https://img.shields.io/badge/Hydra-000?logo=meta&style=for-the-badge" /></a>
</p>

Source code of the paper
[Directed-E<sup>2</sup>](https://arxiv.org/abs/2406.13909) algorithm used in [Model-Based Exploration in Truthful Monitored Markov Decision
Processes](https://arxiv.org/abs/2502.16772).

## Install and Examples

To install and use our environments, run
```
pip install -r requirements.txt
cd src/gym-grid
pip install -e .
```

## Hydra Configs
We use [Hydra](https://hydra.cc/docs/intro/) to configure our experiments.  
Hyperparameters and other settings are defined in YAML files in the `configs/` folder.


## Sweeps
For a sweep over multiple jobs in parallel with Joblib, run
```
python main.py -m hydra/launcher=joblib hydra/sweeper=manual_sweeper
```
Custom sweeps are defined in `configs/hydra/sweeper/`.  
You can further customize a sweep via command line. For example,
```
python main.py -m hydra/launcher=joblib hydra/sweeper=manual_sweeper experiment.rng_seed="range(0, 10)" hydra.launcher.verbose=1000
```
Configs in `configs/hydra/sweeper/` hide the training progress bar of the agent, so we
suggest to pass `hydra.launcher.verbose=1000` to show the progress of the sweep.
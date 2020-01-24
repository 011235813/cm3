# Cooperative Multi-Stage Multi-Goal Multi-Agent Reinforcement Learning (CM3)

This repository provides code for experiments in the paper [CM3](https://arxiv.org/abs/1809.05188)[1], published in ICLR 2020.
It contains the main algorithm and baselines, and the three simulation Markov games on which algorithms were evaluated.


## Dependencies

- All experiments were run on Ubuntu 16.04
- Python 3.6
- TensorFlow 1.10
- [SUMO](https://github.com/eclipse/sumo)
- pygame: `sudo apt-get install python-pygame`
- OpenAI Gym 0.12.1


## Project structure

- `alg`: Implementation of algorithms and config files. `config.json` is the main config file. `config_particle_*.json` specifies various instances of the cooperative navigation task. `config_sumo_stage{1,2}.json` specifies agent initial/goal lane configurations for SUMO. `config_checkers_stage{1,2}.json` specifies parameters of the Checkers game.
- `env`: Python wrappers/definitions of the simulation environments.
- `env_sumo`: XML files that define the road and traffic for the underlying SUMO simulator.
- `log`: Each experiment run will create a subfolder that contains the reward values logged during the training or test run.
- `saved`: Each experiment run will create a subfolder contains trained TensorFlow models.


## Environments

There are three simulations, selected by the `experiment` field in `alg/config.json`.

1. Cooperative navigation: particles must move to individual target locations while avoiding collisions.
   - Environment code located in `env/multiagent-particle-envs/`
2. SUMO
   - Stage 1: single agent on empty road. Corresponds to setting `"stage" : 1`
   - Stage 2: two agents on empty road. Corresponds to setting `"stage" : 2`
   - Python wrappers located in `env/`. Entry point is `env/multicar_simple.py`
   - SUMO topology and traffic defined in `env_sumo/simple/`
3. Checkers: two agents cooperate to collect rewards while avoiding penalties in a checkered map.
   - Implemented in `env/checkers.py`

## Environment setup

1. Cooperative navigation: run `pip install -e .` inside `env/multiagent-particle-envs/`
2. SUMO: Install [SUMO](https://github.com/eclipse/sumo) and add the following to your `.bashrc`
 - `export PYTHONPATH=$PYTHONPATH:path/to/sumo`
 - `export PYTHONPATH=$PYTHONPATH:path/to/sumo/tools`
 - `export SUMO_HOME="path/to/sumo"`
3. Checkers: None required


## Training

### Environment-specific examples

**Cooperative navigation**

- In `config.json`, set
  - `experiment`: "particle"
  - `particle_config` should be one of `config_particle_stage1.json`, `config_particle_stage2_antipodal.json`, `config_particle_stage2_cross.json`, `config_particle_stage2_merge.json`
- Inside `alg/`, execute
  `python train_onpolicy.py`

**SUMO**

- In `config.json`, set
  - `experiment`: "sumo"
  - `port`: if multiple SUMO experiments are run in parallel, each experiment must have its unique number
- Inside `alg/`, execute
  `python train_offpolicy.py --env ../env_sumo/simple/merge.sumocfg`
- Include the option `--gui` to show SUMO GUI while training (at the cost of increased runtime)

**Checkers**

- In `config.json`, set
  - `experiment` : "checkers"
- Inside `alg/`, execute
  `python train_offpolicy.py`

### General notes for running Stage 1 and 2 of CM3
- `stage`: either 1 or 2
- `dir_restore`: for Stage 2 of CM3, this must be equal to the string for `dir_name` when Stage 1 was run.
- `use_alg_credit`: 1 for CM3
- `use_Q_credit`: 1 for CM3. 0 for ablation that uses value function baseline.
- `train_from_nothing`: 1 for Stage 1 of CM3, or the ablation that omits the curriculum. 0 to allow restoring a trained Stage 1 model.
- `model_name`: when training Stage 2 and restoring a Stage 1 model, this must be the name of the model in Stage 1.
- `prob_random`: 1.0 for Stage 1, 0.2 for Stage 2. Not applicable for Checkers.


## Citation

<pre>
@article{yang2018cm3,
  title={Cm3: Cooperative multi-goal multi-stage multi-agent reinforcement learning},
  author={Yang, Jiachen and Nakhaei, Alireza and Isele, David and Fujimura, Kikuo and Zha, Hongyuan},
  journal={arXiv preprint arXiv:1809.05188},
  year={2018}
}
</pre>


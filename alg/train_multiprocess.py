"""Multi-seed runs."""
from multiprocessing import Process

import json
import train_offpolicy
import train_onpolicy

processes = []

with open('config.json', 'r') as f:
    config = json.load(f)

experiment = config['experiment']
use_alg_credit = config['use_alg_credit']
use_qmix = config['use_qmix']
n_seeds = config['n_seeds']
seed_base = config['seed']
dir_name_base = config['dir_name']
dir_idx_start = config['dir_idx_start']
port = config['port']

if use_alg_credit and experiment == 'particle':
    train_function = train_onpolicy.train_function
elif use_alg_credit:
    train_function = train_offpolicy.train_function
elif use_qmix:
    train_function = train_offpolicy.train_function
else:
    train_function = train_onpolicy.train_function

for idx_run in range(n_seeds):
    config_copy = config.copy()
    config_copy['seed'] = seed_base + idx_run
    config_copy['dir_name'] = (dir_name_base +
                               '_{:1d}'.format(dir_idx_start + idx_run))
    if experiment == 'sumo':
        config_copy['port'] = port + idx_run
    p = Process(target=train_function, args=(config_copy,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

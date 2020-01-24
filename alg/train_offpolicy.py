"""Entry point for off-policy training."""

import tensorflow as tf
import numpy as np

import sys, os
sys.path.append('../')
import json
import time
import random

import alg_credit_checkers
import alg_credit
import alg_qmix_checkers
import alg_qmix
import alg_baseline_checkers
import alg_baseline
import replay_buffer
import replay_buffer_dual
import evaluate

# Particle
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

# SUMO
from env import multicar_simple
from env import sumo_simulator

# Checkers
from env import checkers

def train_function(config):

    # ----------- Alg parameters ----------------- #
    experiment = config['experiment']
    if experiment == "particle":
        scenario_name = config['scenario']
    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed)
    
    # Curriculum stage
    stage = config['stage']
    port = config['port']
    dir_name = config['dir_name']
    dir_restore = config['dir_restore']
    use_alg_credit = config['use_alg_credit']
    use_qmix = config['use_qmix']
    use_Q_credit = config['use_Q_credit']
    # If 1, then uses Q-net and global reward
    use_Q = config['use_Q']
    use_V = config['use_V']
    if experiment == "sumo":
        dimensions = config['dimensions_sumo']
    elif experiment == "particle":
        dimensions = config['dimensions_particle']
    # If 1, then restores variables from same stage
    restore_same_stage = config['restore_same_stage']
    # If 1, then does not restore variables, even if stage > 1
    train_from_nothing = config['train_from_nothing']
    # Name of model to restore
    model_name = config['model_name']
    # Total number of training episodes
    N_train = config['N_train']
    period = config['period']
    # Number of evaluation episodes to run every <period>
    N_eval = config['N_eval']
    summarize = config['summarize']
    alpha = config['alpha']
    lr_Q = config['lr_Q']
    lr_V = config['lr_V']
    lr_actor = config['lr_actor']
    dual_buffer = config['dual_buffer']
    buffer_size = config['buffer_size']
    threshold = config['threshold']
    batch_size = config['batch_size']
    pretrain_episodes = config['pretrain_episodes']
    steps_per_train = config['steps_per_train']
    max_steps = config['max_steps']
    # Probability of using random configuration
    prob_random = config['prob_random']
    
    epsilon_start = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_div = config['epsilon_div']
    epsilon_step = (epsilon_start - epsilon_end)/float(epsilon_div)
    
    if experiment == "sumo":
        # ----------- SUMO parameters ---------------- #
        with open('config_sumo_stage%d.json' % stage) as f:
            config_sumo = json.load(f)
        n_agents = config_sumo["n_agents"]
        list_goals_fixed = config_sumo['goal_lane']
        list_routes_fixed = config_sumo['route']
        list_lanes_fixed = config_sumo['lane']
        list_goal_pos = config_sumo['goal_pos']
        list_speeds = config_sumo['speed']
        init_positions = config_sumo['init_position']
        list_id = config_sumo['id']
        list_vtypes = config_sumo['vtypes']
        depart_mean = config_sumo['depart_mean']
        depart_stdev = config_sumo['depart_stdev']
        total_length = config_sumo['total_length']
        total_width = config_sumo['total_width']
        save_threshold = config_sumo['save_threshold']
        map_route_idx = {'route_ramp':0, 'route_straight':1}
        
        sim = sumo_simulator.Simulator(port, list_id=list_id,
                                       other_lc_mode=0b1000000001,
                                       sublane_res=0.8, seed=seed)
        for i in range(int(2/sim.dt)):
            sim.step()
    elif experiment == 'particle':
        with open(config["particle_config"]) as f:
            config_particle = json.load(f)
        n_agents = config_particle['n_agents']
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world(n_agents, config_particle, prob_random)
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.done, max_steps=max_steps)
    elif experiment == 'checkers':
        with open("config_checkers_stage%d.json" % stage) as f:
            config_checkers = json.load(f)
        n_agents = config_checkers['n_agents']
        dimensions = config_checkers['dimensions']
        init = config_checkers['init']
        env = checkers.Checkers(init['n_rows'], init['n_columns'], init['n_obs'], init['agents_r'], init['agents_c'], n_agents, max_steps)
    
    l_action = dimensions['l_action']
    l_goal = dimensions['l_goal']
    
    # Create entire computational graph
    # Creation of new trainable variables for new curriculum
    # stage is handled by networks.py, given the stage number
    if use_alg_credit:
        if experiment == 'checkers':
            alg = alg_credit_checkers.Alg(experiment, dimensions, stage, n_agents, lr_V=lr_V, lr_Q=lr_Q, lr_actor=lr_actor, use_Q_credit=use_Q_credit, use_V=use_V, nn=config_checkers['nn'])
        else:
            alg = alg_credit.Alg(experiment, dimensions, stage, n_agents, lr_V=lr_V, lr_Q=lr_Q, lr_actor=lr_actor, use_Q_credit=use_Q_credit, use_V=use_V, nn=config['nn'])
    elif not use_qmix:
        if experiment == 'checkers':
            alg = alg_baseline_checkers.Alg(experiment, dimensions, stage, n_agents, lr_V=lr_V, lr_Q=lr_Q, lr_actor=lr_actor, use_Q=use_Q, use_V=use_V, alpha=alpha, nn=config_checkers['nn'], IAC=config['IAC'])
        else:
            alg = alg_baseline.Alg(experiment, dimensions, stage, n_agents, lr_V=lr_V, lr_Q=lr_Q, lr_actor=lr_actor, use_Q=use_Q, use_V=use_V, alpha=alpha, nn=config['nn'], IAC=config['IAC'])
    else:
        print("Using QMIX")
        if experiment == 'checkers':
            alg = alg_qmix_checkers.Alg(experiment, dimensions, stage, n_agents, lr_Q=lr_Q, nn=config_checkers['nn'])
        else:
            alg = alg_qmix.Alg(experiment, dimensions, stage, n_agents, lr_Q=lr_Q)
    
    print("Initialized computational graph")
    
    list_variables = tf.trainable_variables()
    if stage == 1 or restore_same_stage or train_from_nothing:
        saver = tf.train.Saver()
    elif stage == 2:
        # to_restore = [v for v in list_variables if ('stage-%d'%stage not in v.name.split('/') and 'Policy_target' not in v.name.split('/'))]
        to_restore = []
        for v in list_variables:
            list_split = v.name.split('/')
            if ('stage-%d'%stage not in list_split) and ('Policy_target' not in list_split) and ('Q_credit_main' not in list_split) and ('Q_credit_target' not in list_split):
                to_restore.append(v)
        saver = tf.train.Saver(to_restore)
    else:
        # restore only those variables that were not
        # just created at this curriculum stage
        to_restore = [v for v in list_variables if 'stage-%d'%stage not in v.name.split('/')]
        saver = tf.train.Saver(to_restore)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.set_random_seed(seed)
    sess = tf.Session(config=config)
    
    writer = tf.summary.FileWriter('../saved/%s' % dir_name, sess.graph)
    
    sess.run(tf.global_variables_initializer())
    print("Initialized variables")
    
    if train_from_nothing == 0:
        print("Restoring variables from %s" % dir_restore)
        saver.restore(sess, '../saved/%s/%s' % (dir_restore, model_name))
        if stage == 2 and use_alg_credit and use_Q_credit:
            # Copy weights of Q_global to Q_credit at the start of Stage 2
            sess.run(alg.list_initialize_credit_ops)
            for var in list_variables:
                if var.name == 'Q_global_main/Q_branch1/kernel:0':
                    print("Q_global")
                    print(sess.run(var))
                    print("")
                if var.name == 'Q_credit_main/Q_branch1/kernel:0':
                    print("Q_credit")
                    print(sess.run(var))
                    print("")
    
    # initialize target networks to equal main networks
    sess.run(alg.list_initialize_target_ops)
    
    # save everything without exclusion
    saver = tf.train.Saver(max_to_keep=None)
    
    epsilon = epsilon_start
    # For computing average over 100 episodes
    reward_local_century = np.zeros(n_agents)
    reward_global_century = 0
    
    # Write log headers
    header = "Episode,r_global"
    header_c = "Century,r_global_avg"
    for idx in range(n_agents):
        header += ',r_%d' % idx
        header_c += ',r_avg_%d' % idx
    header_c += ",r_global_eval"
    for idx in range(n_agents):
        header_c += ',r_eval_%d' % idx
        
    if experiment == 'sumo':
        for idx in range(n_agents):
            header += ',route_%d,lane_%d,goal_%d' % (idx,idx,idx)
    header_c += ',r_eval_local,duration (s)'
    header += '\n'
    header_c += '\n'
    if not os.path.exists('../log/%s' % dir_name):
        os.makedirs('../log/%s' % dir_name)
    with open('../log/%s/log.csv' % dir_name, 'w') as f:
        f.write(header)
    with open('../log/%s/log_century.csv' % dir_name, 'w') as f:
        f.write(header_c)
    
    if dual_buffer:
        buf = replay_buffer_dual.Replay_Buffer(size=buffer_size)
    else:
        buf = replay_buffer.Replay_Buffer(size=buffer_size)
    
    t_start = time.time()
    
    dist_action = np.zeros(l_action)
    step = 0
    # Each iteration is a training episode
    for idx_episode in range(1, N_train+1):
        # print("Episode", idx_episode)
        if experiment == "sumo":
            t_ms = sim.traci.simulation.getCurrentTime()
            # SUMO time functions return negative values afer 24 days (in millisecond) of simulation time
            # Hence use 0 for departure time, essentially triggering an immediate departure
            if 0 < t_ms and t_ms < 2073600e3:
                depart_times = [np.random.normal(t_ms/1000.0 + depart_mean[idx], depart_stdev) for idx in range(n_agents)]
            else:
                depart_times = [0 for idx in range(n_agents)]
            
            # Goals for input to policy and value function
            goals = np.zeros([n_agents, l_goal])
            list_routes = ['route_straight'] * n_agents
            list_lanes = [0] * n_agents
            list_goal_lane = [0] * n_agents
            rand_num = random.random()
            if rand_num < prob_random:
                # Random settings for route, lane and goal
                init = 'Random'
                for idx in range(n_agents):
                    route = 'route_straight'
                    lane = np.random.choice([0,1,2,3], p=np.ones(4)*0.25)
                    goal_lane = np.random.choice(np.arange(l_goal), p=np.ones(l_goal)/float(l_goal))
                    list_routes[idx] = route
                    list_lanes[idx] = lane
                    list_goal_lane[idx] = goal_lane
                    goals[idx, goal_lane] = 1
            else:
                init = 'Preset'
                # Use predetermined values for route, lane, goal
                for idx in range(n_agents):
                    list_routes[idx] = list_routes_fixed[idx]
                    goal_lane = list_goals_fixed[idx]
                    list_goal_lane[idx] = goal_lane
                    list_lanes[idx] = list_lanes_fixed[idx]
                    goals[idx, goal_lane] = 1
            
            env = multicar_simple.Multicar(sim, n_agents, list_goal_lane,
                                           list_goal_pos, list_routes,
                                           list_speeds, list_lanes,
                                           init_positions, list_id,
                                           list_vtypes, depart_times, total_length=total_length,
                                           total_width=total_width, safety=True)
            global_state, local_others, local_self, done = env.reset()
        elif experiment == "particle":
            global_state, local_others, local_self, done = env.reset()
            goals = np.zeros([n_agents, l_goal])
            for idx in range(n_agents):
                goals[idx] = env.world.landmarks[idx].state.p_pos
        elif experiment == "checkers":
            if n_agents == 1:
                if np.random.randint(2) == 0:
                    goals = np.array([[1,0]])
                else:
                    goals = np.array([[0,1]])
            else:
                goals = np.eye(n_agents)
            global_state, local_others, local_self_t, local_self_v, done = env.reset(goals)
            actions_prev = np.zeros(n_agents, dtype=np.int)
    
        reward_global = 0
        reward_local = np.zeros(n_agents)
    
        # step = 0
        summarized = False
        if dual_buffer:
            buf_episode = []
        while not done:
    
            if idx_episode < pretrain_episodes and (stage == 1 or train_from_nothing == 1):
                # Random actions when filling replay buffer
                actions = np.random.randint(0, l_action, n_agents)
            else:
                # Run actor network for all agents as batch
                if experiment == 'checkers':
                    actions = alg.run_actor(actions_prev, local_others, local_self_t, local_self_v, goals, epsilon, sess)
                else:
                    actions = alg.run_actor(local_others, local_self, goals, epsilon, sess)
    
            dist_action[actions[0]] += 1
            if experiment == 'sumo':
                # check feasible actions
                actions = env.check_actions(actions)
    
            # step environment
            if experiment == 'checkers':
                next_global_state, next_local_others, next_local_self_t , next_local_self_v, reward, local_rewards, done = env.step(actions)
            else:
                next_global_state, next_local_others, next_local_self, reward, local_rewards, done = env.step(actions)
    
            step += 1
            
            # store transition into memory
            if dual_buffer:
                if experiment == 'checkers':
                    buf_episode.append( np.array([ global_state[0], global_state[1], np.array(local_others), np.array(local_self_t), np.array(local_self_v), actions_prev, actions, reward, local_rewards, next_global_state[0], next_global_state[1], np.array(next_local_others), np.array(next_local_self_t), np.array(next_local_self_v), done, goals]) )
                else:
                    buf_episode.append( np.array([ global_state, np.array(local_others), np.array(local_self), actions, reward, local_rewards, next_global_state, np.array(next_local_others), np.array(next_local_self), done, goals]) )
            else:
                if experiment == 'checkers':
                    buf.add( np.array([ global_state[0], global_state[1], np.array(local_others), np.array(local_self_t), np.array(local_self_v), actions_prev, actions, reward, local_rewards, next_global_state[0], next_global_state[1], np.array(next_local_others), np.array(next_local_self_t), np.array(next_local_self_v), done, goals]) )
                else:
                    buf.add( np.array([ global_state, np.array(local_others), np.array(local_self), actions, reward, local_rewards, next_global_state, np.array(next_local_others), np.array(next_local_self), done, goals]) )
    
            if (idx_episode >= pretrain_episodes) and (step % steps_per_train == 0):
                # Sample batch of transitions from replay buffer
                batch = buf.sample_batch(batch_size)
    
                if summarize and idx_episode % period == 0 and not summarized:
                    # Write TF summary every <period> episodes,
                    # at the first <steps_per_train> step
                    alg.train_step(sess, batch, epsilon, idx_episode, summarize=True, writer=writer)
                    summarized = True
                else:
                    alg.train_step(sess, batch, epsilon, idx_episode, summarize=False, writer=None)
    
            global_state = next_global_state
            local_others = next_local_others
            if experiment == 'checkers':
                local_self_t = next_local_self_t
                local_self_v = next_local_self_v
                actions_prev = actions
            else:
                local_self = next_local_self
    
            reward_local += local_rewards
            reward_global += reward
    
        if dual_buffer:
            if experiment == 'sumo':
                buf.add(buf_episode, np.sum(reward_local) < threshold)
            elif experiment == 'particle':
                buf.add(buf_episode, scenario.collisions != 0)
    
        if idx_episode >= pretrain_episodes and epsilon > epsilon_end:
            epsilon -= epsilon_step
    
        reward_local_century += reward_local
        reward_global_century += reward_global
    
    
        # ----------- Log performance --------------- #
        
        if idx_episode % period == 0:
            dist_action = dist_action / np.sum(dist_action)
            t_end = time.time()    
            print("\n Evaluating")
            if experiment == 'sumo':
                r_local_eval, r_global_eval = evaluate.test(N_eval, sim, sess, depart_mean, depart_stdev, n_agents, l_goal, list_routes_fixed, list_lanes_fixed, list_goals_fixed, prob_random, list_goal_pos, list_speeds, init_positions, list_id, list_vtypes, alg)
                if np.all(r_local_eval > save_threshold):
                    saver.save(sess, '../saved/%s/model_good_%d.ckpt' % (dir_name, idx_episode))
            elif experiment == 'particle':
                r_local_eval, r_global_eval = evaluate.test_particle(N_eval, env, sess, n_agents, l_goal, alg, render=False)
            elif experiment == 'checkers':
                r_local_eval, r_global_eval = evaluate.test_checkers(N_eval, env, sess, n_agents, alg)
                if stage==1 and np.sum(r_local_eval) > 9.0:
                    saver.save(sess, '../saved/%s/model_good_%d.ckpt' % (dir_name, idx_episode))
            s = '%d,%.2f,' % (idx_episode, reward_global_century/float(period))
            s += ','.join(['{:.2f}'.format(val/float(period)) for val in reward_local_century])
            s += ',%.2f,' % (r_global_eval)
            s += ','.join(['{:.2f}'.format(val) for val in r_local_eval])
            s += ',%.2f,%d' % (np.sum(r_local_eval), int(t_end - t_start))
            s += '\n'
            print(s)
            with open('../log/%s/log_century.csv' % dir_name, 'a') as f:
                f.write(s)
            reward_local_century = np.zeros(n_agents)
            reward_global_century = 0
            print("Action distribution ", dist_action)
            if dual_buffer:
                print("length buffer good %d, length buffer others %d, epsilon %.3f" % (len(buf.memory_2), len(buf.memory_1), epsilon))
            else:
                print("epsilon %.3f" % epsilon)
            dist_action = np.zeros(l_action)
    
            t_start = time.time()
    
        s = '%d,%.2f,' % (idx_episode, reward_global)
        s += ','.join(['{:.2f}'.format(val) for val in reward_local])
        if experiment == 'sumo':
            for idx in range(n_agents):
                s += ',%d,%d,%d' % (map_route_idx[list_routes[idx]], list_lanes[idx], list_goal_lane[idx])
        s += '\n'
        with open('../log/%s/log.csv' % dir_name, 'a') as f:
            f.write(s)
    
    print("Saving stage %d variables" % stage)
    if not os.path.exists('../saved/%s' % dir_name):
        os.makedirs('../saved/%s' % dir_name)
    saver.save(sess, '../saved/%s/model_final.ckpt' % dir_name)

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    train_function(config)

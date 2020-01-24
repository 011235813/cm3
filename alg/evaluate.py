import numpy as np

import random
import sys
sys.path.append('../')

from env import multicar_simple


def test(n_eval, sim, sess, depart_mean, depart_stdev, n_agents,
         l_goal, list_routes_fixed, list_lanes_fixed,
         list_goals_fixed, prob_random, list_goal_pos, list_speeds,
         init_positions, list_id, list_vtypes, alg):

    epsilon = 0
    reward_local_total = np.zeros(n_agents)
    reward_global_total = 0

    for idx_eval in range(1, n_eval+1):

        t_ms = sim.traci.simulation.getCurrentTime()
        if 0 < t_ms and t_ms < 2073600e3:
            depart_times = [np.random.normal(t_ms/1000.0 + depart_mean[idx], depart_stdev) for idx in range(n_agents)]
        else:
            depart_times = [0 for idx in range(n_agents)]

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
                                       list_vtypes, depart_times, total_length=200,
                                       total_width=12.8, safety=True)
    
        global_state, local_t, local_v, done = env.reset()
        prev_actions = np.zeros(n_agents, dtype=int)
        reward_global = 0
        reward_local = np.zeros(n_agents)        

        while not done:
            actions = alg.run_actor(local_t, local_v,
                                    goals, epsilon, sess)
            actions = env.check_actions(actions)

            # step environment
            next_global_state, next_local_t, next_local_v, reward, local_rewards, done = env.step(actions)

            global_state = next_global_state
            local_t = next_local_t
            local_v = next_local_v
            prev_actions = actions

            reward_local += local_rewards
            reward_global += reward

        reward_local_total += reward_local
        reward_global_total += reward_global

    return reward_local_total/float(n_eval), reward_global_total/float(n_eval)


def test_particle(n_eval, env, sess, n_agents, l_goal, alg, render=False):

    epsilon = 0
    reward_local_total = np.zeros(n_agents)
    reward_global_total = 0

    for idx_eval in range(1, n_eval+1):

        global_state, local_others, local_v, done = env.reset()
        goals = np.zeros([n_agents, l_goal])
        for idx in range(n_agents):
            goals[idx] = env.world.landmarks[idx].state.p_pos
        prev_actions = np.zeros(n_agents, dtype=int)
        reward_global = 0
        reward_local = np.zeros(n_agents)        

        while not done:
            actions = alg.run_actor(local_others, local_v, goals, epsilon, sess)

            # step environment
            next_global_state, next_local_others, next_local_v, reward, local_rewards, done = env.step(actions)
            if render:
                if idx_eval == 1:
                    env.render()

            global_state = next_global_state
            local_others = next_local_others
            local_v = next_local_v
            prev_actions = actions

            reward_local += local_rewards
            reward_global += reward

        reward_local_total += reward_local
        reward_global_total += reward_global

    return reward_local_total/float(n_eval), reward_global_total/float(n_eval)


def test_particle_coma(n_eval, env, sess, n_agents, alg, render=False):

    epsilon = 0
    reward_local_total = np.zeros(n_agents)
    reward_global_total = 0

    for idx_eval in range(1, n_eval+1):

        global_state, local_v, done = env.reset()
        reward_global = 0
        reward_local = np.zeros(n_agents)        

        while not done:
            actions = alg.run_actor(local_v, epsilon, sess)

            # step environment
            next_global_state, next_local_v, reward, local_rewards, done = env.step(actions)
            if render:
                if idx_eval == 1:
                    env.render()

            global_state = next_global_state
            local_v = next_local_v

            reward_local += local_rewards
            reward_global += reward

        reward_local_total += reward_local
        reward_global_total += reward_global

    return reward_local_total/float(n_eval), reward_global_total/float(n_eval)    


def test_checkers(n_eval, env, sess, n_agents, alg):

    epsilon = 0
    reward_local_total = np.zeros(n_agents)
    reward_global_total = 0
    dist_action = np.zeros((n_agents,5))

    for idx_eval in range(1, n_eval+1):

        if n_agents == 1:
            if np.random.randint(2) == 0:
                goals = np.array([[1,0]])
            else:
                goals = np.array([[0,1]])
        else:
            goals = np.eye(n_agents)
        global_state, obs_others, obs_self_t, obs_self_v, done = env.reset(goals)

        reward_global = 0
        reward_local = np.zeros(n_agents)
        actions_prev = np.zeros(n_agents, dtype=np.int)

        while not done:
            actions = alg.run_actor(actions_prev, obs_others, obs_self_t, obs_self_v, goals, epsilon, sess)
            for idx in range(n_agents):
                dist_action[idx,actions[idx]] += 1
            # step environment
            next_global_state, next_obs_others, next_obs_self_t, next_obs_self_v, reward, local_rewards, done = env.step(actions)

            global_state = next_global_state
            obs_others = next_obs_others
            obs_self_t = next_obs_self_t
            obs_self_v = next_obs_self_v
            actions_prev = actions

            reward_local += local_rewards
            reward_global += reward
            
        reward_local_total += reward_local
        reward_global_total += reward_global

    dist_action = dist_action / np.sum(dist_action)
    print("Action distribution during eval", dist_action)

    return reward_local_total/float(n_eval), reward_global_total/float(n_eval)    

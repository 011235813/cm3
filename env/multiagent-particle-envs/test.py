import make_env
import numpy as np
import json

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

np.set_printoptions(precision=2, suppress=True)

# with open('../../alg/config_particle_stage2_antipodal.json') as f:
#     config = json.load(f)
# n_agents = 4

with open('../../alg/config_particle_stage1.json') as f:
    config = json.load(f)
n_agents = 1

scenario = scenarios.load("multi-goal_spread.py").Scenario()
world = scenario.make_world(n_agents, config, 0.2)
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.done, max_steps=25)

print("Landmark locations")
for i, landmark in enumerate(env.world.landmarks):
    print("Landmark", i, landmark.state.p_pos)
print("Agent locations")
for i, agent in enumerate(env.world.agents):
    print("Agent", i, agent.state.p_pos)
    

for episode in range(10):

    print("Episode", episode)
    global_state, local_others, local_self, done = env.reset()
    env.render()
    print("Goals")
    for idx in range(n_agents):
        print(env.world.landmarks[idx].state.p_pos)
    print(global_state)
    print(local_others)
    print(local_self)
    print(done)
    
    while not done:
    
        l = input("Enter actions as a comma-separated string: ")
        actions = list(map(int, l.split(',')))
        # print(actions)
        # s,r,d,i = env.step(actions)
        next_global_state, next_local_others, next_local_self, reward, local_rewards, done = env.step(actions)
        env.render()
        print("Next global state", next_global_state)
        print("Next local others", next_local_others)
        print("Next local self", next_local_self)
        print("Reward", reward)
        print("Local rewards", local_rewards)
        print("Done", done)

    print("Number of collisions", scenario.collisions/2)

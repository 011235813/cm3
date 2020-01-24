import numpy as np
import pandas as pd
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random

colors = np.array([[221,127,106],
                   [204,169,120],
                   [191,196,139],
                   [176,209,152],
                   [152,209,202],
                   [152,183,209],
                   [152,152,209],
                   [185,152,209],
                   [209,152,203],
                   [209,152,161]])

class Scenario(BaseScenario):
    def make_world(self, n_agents, config, prob_random):
        """
        n_agents - number of agents (also equal to number of target landmarks)
        config - dictionary
        prob_random - probability of random agent and landmark initial locations
        """
        world = World()
        # set any world properties first
        world.dim_c = 0
        self.n_agents = n_agents
        self.agents_x = config['agents_x']
        self.agents_y = config['agents_y']
        self.landmarks_x = config['landmarks_x']
        self.landmarks_y = config['landmarks_y']
        # standard deviation of Gaussian noise on initial agent location
        self.initial_std = config['initial_std']
        self.prob_random = prob_random
        num_agents = n_agents
        num_landmarks = n_agents
        # deliberate False, to prevent environment.py from sharing reward
        world.collaborative = False 
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.idx = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            # agent.size = 0.1
            agent.reached = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.idx = i
            landmark.collide = False
            landmark.movable = False
            # landmark.size = 0.05
        # read colors
        # self.colors = np.loadtxt('colors.csv', delimiter=',')
        self.colors = colors
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # agent.color = np.array([0.35, 0.35, 0.85])
            agent.color = self.colors[i]/256
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.color = self.colors[i]/256
        # set initial states
        rand_num = random.random()
        for i, agent in enumerate(world.agents):
            if rand_num < self.prob_random:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            else:
                x = self.agents_x[i] + np.random.normal(0,self.initial_std)
                y = self.agents_y[i] + np.random.normal(0,self.initial_std)
                # agent.state.p_pos = np.array([self.agents_x[i], self.agents_y[i]])
                agent.state.p_pos = np.array([x, y])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.reached = False
        for i, landmark in enumerate(world.landmarks):
            if rand_num < self.prob_random:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            else:
                landmark.state.p_pos = np.array([self.landmarks_x[i], self.landmarks_y[i]])
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.collisions = 0

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    def reward(self, agent, world):
        # Agent is rewarded based on distance to target landmark, and penalized for collisions
        rew = 0
        target = world.landmarks[ agent.idx ]
        rew -= np.sqrt(np.sum(np.square( agent.state.p_pos - target.state.p_pos )))
        if rew >= -0.05:
            agent.reached = True
        else:
            agent.reached = False
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 1
                    # Note that this will double count, since reward()
                    # will be called by environment again for the other agent
                    self.collisions += 1
        return rew

    def done(self, agent, world):
        if agent.reached:
            return True
        return False

    def observation(self, agent, world):
        others = []
        for other in world.agents:
            if other is agent and self.n_agents > 1:
                # allow storing of self in other_pos when n=1, so that
                # np.concat works. It won't be used in alg.py anyway when n=1
                continue
            others.append(other.state.p_vel - agent.state.p_vel) # relative velocity
            others.append(other.state.p_pos - agent.state.p_pos) # relative position
        return np.concatenate([agent.state.p_vel, agent.state.p_pos]), np.concatenate(others)

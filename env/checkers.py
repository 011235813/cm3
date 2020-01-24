import numpy as np

class Checkers(object):

    def __init__(self, n_rows=3, n_columns=16, n_obs=2, agents_r=[0,2],
                 agents_c=[16,16], n_agents=1, max_steps=50):
        """
        n_rows - number of rows with collectible reward
        n_columns - number of columns with collectible reward, not including
                    column where agents are initialized
        n_obs - number of squares on left,right,front,back that
                agents can observe, centered on its current location
        agents_r : list of row indices of agents (before expansion)
        agents_c : list of column indices of agents (before expansion)
        """
        assert(n_rows % 2 == 1)
        assert(n_columns % 2 == 0)
        # Only n_rows and n_columns have green and orange squares
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_obs = n_obs
        # Total grid size is larger so that agents' observations are valid
        # when they are located on the boundary
        self.total_rows = self.n_rows + 2*self.n_obs
        self.total_columns = self.n_columns + 2*self.n_obs + 1

        # Used to determine episode termination
        self.max_collectible = self.n_rows * self.n_columns

        self.n_agents = n_agents
        self.max_steps = max_steps

        # Initial agent locations, situated in expanded grid 
        self.agents_r = np.array(agents_r) + self.n_obs
        self.agents_c = np.array(agents_c) + self.n_obs


    def populate_world(self):
        """
        Initialize the world with green, orange, empty, invalid indicators
        """
        # Fill in invalid cells
        self.world[0:self.total_rows, 0:self.n_obs, 2] = np.ones((self.total_rows, self.n_obs))
        self.world[0:self.n_obs, 0:self.total_columns, 2] = np.ones((self.n_obs, self.total_columns))
        self.world[self.n_obs+self.n_rows:self.total_rows, 0:self.total_columns, 2] = np.ones((self.n_obs, self.total_columns))
        self.world[self.n_obs:self.n_obs+self.n_rows, self.n_obs+self.n_columns+1:self.total_columns, 2] = np.ones((self.n_rows, self.n_obs))

        # Agent locations are also invalid, i.e. cannot move there
        for idx in range(self.n_agents):
            loc = self.agents_location[idx]
            self.world[loc[0], loc[1], 2] = -1

        # Fill in rewards
        green_first = True
        for row in range(self.n_obs, self.n_obs+self.n_rows):
            if green_first:
                self.world[row, self.n_obs:self.n_obs+self.n_columns:2, 0] = -np.ones((1, self.n_columns//2))
                self.world[row, self.n_obs+1:self.n_obs+self.n_columns:2, 1] = -np.ones((1, self.n_columns//2))
                green_first = False
            else:
                self.world[row, self.n_obs:self.n_obs+self.n_columns:2, 1] = -np.ones((1, self.n_columns//2))
                self.world[row, self.n_obs+1:self.n_obs+self.n_columns:2, 0] = -np.ones((1, self.n_columns//2))
                green_first = True


    def get_valid_grid(self):

        top = self.n_obs
        bot = self.n_obs + self.n_rows
        left = self.n_obs
        # the +1 is to include agents' starting column
        right = self.n_obs + self.n_columns + 1

        # No need to include the invalid channel, which is only used
        # for agents' observations
        return self.world[top:bot, left:right, 0:2]
        

    def get_global_state(self):
        """
        Returns 2-tuple:
        grid : [r,c,2] tensor, one channel for each of green and orange
        vec : list of 1D arrays, one array per agent, contains location and 
              number of collected green and orange so far by agent
        """

        grid = self.get_valid_grid()
        # each agent's coordinates and number of collected orange and green
        vec = []
        for idx in range(self.n_agents):
            location = np.reshape(self.agents_location[idx], 2)
            number_collected = np.reshape(self.agents_collected[idx], 2)
            vec.append( np.concatenate([location, number_collected]) )
        return grid, vec


    def get_obs(self, agent_location):
        """
        agent_location : np array containing coordinates [r,c]
        Return [n_obs+1, n_obs+1, 3] tensor centered at agent location
        """
        r = agent_location[0]
        c = agent_location[1]

        grid = np.array(self.world[r-self.n_obs:r+self.n_obs+1, c-self.n_obs:c+self.n_obs+1, :])
        # edit the "invalid" channel so that agent's current location is valid
        grid[self.n_obs, self.n_obs, 2] = 0

        return grid


    def normalize(self, location):
        """
        location : either 1D array or 2D array, where each row is (r,c) 
                   coordinate of an agent
        """
        loc = np.array(location, dtype=np.float)
        n_dims = len(loc.shape)
        if n_dims == 1:
            loc[0] = (loc[0] - self.total_rows/2.0) / self.total_rows
            loc[1] = (loc[1] - self.total_columns/2.0) / self.total_columns
        elif n_dims == 2:
            loc[:,0] = (loc[:,0] - self.total_rows/2.0) / self.total_rows
            loc[:,1] = (loc[:,1] - self.total_columns/2.0) / self.total_columns
        return loc


    def get_local_observation(self):

        list_obs_self_t = []
        list_obs_self_v = []
        list_obs_others = []
        for idx in range(self.n_agents):
            # [n_obs+1, n_obs+1] grid centered on agent location
            obs_self_t = self.get_obs(self.agents_location[idx]) # observation grid
            # Compute normalized agent coordinates
            obs_self_v = self.normalize(self.agents_location[idx])
            # Concatenate with normalized number of collected things
            obs_self_v = np.concatenate( [obs_self_v, self.agents_collected[idx]/(self.max_collectible/2.0)] )
            # list_obs_self.append(obs_self)
            list_obs_self_t.append( obs_self_t )
            list_obs_self_v.append( obs_self_v )
            # coordinates of all other agents
            # If only one agent, use its own location (needed for placeholder
            # to work, but it is ignored)
            if self.n_agents == 1:
                obs_others = np.reshape(self.normalize(self.agents_location[idx]), 2)
            else:
                obs_others = np.reshape(self.normalize(self.agents_location[np.arange(self.n_agents)!=idx,:]),
                                        (self.n_agents-1)*2)
            list_obs_others.append(obs_others)

        # Stopped here
        return list_obs_others, list_obs_self_t, list_obs_self_v


    def agent_act(self, idx, action):
        """
        idx : agent index
        action : integer from 0 to 4 (stay,up,down,left,right)
        """
        # Current location
        r = self.agents_location[idx,0]
        c = self.agents_location[idx,1]
        reward = 0
        if action == 0:
            pass
        elif action == 1 and self.world[r-1,c,2] == 0: # up
            self.world[r-1,c,2] = -1
            self.world[r,c,2] = 0
            self.agents_location[idx] = [r-1,c]
        elif action == 2 and self.world[r+1,c,2] == 0: # down
            self.world[r+1,c,2] = -1
            self.world[r,c,2] = 0
            self.agents_location[idx] = [r+1,c]
        elif action == 3 and self.world[r,c-1,2] == 0: # left
            self.world[r,c-1,2] = -1
            self.world[r,c,2] = 0
            self.agents_location[idx] = [r,c-1]
        elif action == 4 and self.world[r,c+1,2] == 0: # right
            self.world[r,c+1,2] = -1
            self.world[r,c,2] = 0
            self.agents_location[idx] = [r,c+1]
        else:
            # Penalty for trying to move to invalid location
            reward = -0.1
        return reward


    def get_reward(self, idx, goal_idx):
        """
        idx : agent index
        goal_idx : 0 or 1
        Return local agent reward
        If goal_idx == 0, return +1 for green and -1 for orange
        If goal_idx == 1, return -1 for green and +1 for orange
        Rewards, once collectd, are removed from the grid cell
        """
        # agent_act() was called, so this is the agent's new location
        r = self.agents_location[idx,0]
        c = self.agents_location[idx,1]
        if goal_idx == 0:
            if self.world[r,c,0] == -1: # green
                self.world[r,c,0] = 1
                reward = 1.0
                self.agents_collected[idx, 0] += 1
            elif self.world[r,c,1] == -1: # orange
                self.world[r,c,1] = 1
                reward = -0.5
                self.agents_collected[idx, 1] += 1
            else:
                reward = 0
        elif goal_idx == 1:
            if self.world[r,c,0] == -1: # green
                self.world[r,c,0] = 1
                reward = -0.5
                self.agents_collected[idx, 0] += 1
            elif self.world[r,c,1] == -1: # orange
                self.world[r,c,1] = 1
                reward = 1.0
                self.agents_collected[idx, 1] += 1
            else:
                reward = 0

        return reward


    def step(self, actions):
        """
        actions : list of integers from 0 to 4
        """
        local_rewards = []
        for idx in range(self.n_agents):
            penalty = self.agent_act(idx, actions[idx])
            goal_idx = np.where(self.goals[idx]==1)[0][0]
            reward = penalty + self.get_reward(idx, goal_idx)
            local_rewards.append(reward)
            # local_rewards.append(penalty+self.get_reward(idx))

        global_state = self.get_global_state()
        obs_others, obs_self_t, obs_self_v = self.get_local_observation()

        total_reward = np.sum(local_rewards)
        self.steps += 1
        
        if self.steps == self.max_steps:
            done = True
        elif self.n_agents == 1:
            goal_idx = np.where(self.goals[0]==1)[0][0]
            if goal_idx == 0 and np.sum(self.world[:,:,0]) == (self.max_collectible/2.0):
                done = True
            elif goal_idx == 1 and np.sum(self.world[:,:,1]) == (self.max_collectible/2.0):
                done = True
            else:
                done = False
        elif self.n_agents > 1:
            if np.sum(self.world[:,:,0:2]) == self.max_collectible:
                done = True
            else:
                done = False
        
        return global_state, obs_others, obs_self_t, obs_self_v, total_reward, local_rewards, done
        

    def reset(self, goals):

        self.world = np.zeros((self.total_rows, self.total_columns, 3))
        self.steps = 0
        self.goals = goals
        # If single agent, initial row is random 0,1,2
        if self.n_agents == 1:
            goal_idx = np.where(self.goals[0]==1)[0][0]
            if goal_idx == 0:
                self.agents_r = np.array([0]) + self.n_obs
            else:
                self.agents_r = np.array([2]) + self.n_obs

        # each row is one agent's (r,c) coordinates
        self.agents_location = np.zeros((self.n_agents, 2), dtype=np.int)
        self.agents_location[:,0] = self.agents_r
        self.agents_location[:,1] = self.agents_c

        self.populate_world()

        # each row is one agent's [#green, #orange]
        self.agents_collected = np.zeros((self.n_agents, 2))

        global_state = self.get_global_state()
        obs_others, obs_self_t, obs_self_v = self.get_local_observation()

        return global_state, obs_others, obs_self_t, obs_self_v, False


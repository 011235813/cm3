"""Implementations of QMIX for Checkers.

Same as alg_qmix.py, except that Checkers global state has more components.
"""

import numpy as np
import tensorflow as tf

import sys
import networks

class Alg(object):

    def __init__(self, experiment, dimensions, stage=1, n_agents=1,
                 tau=0.01, lr_Q=0.001, gamma=0.99, nn={}):
        """
        Same as alg_qmix. Checkers state has more components
        Inputs:
        experiment - string
        dimensions - dictionary containing tensor dimensions
                     (h,w,c) for tensor
                     l for 1D vector
        stage - curriculum stage (always 2 for IAC and COMA)
        tau - target variable update rate
        lr_Q - learning rates for optimizer
        gamma - discount factor
        """
        self.experiment = experiment
        if self.experiment == "checkers":
            # Global state
            self.rows_state = dimensions['rows_state']
            self.columns_state = dimensions['columns_state']
            self.channels_state = dimensions['channels_state']
            self.l_state = n_agents * dimensions['l_state_one']
            self.l_state_one_agent = dimensions['l_state_one']
            self.l_state_other_agents = (n_agents-1) * dimensions['l_state_one']
            # Agent observations
            self.l_obs_others = dimensions['l_obs_others']
            self.l_obs_self = dimensions['l_obs_self']
            # Dimensions for image input
            self.rows_obs = dimensions['rows_obs']
            self.columns_obs = dimensions['columns_obs']
            self.channels_obs = dimensions['channels_obs']

        self.l_action = dimensions['l_action']
        self.l_goal = dimensions['l_goal']

        self.n_agents = n_agents
        self.tau = tau
        self.lr_Q = lr_Q
        self.gamma = gamma
        self.nn = nn

        self.agent_labels = np.eye(self.n_agents)
        self.actions = np.eye(self.l_action)
        
        # Initialize computational graph
        self.create_networks(stage)
        self.list_initialize_target_ops, self.list_update_target_ops = self.get_assign_target_ops(tf.trainable_variables())
        self.create_train_op()

    def create_networks(self, stage):

        # Placeholders
        self.state_env = tf.placeholder(tf.float32, [None, self.rows_state, self.columns_state, self.channels_state], 'state_env')
        self.v_state = tf.placeholder(tf.float32, [None, self.l_state], 'v_state')
        self.v_goal_all = tf.placeholder(tf.float32, [None, self.n_agents*self.l_goal], 'v_goal_all')
        self.v_state_one_agent = tf.placeholder(tf.float32, [None, self.l_state_one_agent], 'v_state_one_agent')
        self.v_state_other_agents = tf.placeholder(tf.float32, [None, self.l_state_other_agents], 'v_state_other_agents')
        self.v_goal = tf.placeholder(tf.float32, [None, self.l_goal], 'v_goal')
        self.v_goal_others = tf.placeholder(tf.float32, [None, (self.n_agents-1)*self.l_goal], 'v_goal_others')
        self.v_labels = tf.placeholder(tf.float32, [None, self.n_agents])

        self.action_others = tf.placeholder(tf.float32, [None, self.n_agents-1, self.l_action], 'action_others')

        if self.experiment == "checkers":
            self.obs_self_t = tf.placeholder(tf.float32, [None, self.rows_obs, self.columns_obs, self.channels_obs], 'obs_self_t')
            self.obs_self_v = tf.placeholder(tf.float32, [None, self.l_obs_self], 'obs_self_v')
            self.obs_others = tf.placeholder(tf.float32, [None, self.l_obs_others], 'obs_others')
            self.actions_prev = tf.placeholder(tf.float32, [None, self.l_action], 'action_prev')

        # Individual agent networks
        # output dimension is [time * n_agents, q-values]
        with tf.variable_scope("Agent_main"):
            if self.experiment == 'checkers':
                self.agent_qs = networks.Qmix_single_checkers(self.actions_prev, self.obs_self_t, self.obs_self_v, self.obs_others, self.v_goal, f1=self.nn['A_conv_f'], k1=self.nn['A_conv_k'], n_h1=self.nn['A_n_h1'], n_h2=self.nn['A_n_h2'], n_actions=self.l_action)
        with tf.variable_scope("Agent_target"):
            if self.experiment == 'checkers':
                self.agent_qs_target = networks.Qmix_single_checkers(self.actions_prev, self.obs_self_t, self.obs_self_v, self.obs_others, self.v_goal, f1=self.nn['A_conv_f'], k1=self.nn['A_conv_k'], n_h1=self.nn['A_n_h1'], n_h2=self.nn['A_n_h2'], n_actions=self.l_action)

        self.argmax_Q = tf.argmax(self.agent_qs, axis=1)
        self.argmax_Q_target = tf.argmax(self.agent_qs_target, axis=1)

        # To extract Q-value from agent_qs and agent_qs_target; [batch*n_agents, l_action]
        self.actions_1hot = tf.placeholder(tf.float32, [None, self.l_action], 'actions_1hot')
        self.q_selected = tf.reduce_sum(tf.multiply(self.agent_qs, self.actions_1hot), axis=1)
        self.mixer_q_input = tf.reshape( self.q_selected, [-1, self.n_agents] ) # [batch, n_agents]

        self.q_target_selected = tf.reduce_sum(tf.multiply(self.agent_qs_target, self.actions_1hot), axis=1)
        self.mixer_target_q_input = tf.reshape( self.q_target_selected, [-1, self.n_agents] )

        # Mixing network
        with tf.variable_scope("Mixer_main"):
            self.mixer = networks.Qmix_mixer_checkers(self.mixer_q_input, self.state_env, self.v_state, self.v_goal_all, self.l_state, self.l_goal, self.n_agents, f1=self.nn['Q_conv_f'], k1=self.nn['Q_conv_k'])
        with tf.variable_scope("Mixer_target"):
            self.mixer_target = networks.Qmix_mixer_checkers(self.mixer_q_input, self.state_env, self.v_state, self.v_goal_all, self.l_state, self.l_goal, self.n_agents, f1=self.nn['Q_conv_f'], k1=self.nn['Q_conv_k'])

    def get_assign_target_ops(self, list_vars):

        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []

        list_Agent_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Agent_main')
        map_name_Agent_main = {v.name.split('main')[1] : v for v in list_Agent_main}
        list_Agent_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Agent_target')
        map_name_Agent_target = {v.name.split('target')[1] : v for v in list_Agent_target}
        
        if len(list_Agent_main) != len(list_Agent_target):
            raise ValueError("get_initialize_target_ops : lengths of Agent_main and Agent_target do not match")
        
        for name, var in map_name_Agent_main.items():
            # create op that assigns value of main variable to
            # target variable of the same name
            list_initial_ops.append( map_name_Agent_target[name].assign(var) )
        
        for name, var in map_name_Agent_main.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_Agent_target[name].assign( self.tau*var + (1-self.tau)*map_name_Agent_target[name] ) )

        list_Mixer_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
        map_name_Mixer_main = {v.name.split('main')[1] : v for v in list_Mixer_main}
        list_Mixer_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_target')
        map_name_Mixer_target = {v.name.split('target')[1] : v for v in list_Mixer_target}

        if len(list_Mixer_main) != len(list_Mixer_target):
            raise ValueError("get_initialize_target_ops : lengths of Mixer_main and Mixer_target do not match")

        # ops for equating main and target
        for name, var in map_name_Mixer_main.items():
            # create op that assigns value of main variable to
            # target variable of the same name
            list_initial_ops.append( map_name_Mixer_target[name].assign(var) )

        # ops for slow update of target toward main
        for name, var in map_name_Mixer_main.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_Mixer_target[name].assign( self.tau*var + (1-self.tau)*map_name_Mixer_target[name] ) )
        
        return list_initial_ops, list_update_ops

    def run_actor(self, actions_prev, obs_others, obs_self_t, obs_self_v, goals, epsilon, sess):
        """
        Get actions for all agents as a batch

        actions_prev - list of integers
        obs_others - list of vector or tensor describing other agents
        obs_self - list of observation grid centered on self
        goals - [n_agents, n_lanes]
        """
        # convert to batch
        obs_others = np.array(obs_others)
        obs_self_t = np.array(obs_self_t)
        obs_self_v = np.array(obs_self_v)

        actions_prev_1hot = np.zeros([self.n_agents, self.l_action])
        actions_prev_1hot[np.arange(self.n_agents), actions_prev] = 1

        feed = {self.obs_others:obs_others, self.obs_self_t:obs_self_t,
                self.obs_self_v:obs_self_v, self.v_goal:goals,
                self.actions_prev: actions_prev_1hot}
        actions_argmax = sess.run(self.argmax_Q, feed_dict=feed)

        actions = np.zeros(self.n_agents, dtype=int)
        for idx in range(self.n_agents):
            if np.random.rand(1) < epsilon:
                actions[idx] = np.random.randint(0, self.l_action)
            else:
                actions[idx] = actions_argmax[idx]

        return actions

    def create_train_op(self):
        # TD target calculated in train_step() using Mixer_target
        self.td_target = tf.placeholder(tf.float32, [None], 'td_target')
        self.loss_mixer = tf.reduce_mean(tf.square(self.td_target - tf.squeeze(self.mixer)))

        self.mixer_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.mixer_op = self.mixer_opt.minimize(self.loss_mixer)

    def create_summary(self):

        summaries_mixer = [tf.summary.scalar('loss_mixer', self.loss_mixer)]
        mixer_main_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')]
        for v in mixer_main_variables:
            summaries_Q.append(tf.summary.histogram(v.op.name, v))
        grads = self.Q_opt.compute_gradients(self.loss_mixer, mixer_main_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_Q = tf.summary.merge(summaries_mixer)

    def process_actions(self, n_steps, actions):
        """
        actions must have shape [time, agents],
        and values are action indices
        """
        # Each row of actions is one time step,
        # row contains action indices for all agents
        # Convert to [time, agents, l_action]
        # so each agent gets its own 1-hot row vector
        actions_1hot = np.zeros([n_steps, self.n_agents, self.l_action], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        actions_1hot[grid[0], grid[1], actions] = 1
        # Convert to format [time*agents, agents-1, l_action]
        # so that the set of <n_agent> actions at each time step
        # is duplicated <n_agent> times, and each duplicate
        # now contains all <n_agent>-1 actions representing
        # the OTHER agents actions
        list_to_interleave = []
        for n in range(self.n_agents):
            # extract all actions except agent n's action
            list_to_interleave.append( actions_1hot[:, np.arange(self.n_agents)!=n, :] )
        # interleave
        actions_others_1hot = np.zeros([self.n_agents*n_steps, self.n_agents-1, self.l_action])
        for n in range(self.n_agents):
            actions_others_1hot[n::self.n_agents, :, :] = list_to_interleave[n]
        # In-place reshape of actions to [time*n_agents, l_action]
        actions_1hot.shape = (n_steps*self.n_agents, self.l_action)

        return actions_1hot, actions_others_1hot

    def process_batch(self, batch):
        """
        Extract quantities of the same type from batch.
        Format batch so that each agent at each time step is one
        batch entry.
        Duplicate global quantities <n_agents> times to be
        compatible with this scheme.
        """
        # shapes are [time, ...original dims...]
        state_env = np.stack(batch[:,0]) # [time, grid]
        state_agents = np.stack(batch[:,1]) # [time, agents, l_state_one_agent]
        # note that *_local objects have shape
        # [time, agents, ...original dim...]
        obs_others = np.stack(batch[:,2]) # [time,agents,h,w,c] or [time, agents, obs_others]
        obs_self_t = np.stack(batch[:,3]) # [time,agents,row,column,channel]
        obs_self_v = np.stack(batch[:,4]) # [time,agents,l_obs_self]
        actions_prev = np.stack(batch[:,5])
        actions = np.stack(batch[:,6]) # [time,agents]
        reward = np.stack(batch[:,7]) # [time]
        reward_local = np.stack(batch[:,8]) # [time,agents]
        state_env_next = np.stack(batch[:,9]) # [time, grid]
        state_agents_next = np.stack(batch[:,10]) # [time, agents, l_state_one_agent]
        obs_others_next = np.stack(batch[:,11]) # [time,agents,h,w,c]
        obs_self_t_next = np.stack(batch[:,12]) # [time,agents,row,column,channel]
        obs_self_v_next = np.stack(batch[:,13]) # [time,agents,l_obs_self]
        done = np.stack(batch[:,14]) # [time]
        goals = np.stack(batch[:,15]) # [time, agents, l_goal]

        batch = None
    
        n_steps = state_agents.shape[0]
    
        # For all global quantities, for each time step,
        # duplicate values <n_agents> times for
        # batch processing of all agents
        reward = np.repeat(reward, self.n_agents, axis=0)

        # In-place reshape for *_local quantities,
        # so that one time step for one agent is considered
        # one batch entry
        if self.experiment == 'checkers':
            obs_others.shape = (n_steps*self.n_agents, self.l_obs_others)
            obs_others_next.shape = (n_steps*self.n_agents, self.l_obs_others)
            obs_self_t.shape = (n_steps*self.n_agents, self.rows_obs, self.columns_obs, self.channels_obs)
            obs_self_t_next.shape = (n_steps*self.n_agents, self.rows_obs, self.columns_obs, self.channels_obs)
            obs_self_v.shape = (n_steps*self.n_agents, self.l_obs_self)
            obs_self_v_next.shape = (n_steps*self.n_agents, self.l_obs_self)
 
        reward_local.shape = (n_steps*self.n_agents)
 
        actions_1hot, actions_others_1hot = self.process_actions(n_steps, actions)
        actions_prev_1hot = np.zeros([n_steps, self.n_agents, self.l_action], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        actions_prev_1hot[grid[0], grid[1], actions_prev] = 1
        actions_prev_1hot.shape = (n_steps*self.n_agents, self.l_action)
            
        return n_steps, state_env, state_agents, obs_others, obs_self_t, obs_self_v, actions_prev_1hot, actions_1hot, actions_others_1hot, reward, reward_local, state_env_next, state_agents_next, obs_others_next, obs_self_t_next, obs_self_v_next, done, goals

    def process_goals(self, goals, n_steps):
        """
        goals has shape [batch, n_agents, l_goal]
        convert to two streams:
        1. [n_agents * n_steps, l_goal] : each row is goal for one 
        agent, block of <n_agents> rows belong to one sampled transition from batch
        2. [n_agents * n_steps, (n_agents-1)*l_goal] : each row is 
        the goals of all OTHER agents, as a single row vector. Block of
        <n_agents> rows belong to one sampled transition from batch
        """
        # Reshape so that one time step for one agent is one batch entry
        goals_self = np.reshape(goals, (n_steps*self.n_agents, self.l_goal))
    
        goals_others = np.zeros((n_steps*self.n_agents, self.n_agents-1, self.l_goal))

        for n in range(self.n_agents):
            goals_others[n::self.n_agents, :, :] = goals[:, np.arange(self.n_agents)!=n, :]

        # Reshape to be [n_agents * n_steps, (n_agents-1)*l_goal]
        goals_others.shape = (n_steps*self.n_agents, (self.n_agents-1)*self.l_goal)
    
        return goals_self, goals_others

    def process_global_state(self, v_global, n_steps):
        """
        v_global has shape [n_steps, n_agents, l_state]
        Convert to three streams:
        1. [n_agents * n_steps, l_state_one_agent] : each row is state of one agent,
        and a block of <n_agents> rows belong to one sampled transition from batch
        2. [n_agents * n_steps, l_state_other_agents] : each row is the state of all
        OTHER agents, as a single row vector. A block of <n_agents> rows belong to one 
        sampled transition from batch
        3. [n_steps*n_agents, n_agents*l_state] : each row is concatenation of state of all agents
        For each time step, the row is duplicated <n_agents> times, since the same state s is used
        in <n_agents> different evaluations of Q(s,a^{-n},a^n)
        """
        # Reshape into 2D, each block of <n_agents> rows correspond to one time step
        v_global_one_agent = np.reshape(v_global, (n_steps*self.n_agents, self.l_state_one_agent))
        v_global_others = np.zeros((n_steps*self.n_agents, self.n_agents-1, self.l_state_one_agent))
        for n in range(self.n_agents):
            v_global_others[n::self.n_agents, :, :] = v_global[:, np.arange(self.n_agents)!=n, :]
        # Reshape into 2D, each row is state of all other agents, each block of
        # <n_agents> rows correspond to one time step
        v_global_others.shape = (n_steps*self.n_agents, (self.n_agents-1)*self.l_state_one_agent)

        v_global_concated = np.reshape(v_global, (n_steps, self.l_state))
        state = np.repeat(v_global_concated, self.n_agents, axis=0)

        return v_global_one_agent, v_global_others, state

    def train_step(self, sess, batch, epsilon, idx_train,
                   summarize=False, writer=None):

        # Each agent for each time step is now a batch entry
        n_steps, state_env, state_agents, obs_others, obs_self_t, obs_self_v, actions_prev_1hot, actions_1hot, actions_others_1hot, reward, reward_local, state_env_next, state_agents_next, obs_others_next, obs_self_t_next, obs_self_v_next, done, goals = self.process_batch(batch)
        
        goals_all = np.reshape(goals, (n_steps, self.n_agents*self.l_goal))
        goals_self, goals_others = self.process_goals(goals, n_steps)
        state_agents_next = np.reshape(state_agents_next, (n_steps, self.l_state))
        state_agents = np.reshape(state_agents, (n_steps, self.l_state))
        
        # Get argmax actions from target networks
        feed = {self.actions_prev : actions_1hot,
                self.obs_others : obs_others_next,
                self.obs_self_t : obs_self_t_next,
                self.obs_self_v : obs_self_v_next,
                self.v_goal : goals_self}
        argmax_actions = sess.run(self.argmax_Q_target, feed_dict=feed) # [batch*n_agents]
        # Convert to 1-hot
        actions_target_1hot = np.zeros([n_steps * self.n_agents, self.l_action], dtype=int)
        actions_target_1hot[np.arange(n_steps*self.n_agents), argmax_actions] = 1

        # Get Q_tot target value
        feed = {self.state_env : state_env_next,
                self.v_state : state_agents_next,
                self.v_goal_all : goals_all,
                self.actions_1hot : actions_target_1hot,
                self.actions_prev : actions_1hot,
                self.obs_others : obs_others_next,
                self.obs_self_t : obs_self_t_next,
                self.obs_self_v : obs_self_v_next,
                self.v_goal : goals_self}
        Q_tot_target = sess.run(self.mixer_target, feed_dict=feed)


        done_multiplier = -(done - 1)
        reward_total = np.sum(np.reshape(reward_local, (n_steps, self.n_agents)), axis=1)
        target = reward_total + self.gamma * np.squeeze(Q_tot_target) * done_multiplier

        feed = {self.state_env : state_env,
                self.v_state : state_agents,
                self.v_goal_all : goals_all,
                self.actions_1hot : actions_1hot,
                self.actions_prev : actions_prev_1hot,
                self.obs_others : obs_others,
                self.obs_self_t : obs_self_t,
                self.obs_self_v : obs_self_v,
                self.v_goal : goals_self,
                self.td_target : target}
        _ = sess.run(self.mixer_op, feed_dict=feed)

        sess.run(self.list_update_target_ops)

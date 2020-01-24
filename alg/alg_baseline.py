"""Implementations of IAC and COMA, selected via use_Q and use_V."""

import sys
import networks

import numpy as np
import tensorflow as tf


class Alg(object):

    def __init__(self, experiment, dimensions, stage=1, n_agents=1,
                 tau=0.01, lr_V=0.001, lr_Q=0.001,
                 lr_actor=0.0001, gamma=0.99, alpha=0.5, 
                 use_Q=1, use_V=1, nn={}, IAC=False):
        """Implementation of IAC and COMA

        Inputs:
            experiment: string
            dimensions: dictionary containing tensor dimensions
            (h,w,c) for tensor
            l for 1D vector
            stage: curriculum stage (always 2 for IAC and COMA)
            n_agents: int
            tau: target variable update rate
            lr_V, lr_Q, lr_actor: learning rates for optimizer
            gamma: discount factor
            alpha: weighting of local vs. global gradient
            use_Q: set to 1 and set use_V=0 and IAC=False to run COMA
            use_V: if 1, activates V network
            nn: dictionary that specifies neural net architecture
            IAC: set to True and set use_V to 1 to run IAC
        """
        self.experiment = experiment
        if self.experiment == "sumo":
            # global state is all agents' normalized (x,y,speed)
            self.l_state = n_agents * dimensions['l_state_one']
            self.l_state_one_agent = dimensions['l_state_one']
            self.l_state_other_agents = (n_agents-1) * dimensions['l_state_one']
            # Dimensions for image input
            self.h_obs = dimensions['h_obs']
            self.w_obs = dimensions['w_obs']
            self.c_obs = dimensions['c_obs']
            # Dimension of agent's observation of itself
            self.l_obs = dimensions['l_obs']
        elif self.experiment == "particle":
            # global state is all agents' velocity and position (landmarks are fed via goals)
            self.l_state = n_agents * dimensions['l_obs_self']
            # position and velocity
            self.l_state_one_agent = dimensions['l_obs_self']
            # other agents' position and velocity
            self.l_state_other_agents = (n_agents-1) * dimensions['l_obs_self']
            if n_agents == 1:
                self.l_obs_others = dimensions['l_obs_others']
            else:
                # relative position and velocity
                self.l_obs_others = (n_agents-1) * dimensions['l_obs_others']
            # agent's own velocity and position
            self.l_obs = dimensions['l_obs_self']
        self.l_action = dimensions['l_action']
        self.l_goal = dimensions['l_goal']

        self.n_agents = n_agents
        self.tau = tau
        self.lr_V = lr_V
        self.lr_Q = lr_Q
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.alpha = alpha
        self.use_Q = use_Q
        self.use_V = use_V
        self.nn = nn
        self.IAC = IAC

        self.agent_labels = np.eye(self.n_agents)

        # Initialize computational graph
        self.create_networks(stage)
        self.list_initialize_target_ops, self.list_update_target_ops = self.get_assign_target_ops(tf.trainable_variables())
        if self.use_V:
            self.create_local_critic_train_op()
        if self.n_agents > 1 and self.use_Q:
            self.create_global_critic_train_op()
        self.create_policy_gradient_op()

        # TF summaries
        self.create_summary()

    def create_networks(self, stage):
        """Creates placeholders and neural nets.
        Args:
            stage: int
        """
        # Placeholders
        self.v_state = tf.placeholder(tf.float32, [None, self.l_state], 'v_state')
        self.v_state_one_agent = tf.placeholder(tf.float32, [None, self.l_state_one_agent], 'v_state_one_agent')
        self.v_state_other_agents = tf.placeholder(tf.float32, [None, self.l_state_other_agents], 'v_state_other_agents')
        self.v_goal = tf.placeholder(tf.float32, [None, self.l_goal], 'v_goal')
        self.v_goal_others = tf.placeholder(tf.float32, [None, (self.n_agents-1)*self.l_goal], 'v_goal_others')
        self.v_labels = tf.placeholder(tf.float32, [None, self.n_agents])

        self.action_others = tf.placeholder(tf.float32, [None, self.n_agents-1, self.l_action], 'action_others')

        if self.experiment == "sumo":
            self.obs_others = tf.placeholder(tf.float32, [None, self.h_obs, self.w_obs, self.c_obs], 'obs_others')
        elif self.experiment == "particle":
            self.obs_others = tf.placeholder(tf.float32, [None, self.l_obs_others], 'obs_others')
        self.v_obs = tf.placeholder(tf.float32, [None, self.l_obs], 'v_obs')

        # Actor network
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
        with tf.variable_scope("Policy"):
            if self.experiment == 'sumo':
                probs = networks.actor_staged(self.obs_others, self.v_obs, self.v_goal, n_actions=self.l_action, stage=stage)
            elif self.experiment == 'particle':
                probs = networks.actor_particle(self.obs_others, self.v_obs, self.v_goal, n_actions=self.l_action, n_h1_others=self.nn['Actor_n_others'], n_h2=self.nn['Actor_n_h2'], stage=stage)
        # probs is normalized
        self.probs = (1-self.epsilon) * probs + self.epsilon/float(self.l_action)
        self.action_samples = tf.multinomial(tf.log(self.probs), 1)
        with tf.variable_scope("Policy_target"):
            if self.experiment == 'sumo':
                probs_target = networks.actor_staged(self.obs_others, self.v_obs, self.v_goal, n_actions=self.l_action, stage=stage)
            elif self.experiment == 'particle':
                probs_target = networks.actor_particle(self.obs_others, self.v_obs, self.v_goal, n_actions=self.l_action, n_h1_others=self.nn['Actor_n_others'], n_h2=self.nn['Actor_n_h2'], stage=stage)
        self.action_samples_target = tf.multinomial(tf.log( (1-self.epsilon)*probs_target + self.epsilon/float(self.l_action) ), 1)

        # V(s,g^n)
        if self.use_V:
            with tf.variable_scope("V_main"):
                if self.experiment == 'sumo':
                    if self.IAC:
                        self.V = networks.V_sumo_local(self.obs_others, self.v_obs, self.v_goal, n_conv_reduced=self.nn['V_n_others'], n_h2=self.nn['V_n_h2'], stage=stage)
                    else:
                        self.V = networks.V_sumo_global(self.v_state_one_agent, self.v_goal, self.v_state_other_agents, self.v_goal_others, n_h1_branch2=self.nn['V_n_others'], n_h2=self.nn['V_n_h2'], stage=stage)
                elif self.experiment == 'particle':
                    if self.IAC:
                        self.V = networks.V_particle_local(self.obs_others, self.v_obs, self.v_goal, n_h1_branch2=self.nn['V_n_others'], n_h2=self.nn['V_n_h2'], stage=stage)
                    else:
                        self.V = networks.V_particle_global(self.v_state_one_agent, self.v_goal, self.v_state_other_agents, self.v_goal_others, n_h1_branch2=self.nn['V_n_others'], n_h2=self.nn['V_n_h2'], stage=stage)                        
            with tf.variable_scope("V_target"):
                if self.experiment == 'sumo':
                    if self.IAC:
                        self.V_target = networks.V_sumo_local(self.obs_others, self.v_obs, self.v_goal, n_conv_reduced=self.nn['V_n_others'], n_h2=self.nn['V_n_h2'], stage=stage)
                    else:
                        self.V_target = networks.V_sumo_global(self.v_state_one_agent, self.v_goal, self.v_state_other_agents, self.v_goal_others, n_h1_branch2=self.nn['V_n_others'], n_h2=self.nn['V_n_h2'], stage=stage)
                elif self.experiment == 'particle':
                    if self.IAC:
                        self.V_target = networks.V_particle_local(self.obs_others, self.v_obs, self.v_goal, n_h1_branch2=self.nn['V_n_others'], n_h2=self.nn['V_n_h2'], stage=stage)
                    else:
                        self.V_target = networks.V_particle_global(self.v_state_one_agent, self.v_goal, self.v_state_other_agents, self.v_goal_others, n_h1_branch2=self.nn['V_n_others'], n_h2=self.nn['V_n_h2'], stage=stage)

        # Q(s, a^{-n}, g^n, g^{-n}, n, o^n)
        if self.n_agents > 1 and self.use_Q:
            with tf.variable_scope("Q_main"):
                if self.experiment == 'sumo':
                    self.Q = networks.Q_global(self.v_state, self.action_others, self.v_goal, self.v_goal_others, self.v_labels, self.v_obs, n_actions=self.l_action, stage=stage, units=self.nn['Q_units'])
                elif self.experiment == 'particle':
                    self.Q = networks.Q_global(self.v_state, self.action_others, self.v_goal, self.v_goal_others, self.v_labels, self.v_obs, n_actions=self.l_action, stage=stage, units=self.nn['Q_units'])
            with tf.variable_scope("Q_target"):
                if self.experiment == 'sumo':
                    self.Q_target = networks.Q_global(self.v_state, self.action_others, self.v_goal, self.v_goal_others, self.v_labels, self.v_obs, n_actions=self.l_action, stage=stage, units=self.nn['Q_units'])
                elif self.experiment == 'particle':
                    self.Q_target = networks.Q_global(self.v_state, self.action_others, self.v_goal, self.v_goal_others, self.v_labels, self.v_obs, n_actions=self.l_action, stage=stage, units=self.nn['Q_units'])

    def get_assign_target_ops(self, list_vars):
        """Returns ops for updating target networks."""
        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []

        if self.use_V:
            list_V_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
            map_name_V_main = {v.name.split('main')[1] : v for v in list_V_main}
            list_V_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_target')
            map_name_V_target = {v.name.split('target')[1] : v for v in list_V_target}
            
            if len(list_V_main) != len(list_V_target):
                raise ValueError("get_initialize_target_ops : lengths of V_main and V_target do not match")
            
            for name, var in map_name_V_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_V_target[name].assign(var) )
            
            for name, var in map_name_V_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_V_target[name].assign( self.tau*var + (1-self.tau)*map_name_V_target[name] ) )

        # For policy
        list_P_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy/')
        map_name_P_main = {v.name.split('Policy')[1] : v for v in list_P_main}
        list_P_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_target')
        map_name_P_target = {v.name.split('target')[1] : v for v in list_P_target}

        if len(list_P_main) != len(list_P_target):
            raise ValueError("get_initialize_target_ops : lengths of P_main and P_target do not match")
        # ops for equating main and target
        for name, var in map_name_P_main.items():
            list_initial_ops.append( map_name_P_target[name].assign(var) )
        # ops for slow update of target toward main
        for name, var in map_name_P_main.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_P_target[name].assign( self.tau*var + (1-self.tau)*map_name_P_target[name] ) )

        # Repeat for Q if needed
        if self.n_agents > 1 and self.use_Q:
            list_Q_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_main')
            map_name_Q_main = {v.name.split('main')[1] : v for v in list_Q_main}
            list_Q_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_target')
            map_name_Q_target = {v.name.split('target')[1] : v for v in list_Q_target}

            if len(list_Q_main) != len(list_Q_target):
                raise ValueError("get_initialize_target_ops : lengths of Q_main and Q_target do not match")

            # ops for equating main and target
            for name, var in map_name_Q_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_Q_target[name].assign(var) )

            # ops for slow update of target toward main
            for name, var in map_name_Q_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_Q_target[name].assign( self.tau*var + (1-self.tau)*map_name_Q_target[name] ) )
        
        return list_initial_ops, list_update_ops

    def run_actor(self, local_others, local_v, goals, epsilon, sess):
        """Gets actions for all agents as a batch.

        Args:
            local_others: list of vector or tensor describing other agents
                           (may include self if using observation grid)
            local_v: list of 1D vectors
            goals: [n_agents, n_lanes]
            epsilon: float in [0,1]
            sess: TF session
        
        Returns:
            np.array of actions for all agents
        """
        # convert to batch
        obs_others = np.array(local_others)
        v_obs = np.array(local_v)

        feed = {self.obs_others:obs_others, self.v_obs:v_obs,
                self.v_goal:goals, self.epsilon:epsilon}
        action_samples_res = sess.run(self.action_samples, feed_dict=feed)
        return np.reshape(action_samples_res, action_samples_res.shape[0])
    
    def run_actor_target(self, local_others, local_v, goals, epsilon, sess):
        """Gets actions from the slowly-updating policy."""
        feed = {self.obs_others:local_others, self.v_obs:local_v,
                self.v_goal:goals, self.epsilon:epsilon}
        action_samples_res = sess.run(self.action_samples_target, feed_dict=feed)
        return np.reshape(action_samples_res, action_samples_res.shape[0])

    def create_local_critic_train_op(self):
        # TD target calculated in train_step() using V_target
        self.V_td_target = tf.placeholder(tf.float32, [None], 'V_td_target')
        self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.V)))

        self.V_opt = tf.train.AdamOptimizer(self.lr_V)
        self.V_op = self.V_opt.minimize(self.loss_V)

    def create_global_critic_train_op(self):
        # TD target calculated in train_step() using Q_target
        self.Q_td_target = tf.placeholder(tf.float32, [None], 'Q_td_target')
        self.actions_self_1hot = tf.placeholder(tf.float32, [None, self.l_action], 'actions_self_1hot')
        # Get Q-value of action actually taken by point-wise mult
        self.Q_action_taken = tf.reduce_sum( tf.multiply( self.Q, self.actions_self_1hot ), axis=1 )
        self.loss_Q = tf.reduce_mean(tf.square(self.Q_td_target - self.Q_action_taken))
        self.Q_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.Q_op = self.Q_opt.minimize(self.loss_Q)

    def create_policy_gradient_op(self):

        # batch of 1-hot action vectors
        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action], 'action_taken')
        # self.probs has shape [batch, l_action]
        log_probs = tf.log(tf.reduce_sum(tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        # --------------- COMA -------------------- #
        if self.n_agents > 1 and self.use_Q:
            # Q-values of the action actually taken [batch_size]
            self.Q_evaluated = tf.placeholder(tf.float32, [None, self.l_action], 'Q_evaluated')
            self.COMA_1 = tf.reduce_sum(tf.multiply(self.Q_evaluated, self.action_taken), axis=1)
            # Use all Q-values at output layer [batch_size]
            self.probs_evaluated = tf.placeholder(tf.float32, [None, self.l_action])
            self.COMA_2 = tf.reduce_sum(tf.multiply(self.Q_evaluated, self.probs_evaluated), axis=1)
            self.COMA = tf.subtract(self.COMA_1, self.COMA_2)
            self.policy_loss_global = -tf.reduce_mean( tf.reduce_sum( tf.reshape( tf.multiply(log_probs, self.COMA), [-1, self.n_agents] ), axis=1) )
        # -------------- End COMA ----------------- #

        if self.use_V and not self.IAC:
            self.V_evaluated = tf.placeholder(tf.float32, [None], 'V_evaluated')
            self.V_td_error = self.V_td_target - self.V_evaluated
            sum_log_probs = tf.reduce_sum( tf.reshape(log_probs, [-1, self.n_agents]), axis=1 )
            sum_td_error = tf.reduce_sum( tf.reshape(self.V_td_error, [-1, self.n_agents]), axis=1 )
            self.policy_loss_local = -tf.reduce_mean( tf.multiply( sum_log_probs, sum_td_error ) )
        elif self.use_V and self.IAC:
            self.V_evaluated = tf.placeholder(tf.float32, [None], 'V_evaluated')
            self.V_td_error = self.V_td_target - self.V_evaluated
            self.policy_loss_local = -tf.reduce_mean( tf.multiply( log_probs, self.V_td_error ) )

        if self.n_agents > 1 and self.use_Q and self.use_V:
            self.policy_loss = self.alpha * self.policy_loss_local + (1-self.alpha) * self.policy_loss_global
        elif self.n_agents > 1 and self.use_Q and self.use_V==0:
            self.policy_loss = self.policy_loss_global
        else:
            self.policy_loss = self.policy_loss_local

        self.policy_opt = tf.train.AdamOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.minimize(self.policy_loss)

    def create_summary(self):

        if self.use_V:
            summaries_V = [tf.summary.scalar('loss_V', self.loss_V)]
            V_main_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')]
            for v in V_main_variables:
                summaries_V.append( tf.summary.histogram(v.op.name, v) )
            grads = self.V_opt.compute_gradients(self.loss_V, V_main_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries_V.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
            self.summary_op_V = tf.summary.merge(summaries_V)
                       
        if self.n_agents > 1 and self.use_Q:
            summaries_Q = [tf.summary.scalar('loss_Q', self.loss_Q)]
            Q_main_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_main')]
            for v in Q_main_variables:
                summaries_Q.append(tf.summary.histogram(v.op.name, v))
            grads = self.Q_opt.compute_gradients(self.loss_Q, Q_main_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries_Q.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
            self.summary_op_Q = tf.summary.merge(summaries_Q)

        summaries_policy = [tf.summary.scalar('policy_loss', self.policy_loss)]
        if self.use_V:
            summaries_policy.append(tf.summary.scalar('policy_loss_local', self.policy_loss_local))
        if self.n_agents > 1 and self.use_Q:
            summaries_policy.append(tf.summary.scalar('policy_loss_global', self.policy_loss_global))
        policy_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy')]
        for v in policy_variables:
            summaries_policy.append(tf.summary.histogram(v.op.name, v))
        grads = self.policy_opt.compute_gradients(self.policy_loss, policy_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_policy.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_policy = tf.summary.merge(summaries_policy)

    def process_actions(self, n_steps, actions):
        """Reformats actions for better matrix computation.

        Args:
            n_steps: int
            actions: np.array shape [time, agents], values are action indices

        Returns:
            1. actions_1hot [n_agents * n_steps, l_action] : each row is
            action taken by one agent at one time step
            2. actions_others_1hot [n_agents * n_steps, n_agents-1, l_action] :
            each row is for one agent at one time step, containing all 
            (n-1) other agents' actions
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
        """Extract quantities of the same type from batch.

        Formats batch so that each agent at each time step is one batch entry.
        Duplicate global quantities <n_agents> times to be
        compatible with this scheme.

        Args:
            batch: a large np.array containing a batch of transitions.
        
        Returns many np.arrays
        """
        # shapes are [time, ...original dims...]
        v_global = np.stack(batch[:,0]) # [time, agents, l_state_one_agent]
        # note that *_local objects have shape
        # [time, agents, ...original dim...]
        obs_others = np.stack(batch[:,1]) # [time,agents,h,w,c] or [time, agents, obs_others]
        v_local = np.stack(batch[:,2]) # [time,agents,l]
        actions = np.stack(batch[:,3]) # [time,agents]
        reward = np.stack(batch[:,4]) # [time]
        reward_local = np.stack(batch[:,5]) # [time,agents]
        v_global_next = np.stack(batch[:,6]) # [time, agents, l_state_one_agent]
        obs_others_next = np.stack(batch[:,7]) # [time,agents,h,w,c]
        v_local_next = np.stack(batch[:,8]) # [time,agents,l]
        done = np.stack(batch[:,9]) # [time]
        goals = np.stack(batch[:,10]) # [time, agents, l_goal]

        batch = None
    
        n_steps = v_global.shape[0]
    
        # For all global quantities, for each time step,
        # duplicate values <n_agents> times for
        # batch processing of all agents
        reward = np.repeat(reward, self.n_agents, axis=0)
        done = np.repeat(done, self.n_agents, axis=0)

        # In-place reshape for *_local quantities,
        # so that one time step for one agent is considered
        # one batch entry
        if self.experiment == 'sumo':
            obs_others.shape = (n_steps*self.n_agents, self.h_obs,
                                self.w_obs, self.c_obs)
            obs_others_next.shape = (n_steps*self.n_agents, self.h_obs,
                                     self.w_obs, self.c_obs)
        elif self.experiment == 'particle':
            obs_others.shape = (n_steps*self.n_agents, self.l_obs_others)
            obs_others_next.shape = (n_steps*self.n_agents, self.l_obs_others)
        v_local.shape = (n_steps*self.n_agents, self.l_obs)
        reward_local.shape = (n_steps*self.n_agents)
        v_local_next.shape = (n_steps*self.n_agents, self.l_obs)

        actions_1hot, actions_others_1hot = self.process_actions(n_steps, actions)
            
        return n_steps, v_global, obs_others, v_local, actions_1hot, actions_others_1hot, reward, reward_local, v_global_next, obs_others_next, v_local_next, done, goals

    def process_goals(self, goals, n_steps):
        """Reformats goals for better matrix computation.

        Args:
            goals: np.array with shape [batch, n_agents, l_goal]

        Returns: two np.arrays
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
        """Reformats global state for more efficient computation of the gradient.

        Args:
            v_global: has shape [n_steps, n_agents, l_state_one_agent]
            n_steps: int

        Returns: three streams
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
        n_steps, v_global, obs_others, v_local, actions_1hot, actions_others_1hot, reward, reward_local, v_global_next, obs_others_next, v_local_next, done, goals = self.process_batch(batch)
        
        goals_self, goals_others = self.process_goals(goals, n_steps)
        v_global_one_agent, v_global_others, state = self.process_global_state(v_global, n_steps)
        v_global_one_agent_next, v_global_others_next, state_next = self.process_global_state(v_global_next, n_steps)

        # Create 1-hot agent labels [n_steps*n_agents, n_agents]
        agent_labels = np.tile(self.agent_labels, (n_steps,1))

        # ------------ Train local critic ----------------#
        
        if self.use_V:
            if self.IAC:
                # V_target(o^n_{t+1}, g^n). V_next_res = V(o^n_{t+1},g^n) used in policy gradient
                feed = {self.obs_others : obs_others_next,
                        self.v_obs : v_local_next,
                        self.v_goal : goals_self}
            else:
                # Get V_target(s',g). V_next_res = V(s',g) is used later in policy gradient
                feed = {self.v_state_one_agent : v_global_one_agent_next,
                        self.v_goal : goals_self,
                        self.v_state_other_agents : v_global_others_next,
                        self.v_goal_others : goals_others}
            V_target_res, V_next_res = sess.run([self.V_target, self.V], feed_dict=feed)
            V_target_res = np.squeeze(V_target_res)
            V_next_res = np.squeeze(V_next_res)
            # if true, then 0, else 1
            done_multiplier = -(done - 1)
            V_td_target = reward_local + self.gamma * V_target_res * done_multiplier
            
            # Run optimizer for local critic
            if self.IAC:
                feed = {self.V_td_target : V_td_target,
                        self.obs_others : obs_others,
                        self.v_obs : v_local,
                        self.v_goal : goals_self}
            else:
                feed = {self.V_td_target : V_td_target,
                        self.v_state_one_agent : v_global_one_agent,
                        self.v_goal : goals_self,
                        self.v_state_other_agents : v_global_others,
                        self.v_goal_others : goals_others}
            if summarize:
                # Get V_res = V(s,g) for use later in policy gradient
                summary, _, V_res = sess.run([self.summary_op_V, self.V_op, self.V], feed_dict=feed)
                writer.add_summary(summary, idx_train)
            else:
                _, V_res = sess.run([self.V_op, self.V], feed_dict=feed)

        # ----------- Train global critic -------------- #

        if self.n_agents > 1 and self.use_Q:
            # Need a_{t+1} to evaluate TD target for Q
            # actions is single dimension [n_steps*n_agents]
            actions = self.run_actor_target(obs_others_next, v_local_next, goals_self, epsilon, sess)
            # Now each row is one time step, containing action
            # indices for all agents
            actions_r = np.reshape(actions, [n_steps, self.n_agents])
            actions_self_next, actions_others_next = self.process_actions(n_steps, actions_r)
            feed = {self.v_state : state_next,
                    self.action_others : actions_others_next,
                    self.v_goal : goals_self,
                    self.v_goal_others : goals_others,
                    self.v_labels : agent_labels,
                    self.v_obs : v_local_next}
            Q_target_res = sess.run(self.Q_target, feed_dict=feed)
            Q_target_res = np.sum(Q_target_res * actions_self_next, axis=1)

            # if true, then 0, else 1
            done_multiplier = -(done - 1)
            Q_td_target = reward + self.gamma * Q_target_res * done_multiplier
            
            feed = {self.Q_td_target : Q_td_target,
                    self.actions_self_1hot : actions_1hot,
                    self.v_state : state,
                    self.action_others : actions_others_1hot,
                    self.v_goal : goals_self,
                    self.v_goal_others : goals_others,
                    self.v_labels : agent_labels,
                    self.v_obs : v_local}
                    
            # Run optimizer for global critic
            if summarize:
                summary, _ = sess.run([self.summary_op_Q, self.Q_op], feed_dict=feed)
                writer.add_summary(summary, idx_train)
            else:
                _ = sess.run(self.Q_op, feed_dict=feed)

        # --------------- Train policy ------------- #

        if self.use_V:
            # Already computed V_res when running V_op above
            V_res = np.squeeze(V_res)
            V_td_target = reward_local + self.gamma * V_next_res * done_multiplier

        if self.n_agents > 1 and self.use_Q:
            # Evaluate Q(s,a,g) and pi(a|o,g) to feed placeholders for policy_loss_global
            feed = {self.v_state : state,
                    self.v_obs : v_local,
                    self.action_others : actions_others_1hot,
                    self.obs_others : obs_others,
                    self.v_goal : goals_self,
                    self.v_goal_others : goals_others,
                    self.v_labels : agent_labels,
                    self.epsilon : epsilon}
            Q_res, probs_res = sess.run([self.Q, self.probs], feed_dict=feed)

        if self.n_agents > 1 and self.use_Q and self.use_V:
            feed = {self.action_taken : actions_1hot,
                    self.V_td_target : V_td_target,
                    self.V_evaluated : V_res,
                    self.epsilon : epsilon,
                    self.obs_others : obs_others,
                    self.v_obs : v_local,
                    self.v_goal : goals_self,
                    self.v_goal_others : goals_others,
                    self.v_labels : agent_labels,
                    self.Q_evaluated : Q_res,
                    self.probs_evaluated : probs_res}
        elif self.n_agents > 1 and self.use_Q and self.use_V==0:
            feed = {self.action_taken : actions_1hot,
                    self.epsilon : epsilon,
                    self.obs_others : obs_others,
                    self.v_obs : v_local,
                    self.v_goal : goals_self,
                    self.v_goal_others : goals_others,
                    self.v_labels : agent_labels,
                    self.Q_evaluated : Q_res,
                    self.probs_evaluated : probs_res}
        else:
            feed = {self.action_taken : actions_1hot,
                    self.epsilon : epsilon,
                    self.obs_others : obs_others,
                    self.v_obs : v_local,
                    self.v_goal : goals_self,
                    self.V_td_target : V_td_target,
                    self.V_evaluated : V_res}

        if summarize:
            summary, _ = sess.run([self.summary_op_policy, self.policy_op], feed_dict=feed)
            writer.add_summary(summary, idx_train)
        else:
            _ = sess.run(self.policy_op, feed_dict=feed)

        sess.run(self.list_update_target_ops)

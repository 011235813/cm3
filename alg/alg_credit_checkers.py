"""CM3 algorithm for Checkers environment.

Same algorithm as alg_credit.py, except that Checkers global state is defined
by three parts (s_env, s^n, s^{-n}) instead of just (s^n, s^{-n})
"""

import numpy as np
import tensorflow as tf

import sys
import networks


class Alg(object):

    def __init__(self, experiment, dimensions, stage=1, n_agents=1,
                 tau=0.01, lr_V=0.001, lr_Q=0.001,
                 lr_actor=0.0001, gamma=0.99, use_Q_credit=1,
                 use_V=0, nn={}):
        """
        Same as alg_credit. Checkers global state has two parts
        Inputs:
        experiment - string
        dimensions - dictionary containing tensor dimensions
                     (h,w,c) for tensor
                     l for 1D vector
        stage - 1: Q_global and actor, does not use Q_credit
                2: Q_global, actor and Q_credit
        tau - target variable update rate
        lr_V, lr_Q, lr_actor - learning rates for optimizer
        gamma - discount factor
        use_Q_credit - if 1, activates Q_credit network  for use in policy gradient
        use_V - if 1, uses V_n(s) as the baseline in the policy gradient (this is an ablation)
        nn : neural net architecture parameters
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
            # Dimension of agent's observation of itself

        self.l_action = dimensions['l_action']
        self.l_goal = dimensions['l_goal']

        self.n_agents = n_agents
        self.tau = tau
        self.lr_V = lr_V
        self.lr_Q = lr_Q
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.use_V = use_V
        self.use_Q_credit = use_Q_credit
        self.nn = nn

        self.agent_labels = np.eye(self.n_agents)
        self.actions = np.eye(self.l_action)

        # Initialize computational graph
        self.create_networks(stage)
        self.list_initialize_target_ops, self.list_update_target_ops = self.get_assign_target_ops(tf.trainable_variables())
        # Use Q_global when n_agents == 1 (the choice is arbitrary,
        # since both networks share the same Stage 1 architecture)
        self.create_Q_global_train_op()
        if self.n_agents > 1 and self.use_Q_credit:
            self.list_initialize_credit_ops = self.get_assign_global_to_credit_ops()
            self.create_Q_credit_train_op()
        elif self.n_agents > 1 and self.use_V:
            self.create_V_train_op()
        self.create_policy_gradient_op()

        # TF summaries
        self.create_summary()

    def create_networks(self, stage):

        # Placeholders
        self.state_env = tf.placeholder(tf.float32, [None, self.rows_state, self.columns_state, self.channels_state], 'state_env')
        self.v_state_one_agent = tf.placeholder(tf.float32, [None, self.l_state_one_agent], 'v_state_one_agent')
        self.v_state_m = tf.placeholder(tf.float32, [None, self.l_state_one_agent], 'v_state_m')
        self.v_state_other_agents = tf.placeholder(tf.float32, [None, self.l_state_other_agents], 'v_state_other_agents')
        self.v_goal = tf.placeholder(tf.float32, [None, self.l_goal], 'v_goal')
        self.v_goal_others = tf.placeholder(tf.float32, [None, (self.n_agents-1)*self.l_goal], 'v_goal_others')
        self.v_labels = tf.placeholder(tf.float32, [None, self.n_agents])

        self.action_others = tf.placeholder(tf.float32, [None, self.n_agents-1, self.l_action], 'action_others')
        self.action_one = tf.placeholder(tf.float32, [None, self.l_action], 'action_one')

        if self.experiment == "checkers":
            self.obs_self_t = tf.placeholder(tf.float32, [None, self.rows_obs, self.columns_obs, self.channels_obs], 'obs_self_t')
            self.obs_self_v = tf.placeholder(tf.float32, [None, self.l_obs_self], 'obs_self_v')
            self.obs_others = tf.placeholder(tf.float32, [None, self.l_obs_others], 'obs_others')
            self.actions_prev = tf.placeholder(tf.float32, [None, self.l_action], 'action_prev')

        # Actor network
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
        with tf.variable_scope("Policy_main"):
            if self.experiment == 'checkers':
                probs = networks.actor_checkers(self.actions_prev, self.obs_self_t, self.obs_self_v, self.obs_others, self.v_goal, f1=self.nn['A_conv_f'], k1=self.nn['A_conv_k'], n_h1=self.nn['A_n_h1'], n_h2=self.nn['A_n_h2'], n_actions=self.l_action, stage=stage)
        # probs is normalized
        self.probs = (1-self.epsilon) * probs + self.epsilon/float(self.l_action)
        self.action_samples = tf.multinomial(tf.log(self.probs), 1)
        with tf.variable_scope("Policy_target"):
            if self.experiment == 'checkers':
                probs_target = networks.actor_checkers(self.actions_prev, self.obs_self_t, self.obs_self_v, self.obs_others, self.v_goal, f1=self.nn['A_conv_f'], k1=self.nn['A_conv_k'], n_h1=self.nn['A_n_h1'], n_h2=self.nn['A_n_h2'], n_actions=self.l_action, stage=stage)
        self.action_samples_target = tf.multinomial(tf.log( (1-self.epsilon)*probs_target + self.epsilon/float(self.l_action) ), 1)

        # Q_n(s,\abf)
        with tf.variable_scope("Q_global_main"):
            if self.experiment == 'checkers':
                self.Q_global = networks.Q_global_checkers(self.state_env, self.v_state_one_agent, self.v_goal, self.action_one, self.v_state_other_agents, self.action_others, self.obs_self_t, self.obs_self_v, f1=self.nn['Q_conv_f'], k1=self.nn['Q_conv_k'], n_h1_1=self.nn['Q_n_h1_1'], n_h1_2=self.nn['Q_n_h1_2'], n_h2=self.nn['Q_n_h2'], stage=stage)
        with tf.variable_scope("Q_global_target"):
            if self.experiment == 'checkers':
                self.Q_global_target = networks.Q_global_checkers(self.state_env, self.v_state_one_agent, self.v_goal, self.action_one, self.v_state_other_agents, self.action_others, self.obs_self_t, self.obs_self_v, f1=self.nn['Q_conv_f'], k1=self.nn['Q_conv_k'], n_h1_1=self.nn['Q_n_h1_1'], n_h1_2=self.nn['Q_n_h1_2'], n_h2=self.nn['Q_n_h2'], stage=stage)

        # Q_n(s,a^m)
        if self.n_agents > 1 and self.use_Q_credit:
            with tf.variable_scope("Q_credit_main"):
                if self.experiment == 'checkers':
                    self.Q_credit = networks.Q_credit_checkers(self.state_env, self.v_state_one_agent, self.v_goal, self.action_one, self.v_state_m, self.v_state_other_agents, self.obs_self_t, self.obs_self_v, f1=self.nn['Q_conv_f'], k1=self.nn['Q_conv_k'], n_h1_1=self.nn['Q_n_h1_1'], n_h1_2=self.nn['Q_n_h1_2'], n_h2=self.nn['Q_n_h2'], stage=stage)
            with tf.variable_scope("Q_credit_target"):
                if self.experiment == 'checkers':
                    self.Q_credit_target = networks.Q_credit_checkers(self.state_env, self.v_state_one_agent, self.v_goal, self.action_one, self.v_state_m, self.v_state_other_agents, self.obs_self_t, self.obs_self_v, f1=self.nn['Q_conv_f'], k1=self.nn['Q_conv_k'], n_h1_1=self.nn['Q_n_h1_1'], n_h1_2=self.nn['Q_n_h1_2'], n_h2=self.nn['Q_n_h2'], stage=stage)

        # V(s,g^n), used as ablation at stage 2
        if self.n_agents > 1 and self.use_V:
            with tf.variable_scope("V_main"):
                self.V = networks.V_checkers_ablation(self.state_env, self.v_state_one_agent, self.v_goal, self.v_state_other_agents)
            with tf.variable_scope("V_target"):
                self.V_target = networks.V_checkers_ablation(self.state_env, self.v_state_one_agent, self.v_goal, self.v_state_other_agents)

    def get_assign_target_ops(self, list_vars):

        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []

        list_Q_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_global_main')
        map_name_Q_main = {v.name.split('main')[1] : v for v in list_Q_main}
        list_Q_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_global_target')
        map_name_Q_target = {v.name.split('target')[1] : v for v in list_Q_target}
        if len(list_Q_main) != len(list_Q_target):
            raise ValueError("get_initialize_target_ops : lengths of Q_main and Q_target do not match")
        for name, var in map_name_Q_main.items():
            # create op that assigns value of main variable to target variable of the same name
            list_initial_ops.append( map_name_Q_target[name].assign(var) )
        for name, var in map_name_Q_main.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_Q_target[name].assign( self.tau*var + (1-self.tau)*map_name_Q_target[name] ) )

        # For policy
        list_P_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_main')
        map_name_P_main = {v.name.split('main')[1] : v for v in list_P_main}
        list_P_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_target')
        map_name_P_target = {v.name.split('target')[1] : v for v in list_P_target}
        if len(list_P_main) != len(list_P_target):
            raise ValueError("get_initialize_target_ops : lengths of P_main and P_target do not match")
        for name, var in map_name_P_main.items():
            # op that assigns value of main variable to target variable
            list_initial_ops.append( map_name_P_target[name].assign(var) )
        for name, var in map_name_P_main.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_P_target[name].assign( self.tau*var + (1-self.tau)*map_name_P_target[name] ) )

        # Repeat for Q_credit
        if self.n_agents > 1 and self.use_Q_credit:
            list_Qc_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_credit_main')
            map_name_Qc_main = {v.name.split('main')[1] : v for v in list_Qc_main}
            list_Qc_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_credit_target')
            map_name_Qc_target = {v.name.split('target')[1] : v for v in list_Qc_target}
            if len(list_Qc_main) != len(list_Qc_target):
                raise ValueError("get_initialize_target_ops : lengths of Q_credit_main and Q_credit_target do not match")
            for name, var in map_name_Qc_main.items():
                # create op that assigns value of main variable to target variable of the same name
                list_initial_ops.append( map_name_Qc_target[name].assign(var) )
            for name, var in map_name_Qc_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_Qc_target[name].assign( self.tau*var + (1-self.tau)*map_name_Qc_target[name] ) )
        
        if self.n_agents > 1 and self.use_V:
            list_V_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
            map_name_V_main = {v.name.split('main')[1] : v for v in list_V_main}
            list_V_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_target')
            map_name_V_target = {v.name.split('target')[1] : v for v in list_V_target}
            if len(list_V_main) != len(list_V_target):
                raise ValueError("get_initialize_target_ops : lengths of V_main and V_target do not match")
            for name, var in map_name_V_main.items():
                # create op that assigns value of main variable to target variable of the same name
                list_initial_ops.append( map_name_V_target[name].assign(var) )
            for name, var in map_name_V_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_V_target[name].assign( self.tau*var + (1-self.tau)*map_name_V_target[name] ) )

        return list_initial_ops, list_update_ops

    def get_assign_global_to_credit_ops(self):
        """Get ops that assign value of Q_global network to Q_credit network.
        
        To be used at the start of Stage 2, after Q_global network has been initialized
        with the Stage 1 weights
        """
        list_update_ops = []
        list_Q_global = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_global_main')
        map_name_Q_global = {v.name.split('main')[1] : v for v in list_Q_global}
        list_Q_credit = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_credit_main')
        map_name_Q_credit = {v.name.split('main')[1] : v for v in list_Q_credit}
        if len(list_Q_global) != len(list_Q_credit):
            raise ValueError("get_assign_global_to_credit_ops : lengths of Q_global and Q_credit do not match")
        for name, var in map_name_Q_global.items():
            list_split = name.split('/')
            if ('stage-2' not in list_split):
                # op that assigns value of Q_global variable to Q_credit variable of the same name
                list_update_ops.append( map_name_Q_credit[name].assign(var) )

        return list_update_ops

    def run_actor(self, actions_prev, obs_others, obs_self_t, obs_self_v, goals, epsilon, sess):
        """Get actions for all agents as a batch

        Args:
            actions_prev: list of integers
            obs_others: list of vector or tensor describing other agents
            obs_self_t: list of observation grid centered on self
            obs_self_v: list of 1D observation vectors
            goals: [n_agents, n_lanes]
            epsilon: float
            sess: TF session
        """
        # convert to batch
        obs_others = np.array(obs_others)
        obs_self_t = np.array(obs_self_t)
        obs_self_v = np.array(obs_self_v)

        actions_prev_1hot = np.zeros([self.n_agents, self.l_action])
        actions_prev_1hot[np.arange(self.n_agents), actions_prev] = 1

        feed = {self.obs_others:obs_others, self.obs_self_t:obs_self_t,
                self.obs_self_v:obs_self_v, self.v_goal:goals,
                self.actions_prev: actions_prev_1hot, self.epsilon:epsilon}
        action_samples_res = sess.run(self.action_samples, feed_dict=feed)
        return np.reshape(action_samples_res, action_samples_res.shape[0])
    
    def run_actor_target(self, actions_prev, obs_others, obs_self_t, obs_self_v, goals, epsilon, sess):
        """Gets actions from the slowly-updating policy."""
        feed = {self.obs_others:obs_others, self.obs_self_t:obs_self_t,
                self.obs_self_v:obs_self_v, self.v_goal:goals,
                self.actions_prev: actions_prev, self.epsilon:epsilon}
        action_samples_res = sess.run(self.action_samples_target, feed_dict=feed)
        return np.reshape(action_samples_res, action_samples_res.shape[0])

    def create_Q_credit_train_op(self):
        # TD target calculated in train_step() using V_target
        self.Q_credit_td_target= tf.placeholder(tf.float32, [None], 'Q_credit_td_target')
        # Q_credit network has only one output
        self.loss_Q_credit = tf.reduce_mean(tf.square(self.Q_credit_td_target - tf.squeeze(self.Q_credit)))
        self.Q_credit_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.Q_credit_op = self.Q_credit_opt.minimize(self.loss_Q_credit)

    def create_Q_global_train_op(self):
        # TD target calculated in train_step() using Q_target
        self.Q_global_td_target = tf.placeholder(tf.float32, [None], 'Q_global_td_target')
        # Q_global network has only one output
        self.loss_Q_global = tf.reduce_mean(tf.square(self.Q_global_td_target - tf.squeeze(self.Q_global)))
        self.Q_global_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.Q_global_op = self.Q_global_opt.minimize(self.loss_Q_global)

    def create_V_train_op(self):
        self.V_td_target = tf.placeholder(tf.float32, [None], 'V_td_target')
        self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.V)))
        self.V_opt = tf.train.AdamOptimizer(self.lr_V)
        self.V_op = self.V_opt.minimize(self.loss_V)

    def create_policy_gradient_op(self):

        # batch of 1-hot action vectors
        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action], 'action_taken')
        # self.probs has shape [batch, l_action]
        log_probs = tf.log(tf.reduce_sum(tf.multiply(self.probs, self.action_taken), axis=1)+1e-15)

        # if stage==2, must be [batch*n_agents, n_agents], consecutive <n_agents> rows are same
        self.Q_actual = tf.placeholder(tf.float32, [None, self.n_agents], 'Q_actual')
        # First dim is n_agents^2 * batch;
        # If n_agents=1, first dim is batch; second dim is l_action
        # For <n_agents> > 1, the rows are Q_1(s,a^1),...,Q_N(s,a^1),Q_1(s,a^2),...,Q_N(s,a^2), ... , Q_1(s,a^N),...,Q_N(s,a^N)
        # where each row contains <l_action> Q-values, one for each possible action
        # Note that all Q networks have only one output, and the <l_action> dimension is due to evaluating all possible actions before feeding in feed_dict
        self.Q_cf = tf.placeholder(tf.float32, [None, self.l_action], 'Q_cf')
        # First dim is n_agents^2 * batch;
        # If n_agents=1, first dim is batch; second dim is l_action
        self.probs_evaluated = tf.placeholder(tf.float32, [None, self.l_action]) 
        
        if self.n_agents == 1:
            advantage2 = tf.reduce_sum(tf.multiply(self.Q_cf, self.probs_evaluated), axis=1)
            advantage = tf.subtract(tf.squeeze(self.Q_actual), advantage2)
            self.policy_loss = -tf.reduce_mean( tf.multiply(log_probs, advantage) )
        else:
            if self.use_Q_credit:
                # For the general case of any number of agents (covers n_agents==1)
                pi_mult_Q = tf.multiply( self.probs_evaluated, self.Q_cf )
                # [batch*n_agents, n_agents]
                counterfactuals = tf.reshape( tf.reduce_sum(pi_mult_Q, axis=1), [-1,self.n_agents] )
                # [batch*n_agents, n_agents], each block of nxn is matrix A_{mn} at one time step
                advantages = tf.subtract(self.Q_actual, counterfactuals) 
                # [batch, n_agents]
                sum_n_A = tf.reshape( tf.reduce_sum(advantages, axis=1), [-1, self.n_agents] )
            elif self.use_V:
                self.V_evaluated = tf.placeholder(tf.float32, [None, self.n_agents], 'V_evaluated')
                advantages = tf.subtract(self.Q_actual, self.V_evaluated)
                sum_n_A = tf.reshape( tf.reduce_sum(advantages, axis=1), [-1, self.n_agents] )
            else:
                sum_n_A = tf.reshape( tf.reduce_sum(self.Q_actual, axis=1), [-1, self.n_agents] )
            
            log_probs_shaped = tf.reshape(log_probs, [-1, self.n_agents]) # [batch, n_agents]
            m_terms = tf.multiply( log_probs_shaped, sum_n_A ) # [batch, n_agents]
            self.policy_loss = -tf.reduce_mean( tf.reduce_sum(m_terms, axis=1) )

        self.policy_opt = tf.train.AdamOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.minimize(self.policy_loss)

    def create_summary(self):

        summaries_Q_global = [tf.summary.scalar('loss_Q_global', self.loss_Q_global)]
        Q_global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_global_main')
        for v in Q_global_variables:
            summaries_Q_global.append( tf.summary.histogram(v.op.name, v) )
        grads = self.Q_global_opt.compute_gradients(self.loss_Q_global, Q_global_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q_global.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_Q_global = tf.summary.merge(summaries_Q_global)

        if self.n_agents > 1 and self.use_Q_credit:
            summaries_Q_credit = [tf.summary.scalar('loss_Q_credit', self.loss_Q_credit)]
            Q_credit_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_credit_main')
            for v in Q_credit_variables:
                summaries_Q_credit.append( tf.summary.histogram(v.op.name, v) )
            grads = self.Q_credit_opt.compute_gradients(self.loss_Q_credit, Q_credit_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries_Q_credit.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
            self.summary_op_Q_credit = tf.summary.merge(summaries_Q_credit)
        elif self.n_agents > 1 and self.use_V:
            summaries_V = [tf.summary.scalar('loss_V', self.loss_V)]
            V_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
            for v in V_variables:
                summaries_V.append( tf.summary.histogram(v.op.name, v) )
            grads = self.V_opt.compute_gradients(self.loss_V, V_variables)
            for grad, var in grads:
                if grad is not None:
                    summaries_V.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
            self.summary_op_V = tf.summary.merge(summaries_V)

        summaries_policy = [tf.summary.scalar('policy_loss', self.policy_loss)]
        policy_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_main')
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

        Returns
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

        # Try to free memory
        batch = None
    
        n_steps = state_agents.shape[0]
    
        # For all global quantities, for each time step,
        # duplicate values <n_agents> times for
        # batch processing of all agents
        state_env = np.repeat(state_env, self.n_agents, axis=0)
        state_env_next = np.repeat(state_env_next, self.n_agents, axis=0)
        done = np.repeat(done, self.n_agents, axis=0)

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
        """Main training step."""
        # Each agent for each time step is now a batch entry
        n_steps, state_env, state_agents, obs_others, obs_self_t, obs_self_v, actions_prev_1hot, actions_1hot, actions_others_1hot, reward, reward_local, state_env_next, state_agents_next, obs_others_next, obs_self_t_next, obs_self_v_next, done, goals = self.process_batch(batch)
        
        goals_self, goals_others = self.process_goals(goals, n_steps)
        v_global_one_agent, v_global_others, state = self.process_global_state(state_agents, n_steps)
        v_global_one_agent_next, v_global_others_next, state_next = self.process_global_state(state_agents_next, n_steps)

        # ------------ Train Q^{\pibf}_n(s,\abf) --------------- #
        # ------------------------------------------------------ #
        
        # Need a_{t+1} for all agents to evaluate TD target for Q
        # actions is single dimension [n_steps*n_agents]
        actions = self.run_actor_target(actions_1hot, obs_others_next, obs_self_t_next, obs_self_v_next, goals_self, epsilon, sess)
        # Now each row is one time step, containing action
        # indices for all agents
        actions_r = np.reshape(actions, [n_steps, self.n_agents])
        actions_self_next, actions_others_next = self.process_actions(n_steps, actions_r)

        # Get target Q_n(s',\abf')
        feed = {self.state_env : state_env_next,
                self.v_state_one_agent : v_global_one_agent_next,
                self.v_goal : goals_self,
                self.action_one : actions_self_next,
                self.v_state_other_agents : v_global_others_next,
                self.action_others : actions_others_next,
                self.obs_self_t : obs_self_t_next,
                self.obs_self_v : obs_self_v_next}
        Q_global_target_res = np.squeeze(sess.run(self.Q_global_target, feed_dict=feed))
        # if true, then 0, else 1
        done_multiplier = -(done - 1)
        Q_global_td_target = reward_local + self.gamma * Q_global_target_res * done_multiplier
        
        # Run optimizer
        feed = {self.Q_global_td_target : Q_global_td_target,
                self.state_env : state_env,
                self.v_state_one_agent : v_global_one_agent,
                self.v_goal : goals_self,
                self.action_one : actions_1hot,
                self.v_state_other_agents : v_global_others,
                self.action_others : actions_others_1hot,
                self.obs_self_t : obs_self_t,
                self.obs_self_v : obs_self_v}
        if summarize:
            # Get Q_global_res = Q_n(s,\abf) for use later in policy gradient
            summary, _, Q_global_res = sess.run([self.summary_op_Q_global, self.Q_global_op, self.Q_global], feed_dict=feed)
            writer.add_summary(summary, idx_train)
        else:
            _, Q_global_res = sess.run([self.Q_global_op, self.Q_global], feed_dict=feed)
        # For feeding into Q_actual
        Q_global_res_rep = np.repeat(np.reshape(Q_global_res, [n_steps, self.n_agents]), self.n_agents, axis=0)

        # ----------- Train Q^{\pibf}_n(s,a^m) -------------- #
        # --------------------------------------------------- #
        if self.n_agents > 1 and self.use_Q_credit:
            # When going down rows, n is the inner index and m is the outer index
            
            # Repeat the things indexed by n, so that the group of n_agents rows
            # at each time step is repeated n_agent times
            s_n_next_rep = np.reshape(np.repeat(np.reshape(v_global_one_agent_next, [-1, self.n_agents*self.l_state_one_agent]), self.n_agents, axis=0), [-1, self.l_state_one_agent])
            s_others_next_rep = np.reshape(np.repeat(np.reshape(v_global_others_next, [-1, self.n_agents*self.l_state_other_agents]), self.n_agents, axis=0), [-1, self.l_state_other_agents])
            goals_self_rep = np.reshape(np.repeat(np.reshape(goals_self, [-1, self.n_agents*self.l_goal]), self.n_agents, axis=0), [-1, self.l_goal])
            
            # Duplicate the things indexed by m so that consecutive n_agents rows are the same (so that when summing over n, they are the same)
            actions_self_next_rep = np.repeat(actions_self_next, self.n_agents, axis=0)
            s_m_next_rep = np.repeat(v_global_one_agent_next, self.n_agents, axis=0)
            obs_self_t_next_rep = np.repeat(obs_self_t_next, self.n_agents, axis=0)
            obs_self_v_next_rep = np.repeat(obs_self_v_next, self.n_agents, axis=0)
            
            # Duplicate shared state
            s_env_next_rep = np.repeat(state_env_next, self.n_agents, axis=0)

            # Get target Q_n(s',a'^m)
            feed = {self.state_env : s_env_next_rep,
                    self.v_state_one_agent : s_n_next_rep,
                    self.v_goal : goals_self_rep,
                    self.action_one : actions_self_next_rep,
                    self.v_state_m : s_m_next_rep,
                    self.v_state_other_agents : s_others_next_rep,
                    self.obs_self_t : obs_self_t_next_rep,
                    self.obs_self_v : obs_self_v_next_rep}
            Q_credit_target_res = np.squeeze(sess.run(self.Q_credit_target, feed_dict=feed))
            
            # reward_local is n_steps*n_agents
            # Duplicate into n_steps*n_agents*n_agents so that each time step is
            # [r^1,r^2,r^3,r^4,...,r^1,r^2,r^3,r^4] (assume N=4)
            # Now n_steps*n_agents*n_agents
            reward_local_rep = np.reshape(np.repeat(np.reshape(reward_local, [-1, self.n_agents*1]), self.n_agents, axis=0), [-1])
            done_rep = np.reshape(np.repeat(np.reshape(done, [-1, self.n_agents*1]), self.n_agents, axis=0), [-1])
            # if true, then 0, else 1
            done_multiplier = -(done_rep - 1)
            Q_credit_td_target = reward_local_rep + self.gamma * Q_credit_target_res*done_multiplier
            
            # Duplicate the things indexed by n
            s_n_rep = np.reshape(np.repeat(np.reshape(v_global_one_agent, [-1, self.n_agents*self.l_state_one_agent]), self.n_agents, axis=0), [-1, self.l_state_one_agent])
            s_others_rep = np.reshape(np.repeat(np.reshape(v_global_others, [-1, self.n_agents*self.l_state_other_agents]), self.n_agents, axis=0), [-1, self.l_state_other_agents])
            # Duplicate the things indexed by m
            actions_self_rep = np.repeat(actions_1hot, self.n_agents, axis=0)
            s_m_rep = np.repeat(v_global_one_agent, self.n_agents, axis=0)
            obs_self_t_rep = np.repeat(obs_self_t, self.n_agents, axis=0)
            obs_self_v_rep = np.repeat(obs_self_v, self.n_agents, axis=0)
            # Duplicate shared state
            s_env_rep = np.repeat(state_env, self.n_agents, axis=0)
            
            feed = {self.Q_credit_td_target : Q_credit_td_target,
                    self.state_env : s_env_rep,
                    self.v_state_one_agent : s_n_rep,
                    self.v_goal : goals_self_rep,
                    self.action_one : actions_self_rep,
                    self.v_state_m : s_m_rep,
                    self.v_state_other_agents : s_others_rep,
                    self.obs_self_t : obs_self_t_rep,
                    self.obs_self_v : obs_self_v_rep}
                    
            # Run optimizer for global critic
            if summarize:
                summary, _ = sess.run([self.summary_op_Q_credit, self.Q_credit_op], feed_dict=feed)
                writer.add_summary(summary, idx_train)
            else:
                _ = sess.run(self.Q_credit_op, feed_dict=feed)

        if self.n_agents > 1 and self.use_V:
            # Get target V_n(s')
            feed = {self.state_env : state_env_next,
                    self.v_state_one_agent : v_global_one_agent_next,
                    self.v_goal : goals_self,
                    self.v_state_other_agents : v_global_others_next}
            V_target_res = np.squeeze(sess.run(self.V_target, feed_dict=feed))
            # if true, then 0, else 1
            done_multiplier = -(done - 1)
            V_td_target = reward_local + self.gamma * V_target_res * done_multiplier
            
            # Run optimizer
            feed = {self.V_td_target : V_td_target,
                    self.state_env : state_env,
                    self.v_state_one_agent : v_global_one_agent,
                    self.v_goal : goals_self,
                    self.v_state_other_agents : v_global_others}
            if summarize:
                # Get V_res = V_n(s) for use later in policy gradient
                summary, _, V_res = sess.run([self.summary_op_V, self.V_op, self.V], feed_dict=feed)
                writer.add_summary(summary, idx_train)
            else:
                _, V_res = sess.run([self.V_op, self.V], feed_dict=feed)
            # For feeding into advantage function
            V_res_rep = np.repeat(np.reshape(V_res, [n_steps, self.n_agents]), self.n_agents, axis=0)
            
        # --------------- Train policy ------------- #
        # ------------------------------------------ #
        
        if self.n_agents == 1:
            # Stage 1
            feed = {self.obs_others : obs_others,
                    self.obs_self_t : obs_self_t,
                    self.obs_self_v : obs_self_v,
                    self.actions_prev : actions_prev_1hot,
                    self.v_goal : goals_self,
                    self.epsilon : epsilon}
            probs_res = sess.run(self.probs, feed_dict=feed)
            # compute Q(s,a=all possible,g)
            s_env_rep = np.repeat(state_env, self.l_action, axis=0)
            s_rep = np.repeat(v_global_one_agent, self.l_action, axis=0)
            goals_self_rep = np.repeat(goals_self, self.l_action, axis=0)
            obs_self_t_rep = np.repeat(obs_self_t, self.l_action, axis=0)
            obs_self_v_rep = np.repeat(obs_self_v, self.l_action, axis=0)
            actions_cf = np.tile(self.actions, (n_steps, 1))
            feed = {self.state_env : s_env_rep,
                    self.v_state_one_agent : s_rep,
                    self.v_goal : goals_self_rep,
                    self.action_one : actions_cf,
                    self.v_state_other_agents : v_global_others, # not used
                    self.action_others : actions_others_1hot,
                    self.obs_self_t : obs_self_t_rep,
                    self.obs_self_v : obs_self_v_rep}
            Q_cf = sess.run(self.Q_global, feed_dict=feed)
            Q_cf.shape = (n_steps, self.l_action)
        else:
            if self.use_Q_credit: 
                # Compute values for probs_evaluated
                feed = {self.obs_others : obs_others,
                        self.obs_self_t : obs_self_t,
                        self.obs_self_v : obs_self_v,
                        self.actions_prev : actions_prev_1hot,
                        self.v_goal : goals_self,
                        self.epsilon : epsilon}
                probs_res = sess.run(self.probs, feed_dict=feed)
                probs_res = np.repeat(probs_res, self.n_agents, axis=0)
                
                # Duplicate everything by number of possible actions, to prepare for batch
                # computation of counterfactuals
                s_n_rep = np.repeat(s_n_rep, self.l_action, axis=0)
                goals_self_rep = np.repeat(goals_self_rep, self.l_action, axis=0)
                s_m_rep = np.repeat(s_m_rep, self.l_action, axis=0)
                s_others_rep = np.repeat(s_others_rep, self.l_action, axis=0)
                s_env_rep = np.repeat(s_env_rep, self.l_action, axis=0)
                obs_self_t_rep = np.repeat(obs_self_t_rep, self.l_action, axis=0)
                obs_self_v_rep = np.repeat(obs_self_v_rep, self.l_action, axis=0)
                # Counterfactual actions
                actions_cf = np.tile(self.actions, (self.n_agents*self.n_agents*n_steps,1))
                # Compute Q_n(s,a^m) for all n and m and all actions a^m=i, for all steps
                feed = {self.state_env : s_env_rep,
                        self.v_state_one_agent : s_n_rep,
                        self.v_goal : goals_self_rep,
                        self.action_one : actions_cf,
                        self.v_state_m : s_m_rep,
                        self.v_state_other_agents : s_others_rep,
                        self.obs_self_t : obs_self_t_rep,
                        self.obs_self_v : obs_self_v_rep}
                Q_cf = sess.run(self.Q_credit, feed_dict=feed) # n_steps * n_agents^2 * l_action
                Q_cf.shape = (n_steps*self.n_agents*self.n_agents, self.l_action)
            else:
                # probs_res and Q_cf just need to have dimension [anything, l_action]
                # They are not used in this case
                probs_res = np.zeros([1,self.l_action])
                Q_cf = np.zeros([1,self.l_action])

        feed = {self.obs_others : obs_others,
                self.obs_self_t : obs_self_t,
                self.obs_self_v : obs_self_v,
                self.actions_prev : actions_prev_1hot,
                self.v_goal : goals_self,
                self.epsilon : epsilon,
                self.action_taken : actions_1hot,
                self.Q_actual : Q_global_res_rep,
                self.probs_evaluated : probs_res,
                self.Q_cf : Q_cf}
        if self.n_agents > 1 and self.use_V:
            feed[self.V_evaluated] = V_res_rep
            
        if summarize:
            summary, _ = sess.run([self.summary_op_policy, self.policy_op], feed_dict=feed)
            writer.add_summary(summary, idx_train)
        else:
            _ = sess.run(self.policy_op, feed_dict=feed)
        
        sess.run(self.list_update_target_ops)

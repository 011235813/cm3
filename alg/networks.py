import numpy as np
import tensorflow as tf


def fc2(t_input, n_hidden=64, n_outputs=9, nonlinearity1=tf.nn.relu,
        nonlinearity2=None, scope='fc2'):
    """Two layers."""
    with tf.variable_scope(scope, initializer=tf.initializers.truncated_normal(0, 0.01)):
        h = tf.layers.dense(inputs=t_input, units=n_hidden,
                            activation=nonlinearity1, use_bias=True,
                            name='h')
        
        out = tf.layers.dense(inputs=h, units=n_outputs,
                              activation=nonlinearity2,
                              use_bias=True, name='out')

    return out


def fc3(t_input, n_hidden1=64, n_hidden2=64, n_outputs=9,
        nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='fc3'):
    """Two hidden layer, one output layer."""
    with tf.variable_scope(scope, initializer=tf.initializers.truncated_normal(0, 0.01)):
        h1 = tf.layers.dense(inputs=t_input, units=n_hidden1,
                             activation=nonlinearity1, use_bias=True,
                             name='h1')

        h2 = tf.layers.dense(inputs=h1, units=n_hidden2,
                             activation=nonlinearity2, use_bias=True,
                             name='h2')
        
        out = tf.layers.dense(inputs=h2, units=n_outputs,
                              activation=None, use_bias=True,
                              name='out')

    return out


def convnet(t_input, f1=4, k1=[10,5], s1=[5,2],
            f2=8, k2=[6,3], s2=[3,2], scope='convnet'):
    """
    f1 - number of filters in first layer
    k1 - kernel size of first layer
    s1 - stride of first layer
    f2 - number of filters in second layer
    k2 - kernel size of second layer
    s2 - stride of second layer
    """
    if len(t_input.shape) != 4:
        raise ValueError("networks.py convnet : t_input must be 4D tensor")
    with tf.variable_scope(scope):
        conv1 = tf.contrib.layers.conv2d(inputs=t_input, num_outputs=f1,
                                         kernel_size=k1, stride=s1,
                                         padding="SAME",
                                         activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=f2,
                                         kernel_size=k2, stride=s2,
                                         padding="SAME",
                                         activation_fn=tf.nn.relu)
        # ignore the first dimension, corresponding to batch size
        size = np.prod(conv2.get_shape().as_list()[1:])
        conv2_flat = tf.reshape(conv2, [-1, size])

    return conv2_flat
        

def convnet_1(t_input, f1=4, k1=[5,3], s1=[1,1], scope='convnet_1'):
    if len(t_input.shape) != 4:
        raise ValueError("networks.py convnet_1 : t_input must be 4D tensor")
    with tf.variable_scope(scope):
        conv1 = tf.contrib.layers.conv2d(inputs=t_input, num_outputs=f1, kernel_size=k1, stride=s1, padding="SAME", activation_fn=tf.nn.relu)
        size = np.prod(conv1.get_shape().as_list()[1:])
        conv1_flat = tf.reshape(conv1, [-1, size])

    return conv1_flat


def get_variable(name, shape):

    return tf.get_variable(name, shape, tf.float32,
                           tf.initializers.truncated_normal(0,0.01))


def Q_global(v_global, action_others, v_goal, v_goal_others, agent_labels, v_obs,
             n_actions=5, stage=2, units=256):
    """Used by COMA for both SUMO and particle experiments."""
    n_others = action_others.get_shape().as_list()[1]
    actions_reshaped = tf.reshape(action_others, [-1, n_others*n_actions])
    concated = tf.concat([v_global, actions_reshaped, v_goal, v_goal_others, agent_labels, v_obs], axis=1)

    with tf.variable_scope("stage-2"):
        out = fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=n_actions, nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='Q_goals')

    return out


def Q_global_1output(s_n, g_n, a_n, s_others, a_others, n_h1_1=64, n_h1_2=128,
                     n_h2=64, n_actions=5, stage=1):
    """Used by CM3 for particle env.
    
    See notation in paper for meaning of inputs (replace underscore '_' with '^')
    """
    concated = tf.concat( [s_n, g_n, a_n], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_1, activation=tf.nn.relu, use_bias=True, name='Q_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        n_others = a_others.get_shape().as_list()[1]
        a_others_reshaped = tf.reshape(a_others, [-1, n_others*n_actions])
        with tf.variable_scope("stage-2"):
            concated2 = tf.concat( [s_others, a_others_reshaped], axis=1 )
            others = tf.layers.dense(inputs=concated2, units=n_h1_2, activation=tf.nn.relu, use_bias=True, name='Q_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='Q_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=False, name='Q_out')
    
    return out


def Q_global_sumo(s_n, g_n, a_n, s_others, a_others, g_others,
                  n_h1_1=256, n_h1_2=128, n_h2=256, n_actions=5,
                  activation=tf.nn.relu, stage=1, bias=True):
    """Used by CM3 for SUMO.
    
    See notation in paper for meaning of inputs (replace underscore '_' with '^')
    """
    concated = tf.concat( [s_n, g_n, a_n], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_1, activation=activation, use_bias=bias, name='Q_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        n_others = a_others.get_shape().as_list()[1]
        a_others_reshaped = tf.reshape(a_others, [-1, n_others*n_actions])
        with tf.variable_scope("stage-2"):
            # concated2 = tf.concat( [s_others, a_others_reshaped, g_others], axis=1 )
            concated2 = tf.concat( [s_others, a_others_reshaped], axis=1 ) # Original
            others = tf.layers.dense(inputs=concated2, units=n_h1_2, activation=activation, use_bias=bias, name='Q_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='Q_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=bias, name='Q_out')
    
    return out


def Q_global_checkers(s_grid, s_n, g_n, a_n, s_others, a_others, t_obs, v_obs,
                      f1=4, k1=[3,5], f2=6, k2=[3,3], n_h1_1=128, n_h1_2=32,
                      n_h2=32, n_actions=5, stage=1, bias=True):
    """Used by CM3 for Checkers experiment.
    
    See notation in paper for meaning of inputs (replace underscore '_' with '^')
    """
    conv = convnet_1(s_grid, f1=f1, k1=k1, s1=[1,1], scope='conv')
    conv_o = convnet_1(t_obs, f1=f2, k1=k2, s1=[1,1], scope='conv_o')
    concated = tf.concat( [conv, s_n, g_n, a_n, conv_o, v_obs], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_1, activation=tf.nn.relu, use_bias=bias, name='Q_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        n_others = a_others.get_shape().as_list()[1]
        a_others_reshaped = tf.reshape(a_others, [-1, n_others*n_actions])
        with tf.variable_scope("stage-2"):
            concated2 = tf.concat( [s_others, a_others_reshaped], axis=1 )
            others = tf.layers.dense(inputs=concated2, units=n_h1_2, activation=tf.nn.relu, use_bias=bias, name='Q_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='Q_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=bias, name='Q_out')
    
    return out


def Q_credit(s_n, g_n, a_m, s_m, s_others, n_h1_1=64, n_h1_2=128,
             n_h2=64, stage=2):
    """Used by CM3 for particle env.
    
    See notation in paper for meaning of inputs
    The subset of this network in stage=1 is the same as that in Q_global_1output, so that
    the weights of Q_global_1output trained in Stage 1 can be used for initialization.
    """
    concated = tf.concat( [s_n, g_n, a_m], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_1, activation=tf.nn.relu, use_bias=True, name='Q_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            concated2 = tf.concat( [s_m, s_others], axis=1 )
            others = tf.layers.dense(inputs=concated2, units=n_h1_2, activation=tf.nn.relu, use_bias=True, name='Q_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='Q_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=False, name='Q_out')
    
    return out


def Q_credit_sumo(s_n, g_n, a_m, s_m, s_others, g_others,
                  n_h1_1=256, n_h1_2=128, n_h2=256,
                  activation=tf.nn.relu, stage=2, bias=True):
    """Used by CM3 for SUMO.
    
    See notation in paper for meaning of inputs
    The subset of this network in stage=1 is the same as that in Q_global_1output, so that
    the weights of Q_global_1output trained in Stage 1 can be used for initialization.
    """
    concated = tf.concat( [s_n, g_n, a_m], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_1, activation=activation, use_bias=bias, name='Q_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            # concated2 = tf.concat( [s_m, s_others, g_others], axis=1 )
            concated2 = tf.concat( [s_m, s_others], axis=1 ) # original
            others = tf.layers.dense(inputs=concated2, units=n_h1_2, activation=activation, use_bias=bias, name='Q_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='Q_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=bias, name='Q_out')
    
    return out


def Q_credit_checkers(s_grid, s_n, g_n, a_m, s_m, s_others, t_obs, v_obs,
                      f1=4, k1=[3,5], f2=6, k2=[3,3],
                      n_h1_1=128, n_h1_2=32, n_h2=32, stage=2, bias=True):
    """Used by CM3 for Checkers experiment.
    
    See notation in paper for meaning of inputs
    The subset of this network in stage=1 is the same as that in Q_global_1output, so that
    the weights of Q_global_1output trained in Stage 1 can be used for initialization.
    """
    conv = convnet_1(s_grid, f1=f1, k1=k1, s1=[1,1], scope='conv')
    conv_o = convnet_1(t_obs, f1=f2, k1=k2, s1=[1,1], scope='conv_o')
    concated = tf.concat( [conv, s_n, g_n, a_m, conv_o, v_obs], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_1, activation=tf.nn.relu, use_bias=bias, name='Q_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            concated2 = tf.concat( [s_m, s_others], axis=1 )
            others = tf.layers.dense(inputs=concated2, units=n_h1_2, activation=tf.nn.relu, use_bias=bias, name='Q_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='Q_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=bias, name='Q_out')
    
    return out


def Q_coma(state, action_others, agent_labels, v_obs, n_actions=5, stage=2, units=256):
    """Used for testing pure COMA on environment without individual goals."""
    n_others = action_others.get_shape().as_list()[1]
    actions_reshaped = tf.reshape(action_others, [-1, n_others*n_actions])
    concated = tf.concat([state, actions_reshaped, agent_labels, v_obs], axis=1)

    with tf.variable_scope("stage-2"):
        out = fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=n_actions, nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='Q_coma')

    return out

def Q_coma_1output(state, actions, n_actions=5, stage=2, units=256):
    concated = tf.concat([state, actions], axis=1)
    out = fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=1, nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='Q_coma_1output')

    return out    


def Q_coma_checkers(s_grid, s_agents, a_others, g_n, g_others, agent_labels,
                    t_obs, v_obs, f1=4, k1=[3,5], f2=6, k2=[3,3],
                    n_actions=5, units=256):
    """Used by COMA for Checkers experiment."""
    n_others = a_others.get_shape().as_list()[1]
    a_reshaped = tf.reshape(a_others, [-1, n_others*n_actions])
    conv_s = convnet_1(s_grid, f1=f1, k1=k1, s1=[1,1], scope='conv_s')
    conv_o = convnet_1(t_obs, f1=f2, k1=k2, s1=[1,1], scope='conv_o')
    concated = tf.concat([conv_s, s_agents, a_reshaped, g_n, g_others, agent_labels, conv_o, v_obs], axis=1)

    with tf.variable_scope("stage-2"):
        out = fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=n_actions, nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='Q_coma_checkers')

    return out


def V_sumo_local(t_obs, v_obs, v_goal, n_h1_branch1=64, n_conv_reduced=64, n_h2=64, stage=1):
    """
    Used by IAC for SUMO experiment
    """
    concated = tf.concat( [v_obs, v_goal], axis=1 )
    branch_self = tf.layers.dense(inputs=concated, units=n_h1_branch1, activation=tf.nn.relu, use_bias=True, name='V_sumo_branch_self')
    W_branch_self_out = get_variable("W_branch_self_out", [n_h1_branch1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch_self, W_branch_self_out) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            conv_out = convnet_1(t_obs, f1=4, k1=[5,3], s1=[1,1], scope='V_sumo_conv')
            conv_reduced = tf.layers.dense(inputs=conv_out, units=n_conv_reduced, activation=tf.nn.relu, use_bias=True, name='V_sumo_conv_reduced')
            W_conv_out = get_variable('W_conv_out', [n_conv_reduced, n_h2])
        list_mult.append(tf.matmul(conv_reduced, W_conv_out))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='V_sumo_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=False, name='V_sumo_out')
    
    return out


def V_sumo_global(v_state_n, v_goal, v_state_others, v_goal_others,
                  n_h1_branch1=64, n_h1_branch2=64, n_h2=64, stage=1):
    """Global V(s,g^n)."""
    concated = tf.concat( [v_state_n, v_goal], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_branch1, activation=tf.nn.relu, use_bias=True, name='V_sumo_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_branch1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            concated2 = tf.concat( [v_state_others, v_goal_others], axis=1 )
            others = tf.layers.dense(inputs=concated2, units=n_h1_branch2, activation=tf.nn.relu, use_bias=True, name='V_sumo_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_branch2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='V_sumo_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=False, name='V_sumo_out')
    
    return out


def V_particle_local(v_obs_others, v_obs, v_goal, n_h1_branch1=64, n_h1_branch2=64, n_h2=64, stage=1):
    """Used by IAC for particle experiment."""
    concated = tf.concat( [v_obs, v_goal], axis=1 )
    branch_self = tf.layers.dense(inputs=concated, units=n_h1_branch1, activation=tf.nn.relu, use_bias=True, name='V_particle_branch_self')
    W_branch_self_out = get_variable("W_branch_self_out", [n_h1_branch1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch_self, W_branch_self_out) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            others = tf.layers.dense(inputs=v_obs_others, units=n_h1_branch2, activation=tf.nn.relu, use_bias=True, name='V_local_others')
            W_others_out = get_variable('W_others_out', [n_h1_branch2, n_h2])
        list_mult.append(tf.matmul(others, W_others_out))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='V_particle_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=False, name='V_particle_out')
    
    return out


def V_particle_global(v_state_n, v_goal, v_state_others, v_goal_others,
                      n_h1_branch1=64, n_h1_branch2=64, n_h2=64, stage=1):
    """V(s,g^n).
    v_state_n - state of agent n who is supposed to get to v_goal
    v_goal - g^n, the goal of agent n
    v_state_others - velocity and position of other agents (Stage 2 only)
    v_goal_others - position of other landmarks (Stage 2 only)
    """
    concated = tf.concat( [v_state_n, v_goal], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_branch1, activation=tf.nn.relu, use_bias=True, name='V_particle_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_branch1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            concated2 = tf.concat( [v_state_others, v_goal_others], axis=1 )
            others = tf.layers.dense(inputs=concated2, units=n_h1_branch2, activation=tf.nn.relu, use_bias=True, name='V_particle_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_branch2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='V_particle_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=False, name='V_particle_out')
    
    return out


def V_particle_ablation(s_n, g_n, s_others, n_h1=64, n_h2=64):
    """Used for ablation of CM3 in particle experiment."""
    concated = tf.concat( [s_n, g_n, s_others], axis=1 )
    with tf.variable_scope("stage-2"):
        h1 = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=True, name='V_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='V_h2')
        out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=False, name='V_out')
    return out


def V_checkers_local(t_obs_self, v_obs_self, v_obs_others, v_goal, f1=6, k1=[3,3],
                     n_h1_1=256, n_h1_2=32, n_h2=256, stage=1, bias=True):
    """Used by IAC for Checkers."""
    conv = convnet_1(t_obs_self, f1=f1, k1=k1, s1=[1,1], scope='conv')
    concated = tf.concat([conv, v_obs_self, v_goal], 1)
    branch_self = tf.layers.dense(inputs=concated, units=n_h1_1, activation=tf.nn.relu, use_bias=bias, name='branch_self')
    W_self_h2 = get_variable("W_self_h2", [n_h1_1, n_h2])
                            
    list_mult = []
    list_mult.append( tf.matmul(branch_self, W_self_h2) )    

    if stage > 1:
        with tf.variable_scope("stage-2"):
            branch_others = tf.layers.dense(inputs=v_obs_others, units=n_h1_2, activation=tf.nn.relu, use_bias=bias, name='branch_others')
            W_others_h2 = get_variable("W_others_h2", [n_h1_2, n_h2])
        list_mult.append( tf.matmul(branch_others, W_others_h2) )

    h2 = tf.nn.relu(tf.add_n(list_mult, name='V_checkers_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=bias, name='V_checkers_out')
    
    return out


def V_checkers_global(s_grid, s_n, g_n, s_others, f1=2, k1=[3,5],
                      n_h1_1=128, n_h1_2=32, n_h2=32, stage=1, bias=True):
    """Global value function for checkers."""
    conv = convnet_1(s_grid, f1=f1, k1=k1, s1=[1,1], scope='conv')
    concated = tf.concat( [conv, s_n, g_n], axis=1 )
    branch1 = tf.layers.dense(inputs=concated, units=n_h1_1, activation=tf.nn.relu, use_bias=bias, name='V_branch1')
    W_branch1_h2 = get_variable("W_branch1_h2", [n_h1_1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch1, W_branch1_h2) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            others = tf.layers.dense(inputs=s_others, units=n_h1_2, activation=tf.nn.relu, use_bias=bias, name='V_branch2')
            W_branch2_h2 = get_variable('W_branch2_h2', [n_h1_2, n_h2])
        list_mult.append(tf.matmul(others, W_branch2_h2))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='V_h2'))
    out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=bias, name='V_out')
    
    return out


def V_checkers_ablation(s_grid, s_n, g_n, s_others,
                        f1=4, k1=[3,5], n_h1=128, n_h2=32):
    """Used for ablation of CM3 in particle experiment."""
    with tf.variable_scope("stage-2"):
        conv = convnet_1(s_grid, f1=f1, k1=k1, s1=[1,1], scope='conv')
        concated = tf.concat( [conv, s_n, g_n, s_others], axis=1 )
        h1 = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=True, name='V_h1')
        h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='V_h2')
        out = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=False, name='V_out')
    return out


def actor_staged(t_obs, v_obs, v_goal, n_conv_reduced=64,
                 n_h1=32, n_h2=64, n_actions=9, stage=1):
    """Used by CM3, IAC and COMA for SUMO experiment.
    
    Uses CNN if stage > 1
    """
    branch1 = tf.layers.dense(inputs=v_obs, units=n_h1,
                              activation=tf.nn.relu, use_bias=True,
                              name='actor_branch1')

    branch2 = tf.layers.dense(inputs=v_goal, units=n_h1,
                              activation=tf.nn.relu, use_bias=True,
                              name='actor_branch2')

    concated = tf.concat([branch1, branch2], 1)

    W_concated_h2 = get_variable("W_concated_h2", [2*n_h1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(concated, W_concated_h2) )

    if stage > 1:
        # use different scope name so that
        # model restoration can ignore new variables
        # that did not exist in previous saved models
        with tf.variable_scope("stage-2"):
            # CNN to process local observation matrix
            conv_out = convnet_1(t_obs, f1=4, k1=[5,3], s1=[1,1], scope='actor_conv')
            # Reduce dimension using an fc layer
            conv_reduced = tf.layers.dense(inputs=conv_out, units=n_conv_reduced, activation=tf.nn.relu, use_bias=True, name='actor_conv_reduced')
            W_conv_h2 = get_variable('W_conv_h2', [n_conv_reduced, n_h2])
        list_mult.append( tf.matmul(conv_reduced, W_conv_h2) )

    b = tf.get_variable('b', [n_h2])
    h2 = tf.nn.relu(tf.nn.bias_add(tf.add_n(list_mult), b))

    # Output layer
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='actor_out')

    probs = tf.nn.softmax(out, name='actor_softmax')

    return probs


def actor_particle(v_obs_others, v_obs, v_goal, n_actions=5,
                   n_h1_self=64, n_h1_others=64, n_h2=64, stage=1):
    """Used by CM3, IAC and COMA for particle experiment."""
    concated = tf.concat([v_obs, v_goal], axis=1)
    branch_self = tf.layers.dense(inputs=concated, units=n_h1_self, activation=tf.nn.relu, use_bias=True, name='actor_branch_self')
    W_branch_self_h2 = get_variable("W_branch_self_h2", [n_h1_self, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch_self, W_branch_self_h2) )

    if stage > 1:
        with tf.variable_scope("stage-2"):
            others = tf.layers.dense(inputs=v_obs_others, units=n_h1_others, activation=tf.nn.relu, use_bias=True, name='actor_others')
            W_others_h2 = get_variable('W_others_h2', [n_h1_others, n_h2])
        list_mult.append( tf.matmul(others, W_others_h2) )

    b = tf.get_variable('b', [n_h2])
    h2 = tf.nn.relu(tf.nn.bias_add(tf.add_n(list_mult), b))

    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_softmax')
    return probs


def actor_particle_coma(v_obs, n_actions=5, n_h1=64, n_h2=64):
    """Used for testing pure COMA on particle env without individual goals."""
    out = fc3(v_obs, n_hidden1=n_h1, n_hidden2=n_h2, n_outputs=n_actions, nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='actor_coma')

    probs = tf.nn.softmax(out, name='actor_softmax')
    return probs


def actor_checkers(a_prev, t_obs_self, v_obs_self, v_obs_others, v_goal, f1=3, k1=[3,3],
                   n_h1=64, n_h2=64, n_actions=5, stage=1, bias=True):
    """Used by CM3, IAC and COMA for Checkers experiment."""
    conv = convnet_1(t_obs_self, f1=f1, k1=k1, s1=[1,1], scope='conv')
    conv_linear = tf.layers.dense(inputs=conv, units=32, activation=tf.nn.relu, use_bias=bias, name='conv_linear')
    # concated = tf.concat([conv, v_obs_self, a_prev, v_goal], 1)
    concated = tf.concat([conv_linear, v_obs_self, a_prev, v_goal], 1)
    branch_self = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=bias, name='branch_self')
    W_self_h2 = get_variable("W_self_h2", [n_h1, n_h2])
                            
    list_mult = []
    list_mult.append( tf.matmul(branch_self, W_self_h2) )    

    if stage > 1:
        # use different scope name so that
        # model restoration can ignore new variables
        # that did not exist in previous saved models
        with tf.variable_scope("stage-2"):
            branch_others = tf.layers.dense(inputs=v_obs_others, units=n_h1, activation=tf.nn.relu, use_bias=bias, name='branch_others')
            W_others_h2 = get_variable("W_others_h2", [n_h1, n_h2])
        list_mult.append( tf.matmul(branch_others, W_others_h2) )

    b = tf.get_variable('b', [n_h2])
    h2 = tf.nn.relu(tf.nn.bias_add(tf.add_n(list_mult), b))

    # Output layer
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=bias, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_softmax')

    return probs


def Qmix_single_particle(o_others, o_self, goal, n_actions=5):

    concated = tf.concat( [o_others, o_self, goal], axis=1 )

    h = tf.layers.dense(inputs=concated, units=64, activation=tf.nn.relu,
                        use_bias=True, name='h')

    h = tf.layers.dense(inputs=h, units=64, activation=tf.nn.relu, # HERE
                        use_bias=True, name='h2')

    out = tf.layers.dense(inputs=h, units=n_actions, activation=None,
                          use_bias=True, name='out')

    return out


def Qmix_single_sumo(o_others, o_self, goal, n_actions=5, n_h1_branch1=64,
                     n_conv_reduced=64, n_h2=64):
    concated = tf.concat( [o_self, goal], axis=1 )
    branch_self = tf.layers.dense(inputs=concated, units=n_h1_branch1, activation=tf.nn.relu, use_bias=True, name='Qmix_single_self')
    W_branch_self_out = get_variable("W_branch_self_out", [n_h1_branch1, n_h2])

    list_mult = []
    list_mult.append( tf.matmul(branch_self, W_branch_self_out) )

    conv_out = convnet_1(o_others, f1=4, k1=[5,3], s1=[1,1], scope='Qmix_single_conv')
    conv_reduced = tf.layers.dense(inputs=conv_out, units=n_conv_reduced, activation=tf.nn.relu, use_bias=True, name='Qmix_single_conv_reduced')
    W_conv_out = get_variable('W_conv_out', [n_conv_reduced, n_h2])
    list_mult.append(tf.matmul(conv_reduced, W_conv_out))

    h2 = tf.nn.relu(tf.add_n(list_mult, name='Qmix_single_h2'))
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='Qmix_single_out')

    return out


def Qmix_single_checkers(a_prev, t_obs_self, v_obs_self, v_obs_others, v_goal, f1=3, k1=[3,3],
                         n_h1=64, n_h2=64, n_actions=5, stage=1, bias=True):
    conv = convnet_1(t_obs_self, f1=f1, k1=k1, s1=[1,1], scope='conv')
    conv_linear = tf.layers.dense(inputs=conv, units=32, activation=tf.nn.relu, use_bias=bias, name='conv_linear')
    concated = tf.concat([conv_linear, v_obs_self, a_prev, v_goal], 1)
    branch_self = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=bias, name='branch_self')
    W_self_h2 = get_variable("W_self_h2", [n_h1, n_h2])
                            
    list_mult = []
    list_mult.append( tf.matmul(branch_self, W_self_h2) )    

    branch_others = tf.layers.dense(inputs=v_obs_others, units=n_h1, activation=tf.nn.relu, use_bias=bias, name='branch_others')
    W_others_h2 = get_variable("W_others_h2", [n_h1, n_h2])
    list_mult.append( tf.matmul(branch_others, W_others_h2) )

    b = tf.get_variable('b', [n_h2])
    h2 = tf.nn.relu(tf.nn.bias_add(tf.add_n(list_mult), b))

    # Output layer
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=bias, name='Qmix_single_out')
    return out


def Qmix_mixer(agent_qs, state, goals_all, state_dim, goal_dim, n_agents):
    """
    Args:
        agent_qs: shape [batch, n_agents]
        state: shape [batch, state_dim]
        goals_all: shape [batch, n_agents*goal_dim]
        state_dim: int
        goal_dim: int
        n_agents: int
    """
    embed_dim = 64 # orig: 64
    state_goals_dim = state_dim + n_agents*goal_dim

    agent_qs_reshaped = tf.reshape(agent_qs, [-1, 1, n_agents])
    state_goals = tf.concat([state, goals_all], axis=1)

    # embed_dim * n_agents because result will be reshaped into matrix
    hyper_w_1 = get_variable('hyper_w_1', [state_goals_dim, embed_dim*n_agents]) 
    hyper_w_final = get_variable('hyper_w_final', [state_goals_dim, embed_dim])

    hyper_b_1 = tf.get_variable('hyper_b_1', [state_goals_dim, embed_dim])

    hyper_b_final_l1 = tf.layers.dense(inputs=state_goals, units=embed_dim, activation=tf.nn.relu,
                                       use_bias=False, name='hyper_b_final_l1')
    hyper_b_final = tf.layers.dense(inputs=hyper_b_final_l1, units=1, activation=None,
                                    use_bias=False, name='hyper_b_final')

    # First layer
    w1 = tf.abs(tf.matmul(state_goals, hyper_w_1))
    b1 = tf.matmul(state_goals, hyper_b_1)
    w1_reshaped = tf.reshape(w1, [-1, n_agents, embed_dim]) # reshape into batch of matrices
    b1_reshaped = tf.reshape(b1, [-1, 1, embed_dim])
    # [batch, 1, embed_dim]
    hidden = tf.nn.elu(tf.matmul(agent_qs_reshaped, w1_reshaped) + b1_reshaped)
    
    # Second layer
    w_final = tf.abs(tf.matmul(state_goals, hyper_w_final))
    w_final_reshaped = tf.reshape(w_final, [-1, embed_dim, 1]) # reshape into batch of matrices
    b_final_reshaped = tf.reshape(hyper_b_final, [-1, 1, 1])

    # [batch, 1, 1]
    y = tf.matmul(hidden, w_final_reshaped) + b_final_reshaped

    q_tot = tf.reshape(y, [-1, 1])

    return q_tot


def Qmix_mixer_checkers(agent_qs, state_env, state, goals_all, state_dim, goal_dim, n_agents,
                        f1=4, k1=[3,5]):
    """
    Args:
        agent_qs: shape [batch, n_agents]
        state_env: shape [batch, rows, cols, channels]
        state: shape [batch, state_dim]
        goals_all: shape [batch, n_agents*goal_dim]
    """
    conv = convnet_1(state_env, f1=f1, k1=k1, s1=[1,1], scope='conv')

    embed_dim = 128
    state_goals_dim = state_dim + n_agents*goal_dim + conv.get_shape().as_list()[1]

    agent_qs_reshaped = tf.reshape(agent_qs, [-1, 1, n_agents])
    state_goals = tf.concat([conv, state, goals_all], axis=1)

    # embed_dim * n_agents because result will be reshaped into matrix
    hyper_w_1 = get_variable('hyper_w_1', [state_goals_dim, embed_dim*n_agents]) 
    hyper_w_final = get_variable('hyper_w_final', [state_goals_dim, embed_dim])

    hyper_b_1 = tf.get_variable('hyper_b_1', [state_goals_dim, embed_dim])

    hyper_b_final_l1 = tf.layers.dense(inputs=state_goals, units=embed_dim, activation=tf.nn.relu,
                                       use_bias=False, name='hyper_b_final_l1')
    hyper_b_final = tf.layers.dense(inputs=hyper_b_final_l1, units=1, activation=None,
                                    use_bias=False, name='hyper_b_final')

    # First layer
    w1 = tf.abs(tf.matmul(state_goals, hyper_w_1))
    b1 = tf.matmul(state_goals, hyper_b_1)
    w1_reshaped = tf.reshape(w1, [-1, n_agents, embed_dim]) # reshape into batch of matrices
    b1_reshaped = tf.reshape(b1, [-1, 1, embed_dim])
    # [batch, 1, embed_dim]
    hidden = tf.nn.elu(tf.matmul(agent_qs_reshaped, w1_reshaped) + b1_reshaped)
    
    # Second layer
    w_final = tf.abs(tf.matmul(state_goals, hyper_w_final))
    w_final_reshaped = tf.reshape(w_final, [-1, embed_dim, 1]) # reshape into batch of matrices
    b_final_reshaped = tf.reshape(hyper_b_final, [-1, 1, 1])

    # [batch, 1, 1]
    y = tf.matmul(hidden, w_final_reshaped) + b_final_reshaped

    q_tot = tf.reshape(y, [-1, 1])

    return q_tot

import numpy as np
import matplotlib.pyplot as plt
import random

from sumosimhri import car
from . import observation

class EgoCar(car.Car):

  def __init__(self, sim, goal_lane, goal_pos, route, vel,
               lane, pos, cid='ego_car', vtype='ego_car_type',
               depart_time=-1, safety=False):
               
    """__init__(obj, int, float, string, float, 
    int, float, string)
    
    sim - simulator object
    goal_lane - desired lane to be at end of episode
    goal_pos - position of goal location with respect to 
               start of final edge
    route - 'route_ramp' : edge_ramp -> edge2 -> edge3
            'route_straight' : edge1 -> edge2 -> edge3 
    vel - initial velocity
    lane - initial lane
    pos - initial position on lane
    cid - car unique identifier
    vtype - vehicle type (see .rou.xml file for options)
    depart_time - -1 for 'triggered', else time in seconds
    safety - if True, activates collision avoidance

    Initialize an ego car in sumo
    """
    super(EgoCar, self).__init__(sim, vel, lane, pos, cid)
    self.goal_lane = goal_lane
    self.goal_pos = goal_pos
    # absolute sublane number of center of goal lane
    self.goal_abs_sublane = int((0.5+self.goal_lane) * self.sim.sublanes_per_lane)
    # Car encounters merge situation if it started from
    # lane 0 of straight road and goal lane is 0 on edge 3,
    # or if it started from entrance road

    self.delta_sublane = self.sim.sublanes_per_lane * (self.goal_lane - self.lane)
    self.route = route
    self.traci.vehicle.addLegacy(vehID=self.id, routeID=self.route,
                                 depart=depart_time, pos=pos,
                                 speed=self.vel, lane=self.lane,
                                 typeID=vtype)
    # All speed checks off
    self.traci.vehicle.setSpeedMode(self.id, 0b00000)
    # No checks at all
    self.traci.vehicle.setLaneChangeMode(self.id, 0b0000000000)
    # List of IDs of edges along vehicle's route
    self.list_edges = self.traci.vehicle.getRoute(self.id)
    self.list_edge_length = [self.traci.lane.getLength(edge+'_0') for edge in self.list_edges]
    self.total_length = sum(self.list_edge_length)
    self.current_edge_ID = -1
    self.final_edge_num_lanes = self.traci.edge.getLaneNumber(self.list_edges[-1])
    # One-hot vector, see update_current_edge_type()
    self.current_edge_type = np.zeros(6)
    # Vehicle speed limits
    self.vmax = self.traci.vehicle.getMaxSpeed(self.id)
    self.vmin = 10
    self.num_actions = 5 # 9
    # Actions indices. LEFT and RIGHT cause sublane change,
    # not a compete lane change
    self.NOOP, self.ACC, self.DEC, self.LEFT, self.RIGHT = range(self.num_actions)
    # if vtype == 'ego_car_type':
    if 'ego_car_type' in vtype:
      self.acc_val = 2.5 # acceleration
      self.dec_val = 2.5 # deceleration
    elif vtype == 'ego_truck_type':
      self.acc_val = 1.0
      self.dec_val = 1.0
    # Local observation object
    self.observation = observation.Observation(sim, self.id, left=4, right=4, front=15, back=15, res_forward=2.5)
    # Maximum number of steps allowed before termination
    # Equivalent to requiring minimum average speed of 20m/s
    # over entire route
    self.max_step = round((self.total_length/25.0)/self.dt)
    self.step = 0
    # minimum seconds between lane change operations
    self.max_lc_step = 0 # round(1/self.dt)
    self.lc_step_counter = 0
    self.allow_lc = True # False
    self.collision = False
    self.terminal = False

    # time-to-collision tolerance threshold
    self.ttc_thres = 2.0 # original 10.0
    self.safety_dist = 5.0
    self.safety_system = safety # True
    self.length = self.traci.vehicle.getLength(self.id)

    self.removed = False

    # Distance from next road junctions
    # Will be part of the observation vector
    self.dist_to_next_junction = -1

    # Distance to junctions along route
    # Consider goal position of the final edge as a junction 
    self.list_dist_to_junctions = [-1] * len(self.list_edges)

    # Distance to goal_pos and goal_lane on final edge
    self.dist_to_goal = self.list_dist_to_junctions[-1]


  def __str__(self):
    """ print overload """
    return "<%s: d2g=%.2f, vel=%.2f, lane=%d, sublane=%d>" % (self.id, self.dist_to_goal, self.vel, self.lane, self.sublane_idx)


  def update_current_edge_type(self):
    """
    Indices of one-hot vector
    0 - single-lane entrance road
    1 - lane 0 on edge 2 that must merge
    2 - lane 0 on edge 1
    3 - lane 1 of edge 2 that receives the merge
    4 - lane 0 of edge 3, post merge
    5 - all other lanes
    """
    self.current_edge_type = np.zeros(6)
    if self.current_edge_ID == "edge_ramp":
      self.current_edge_type[0] = 1
    elif self.current_edge_ID == "edge2" and self.lane == 0:
      self.current_edge_type[1] = 1
    elif self.current_edge_ID == "edge1" and self.lane == 0:
      self.current_edge_type[2] = 1
    elif self.current_edge_ID == "edge2" and self.lane == 1:
      self.current_edge_type[3] = 1
    elif self.current_edge_ID == "edge3" and self.lane == 0:
      self.current_edge_type[4] = 1
    else:
      self.current_edge_type[5] = 1


  def update_state(self, map_car_values, action):
    """
    Updates local state for ego car

    map_car_values - map from each car ID to tuple of values
                     to be used for constructing this car's 
                     local observation
    action - taken by this car
    """

    use_for_global_reward = False
    if self.id in self.sim.list_arrived:
      # This means the car has reached end of
      # final edge and was removed by simulator.
      self.terminal = True
      self.removed = True
      reward = 0
    elif self.id in self.sim.set_colliding:
      reward = -1 # -10
      self.terminal = True
      self.collision = True
    else:
      # update basic state
      super(EgoCar, self).updateState()

      # Index of edge along vehicle's route
      route_index = self.traci.vehicle.getRouteIndex(self.id)
      self.current_edge_ID = self.list_edges[route_index]
      # Number of lanes of current edge
      self.num_lanes = self.traci.edge.getLaneNumber(self.current_edge_ID)
      edge_length = self.list_edge_length[route_index]

      # absolute distance from junctions
      for idx in range(len(self.list_edges)-1):
        self.list_dist_to_junctions[idx] = self.sim.traci.vehicle.getDrivingDistance(self.id, self.list_edges[idx+1], 0)
      # last element is distance to goal position
      self.list_dist_to_junctions[-1] = self.sim.traci.vehicle.getDrivingDistance(self.id, self.list_edges[-1], self.goal_pos)
      # normalized distance to next junction
      self.dist_to_next_junction = self.list_dist_to_junctions[route_index] / float(edge_length)
      # normalized distance to goal
      self.dist_to_goal = self.list_dist_to_junctions[-1] / self.total_length
      # difference from goal lane
      self.delta_sublane = self.goal_abs_sublane - self.abs_sublane

      # allow lane change after time max_lc_step has passed
      if not self.allow_lc:
        self.lc_step_counter += 1
        if self.lc_step_counter >= self.max_lc_step:
          self.lc_step_counter = 0
          self.allow_lc = True

      # local observation update
      car_on_left, car_on_right = self.observation.update(self.x, self.y, self.abs_sublane, self.num_lanes, self.vel, map_car_values)
      # self.update_current_edge_type()

      # Assign instantaneous reward
      if self.dist_to_goal <= 0 and self.delta_sublane == 0:
        reward = 10
        self.terminal = True
        use_for_global_reward = True
      elif self.step >= self.max_step:
        reward = -10
        self.terminal = True
      elif self.dist_to_goal <= 0:
        reward = 10.0 * (1-abs(self.delta_sublane)/float(self.num_lanes*self.sim.sublanes_per_lane))
        self.terminal = True
        use_for_global_reward = True
      elif car_on_left and action == 3:
        reward = -1 # -10.0
        self.terminal = True
        self.collision = True
      elif car_on_right and action == 4:
        reward = -1 # -10.0
        self.terminal = True
        self.collision = True
      else:
        reward = 0

      if self.vel >= 35.7:
        reward -= 0.1

      self.step += 1

    return self.terminal, reward, use_for_global_reward


  def execute_action(self, action):
    """
    action - int between 0 and 8 (inclusive)
    Apply action to the ego car for next simulation step
    """
    # parse action
    acc = 0
    if action == self.ACC:
      acc = self.acc_val
    elif action == self.DEC:
      acc = -self.dec_val
    elif action == self.LEFT and self.allow_lc:
      self.traci.vehicle.changeSublane(self.id, 0.8)
      self.traci.vehicle.setSignals(self.id, 0)
    elif action == self.RIGHT and self.allow_lc:
      self.traci.vehicle.changeSublane(self.id, -0.8)
      self.traci.vehicle.setSignals(self.id, 0)      
    else: # self.NOOP
      self.traci.vehicle.setSignals(self.id, 0)

    # finite difference to change velocity
    self.vel += self.dt*acc
    # bound velocity
    if self.vel < 0.0:
      self.vel = 0.0
    elif self.vel > self.vmax:
      self.vel = self.vmax
    # apply velocity
    self.traci.vehicle.slowDown(self.id, self.vel, self.dt)


  def get_feasible_actions(self, map_car_values):
    """
    map_car_values - map from car ID to (x,y,speed,length)
    """
    f_action = np.ones(self.num_actions)
    
    # min, max velocity
    if self.vel >= self.vmax:
      f_action[self.ACC] = np.nan
    elif self.vel <= self.vmin:
      f_action[self.DEC] = np.nan
    
    # lane changing
    if self.lane >= self.num_lanes-1 and self.sublane_idx >= 1:
      f_action[self.LEFT] = np.nan
      # f_action[self.LEFT_L_ON] = np.nan
    elif self.lane <= 0 and self.sublane_idx <= -1:
      f_action[self.RIGHT] = np.nan
      # f_action[self.RIGHT_R_ON] = np.nan
    
    if self.safety_system:
      for cid, tup in map_car_values.items():
        if cid == self.id:
          continue
        x, y, speed, length = tup

        if self.x > x:
          dist = self.x - x - self.length
        else:
          dist = x - self.x - length

        if x > self.x and speed < self.vel:
          ttc = dist / abs(self.vel - speed)
          if ttc <= self.ttc_thres and y < self.y + 1.8 and y > self.y - 1.8:
            f_action[self.NOOP] = np.nan
            f_action[self.ACC] = np.nan

    return f_action


  def getBaselineAction(self):
    """
    Not used
    """
    action = np.zeros(self.num_actions)
    action_index = 0

    f_action = self.get_feasible_actions()
    actions = np.multiply(range(self.num_actions), f_action)
    if self.lane == self.goal_lane:
      if self.ACC in actions:
        action_index = self.ACC
      elif self.DEC in actions:
        action_index = self.DEC
      elif self.NOOP in actions:
        action_index = self.NOOP
    else:
      if self.RIGHT in actions:
        action_index = self.RIGHT
      elif self.DEC in actions:
        action_index = self.DEC
    action[action_index] = 1

    return action


  def getRandomAction(self):
    """
    Not used
    """
    action = np.zeros(self.num_actions)
    action_index = 0

    f_action = self.get_feasible_actions()
    actions = np.multiply(range(self.num_actions), f_action)
    action_index = int(random.choice(actions[~np.isnan(actions)]))
    action[action_index] = 1

    return action


  def wait_until_ready(self):
    """ step the simulator until ego car appears in it """
    while True:
      self.sim.step()
      super(EgoCar, self).updateState()
      if self.lane >= 0:
        return


  def remove(self):
    """ Remove ego car from simulation """
    self.traci.vehicle.remove(self.id)
    self.removed = True

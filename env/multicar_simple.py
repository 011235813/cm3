import numpy as np
from operator import itemgetter

from . import egocar_simple

class Multicar(object):
    """
    This class is the environment of the multi-agent POMDP.
    It instantiates all ego-cars, handles updates of all local
    observations, handles the construction of the global state,
    determines the global reward, and reads individual rewards.
    These state information are given to the RL training module,
    and the step() function receives the next action to be taken by
    each agent.
    """
    def __init__(self, sim, num_ego, goal_lanes, goal_positions,
                 routes, speeds, lanes, init_positions,
                 list_ids, vtypes, depart_times, total_length=200,
                 total_width=12.8, res_forward=2.5, res_side=0.8,
                 follow_threshold=15, v_threshold=29.05, safety=False):
        """
        Must call reset() to instantiate all ego-cars

        sim - the global simulator object, most likely 
              instantiated by the RL training script
        num_ego - number of controlled vehicles
        goal_lanes - list of goal lanes for each ego-car (int)
        goal_positions - list of goal positions w.r.t. final edge (float)
        routes - list of routes for each ego-car (string)
        speeds - list of initial speeds for each ego-car (float)
        lanes - list of initial lane for each ego-car (int)
        init_positions - list of initial positions (float)
        list_ids - list of ID of ego-cars (string)
        vtypes - list of vtypes ('ego_car_type' or 'ego_truck_type')
        total_length and total_width - tightest bounding box dimension
        res_forward and res_side - grid resolution
        follow_threshold - negative reward if car following distance is less than this (meters)
        v_threshold - negative reward if average speed along a section is less than this (m/s); also used for normalization.
        safety - if True, avoid collisions
        """
        self.sim = sim
        self.num_ego = num_ego
        if num_ego != len(routes) or num_ego != len(speeds) or num_ego != len(lanes) or num_ego != len(vtypes):
            raise ValueError("Error in multicar.py : number of agent cars does not match lengths of input lists")
        self.goal_lanes = goal_lanes
        self.goal_positions = goal_positions
        self.routes = routes
        self.speeds = speeds
        self.lanes = lanes
        self.init_positions = init_positions
        self.list_ids = list_ids
        self.vtypes = vtypes
        self.depart_times = depart_times
        self.total_length = total_length
        self.total_width = total_width
        self.res_forward = res_forward
        self.res_side = res_side
        self.follow_threshold = follow_threshold
        self.v_threshold= v_threshold
        self.safety = safety

        self.n_rows = int(self.total_length / self.res_forward)
        self.n_cols = int(self.total_width / self.res_side)

        self.set_lane0_prev = set() # edge2 lane 0 at prev step
        self.set_region1_0_prev = set()
        self.set_region2_0_prev = set()
        # Distance covered at 65mph within one time step
        # Used to define measurement zone length
        self.zone_delta = 29.05 * self.sim.dt
        self.zone1 = self.total_length / 3.0
        self.zone2 = self.total_length * 2.0 / 3.0


    def check_actions(self, actions):
        """
        actions - vector of integers (0-8) chosen by policy for
        all agents

        Returns same vector, with invalid actions replaced
        """
        map_car_values = {}
        for cid in self.sim.set_present:
            x, y = self.sim.traci.vehicle.getPosition(cid)
            speed = self.sim.traci.vehicle.getSpeed(cid)
            length = self.sim.traci.vehicle.getLength(cid)
            map_car_values[cid] = (x,y,speed,length)

        for idx, car in enumerate(self.list_ego_cars):
            if car.removed:
                continue
            # get feasible actions for this car
            feas_actions = car.get_feasible_actions(map_car_values)
            if np.isnan(feas_actions[actions[idx]]):
                # policy's action is not feasible, so choose
                # the first feasible action, starting with no-op
                alternative = np.where(feas_actions==1)[0][0]
                # print("Action %d for car %s is not feasible, using alternative %d" % (actions[idx], car.id, alternative))
                actions[idx] = alternative

        return actions

        
    def execute_action(self, actions):
        """
        Executes action for all agents

        actions - vector of integers (0-8) for all agents.
        """
        if len(actions) != self.num_ego:
            raise ValueError("Error in multicar.py : length of action vector does not match number of agents")

        for idx, car in enumerate(self.list_ego_cars):
            if not car.removed:
                car.execute_action(actions[idx])


    def get_avg_speeds(self, map_car_values):
        """
        Returns average speed of all vehicles (SUMO and controlled)
        for each road type
        0 - single-lane entrance road
        1 - lane 0 on edge 2 that must merge
        2 - lane 0 on edge 1
        3 - lane 1 of edge 2 that receives the merge
        4 - lane 0 of edge 3, post merge
        5 - all other lanes
        """
        vec_avg_speeds = np.zeros(6)
        vec_counts = np.zeros(6, dtype=np.int)
        for key, val in map_car_values.items():
            if val[7] == 'edge_ramp':
                vec_avg_speeds[0] += val[3]
                vec_counts[0] += 1
            elif val[7] == 'edge2' and val[2] == 0:
                vec_avg_speeds[1] += val[3]
                vec_counts[1] += 1
            elif val[7] == 'edge1' and val[2] == 0:
                vec_avg_speeds[2] += val[3]
                vec_counts[2] += 1
            elif val[7] == 'edge2' and val[2] == 1:
                vec_avg_speeds[3] += val[3]
                vec_counts[3] += 1
            elif val[7] == 'edge3' and val[2] == 0:
                vec_avg_speeds[4] += val[3]
                vec_counts[4] += 1
            else:
                vec_avg_speeds[5] += val[3]
                vec_counts[5] += 1
        # element-wise division to get average, then normalize
        vec_avg_speeds = (vec_avg_speeds / vec_counts) / self.v_threshold
        # NaN may appear when nothing is on the lane
        # Convert to 1 because this should not incur penalty
        vec_avg_speeds[np.isnan(vec_avg_speeds)] = 1.0
    
        return vec_avg_speeds


    def get_avg_speed(self, map_car_values):

        speed_total = 0

        for key, tup in map_car_values.items():
            speed_total += tup[3]
        
        return speed_total/float(len(map_car_values)) / self.v_threshold
    

    def get_count_close(self, map_car_values):
        """
        Counts the number of pair of vehicles with
        forward distance less than threshold and lateral distance
        less than 0.5 lane width (assumed to be 3.2m)

        map_car_values - map from each car ID to tuples
                         (x,y,lane,speed,vtype_ID,signal,length,edgeID)
        """
        # Categorize vehicles into straight lanes
        list_lanes = [[] for i in range(7)]
        for key, val in map_car_values.items():
            if val[7] == 'edge_ramp':
                list_lanes[0].append( (val[0], val[1], val[6]) )
            elif val[7] == 'edge2' and val[2] == 0:
                list_lanes[1].append( (val[0], val[1], val[6]) )
            elif (val[7]=='edge1' or val[7]=='edge3') and val[2]==0:
                list_lanes[2].append( (val[0], val[1], val[6]) )
            elif val[7]=='edge2' and val[2]==1:
                list_lanes[2].append( (val[0], val[1], val[6]) )
            elif (val[7]=='edge1' or val[7]=='edge3') and val[2]==1:
                list_lanes[3].append( (val[0], val[1], val[6]) )
            elif val[7]=='edge2' and val[2]==2:
                list_lanes[3].append( (val[0], val[1], val[6]) )
            elif (val[7]=='edge1' or val[7]=='edge3') and val[2]==2:
                list_lanes[4].append( (val[0], val[1], val[6]) )
            elif val[7]=='edge2' and val[2]==3:
                list_lanes[4].append( (val[0], val[1], val[6]) )
            elif (val[7]=='edge1' or val[7]=='edge3') and val[2]==3:
                list_lanes[5].append( (val[0], val[1], val[6]) )
            elif val[7]=='edge2' and val[2]==4:
                list_lanes[5].append( (val[0], val[1], val[6]) )
            elif (val[7]=='edge1' or val[7]=='edge3') and val[2]==4:
                list_lanes[6].append( (val[0], val[1], val[6]) )
            elif val[7]=='edge2' and val[2]==5:
                list_lanes[6].append( (val[0], val[1], val[6]) )

        for idx in range(7):
            # sort by x value
            list_lanes[idx].sort()
            
        # Count
        count_close = 0
        for idx in range(7):
            for idx2 in range(len(list_lanes[idx])-1):
                x1 = list_lanes[idx][idx2][0]
                y1 = list_lanes[idx][idx2][1]
                x2 = list_lanes[idx][idx2+1][0]
                y2 = list_lanes[idx][idx2+1][1]
                # only need the length of the vehicle in front
                length_2 = list_lanes[idx][idx2+1][2]
                if abs(y2 - y1) < 1.6 and (x2 - length_2 - x1) < self.follow_threshold:
                    count_close += 1
                    if x2 - length_2 - x1 < 0:
                        print("multicar.get_count_close(): cars overlap, check for collision")

        return count_close
    

    def count_success(self):
        """
        Returns count of CONTROLLED vehicles with must_merge=True
        that have succeeded.
        i.e. started from lane 0 of either straight or entrance edge,
        and have edge 3 lane 0 as goal lane, and succeeded
        """
        count = 0
        for car in self.list_ego_cars:
            if car.must_merge and car.current_edge_ID=="edge3" and car.lane==0:
                count += 1
    
        return count


    def count_remaining(self, map_car_values):
        """
        Returns count of CONTROLLED vehicles on lane 0 of edge 2

        Not sure whether to use this for reward
        """
        count = 0
        for car in self.list_ego_cars:
            tup = map_car_values[car.id]
            if tup[2] == 0 and tup[7] == 'edge2':
                count += 1

        return count


    def construct_global_tensor(self, map_car_values):
        """
        Returns a tensor composed of:
        1. binary occupancy grid, in global coordinates
        2. grid of ratio between absolute speed and speed limit
        3. binary matrices for signal status

        (0,0) of grids correspond to (x=0,y=0) of SUMO network
        """
        t = np.zeros((self.n_rows, self.n_cols, 4))
        mat_occupancy = np.zeros((self.n_rows, self.n_cols))
        mat_relspeed = np.zeros((self.n_rows, self.n_cols))
        mat_signal_left = np.zeros((self.n_rows, self.n_cols))
        mat_signal_right = np.zeros((self.n_rows, self.n_cols))

        # occupancy
        for key, tup in map_car_values.items():
            x = tup[0]
            y = tup[1] # is negative
            speed = tup[3]
            signal = tup[5]
            num_cells = int(round(tup[6] / self.res_forward))

            row = int(round( x / self.res_forward ))
            col = int(round( abs(y) / self.res_side ))

            for r in range(row-num_cells, row):
                if 0 <= r and r < self.n_rows and 0 <= col and col < self.n_cols:
                    mat_occupancy[r,col] = 1
                    mat_relspeed[r,col] = speed / 29.0
                    if 2 & signal:
                        mat_signal_left[r,col] = 1
                    elif 1 & signal:
                        mat_signal_right[r,col] = 1
                else:
                    with open('log_error.txt', 'a') as f:
                        f.write('multicar.py construct_global_tensor(): row %d, col %d, x %.2f, y %.2f' % (r, col, x, y))

        t[:,:,0] = mat_occupancy
        t[:,:,1] = mat_relspeed
        t[:,:,2] = mat_signal_left
        t[:,:,3] = mat_signal_right

        return t


    def get_global_state(self):
        """
        Returns 2D matrix, each row is state of one car, specifically
        (normalized x, normalized y, normalized speed)
        """
        global_state = []

        # temp_car = self.list_ego_cars[0]
        # sublane_normalizer = float(temp_car.final_edge_num_lanes * self.sim.sublanes_per_lane)

        for idx, car in enumerate(self.list_ego_cars):
            x = (car.x - self.total_length/2) / self.total_length
            y = (car.y + self.total_width/2) / self.total_width # because y is always negative
            v = car.vel / 29.0
            global_state.append( np.array([ x, y, v ]) )

        return np.array(global_state)
        

    def get_local_observations(self):
        """
        May need to alter the format of returned object,
        to fit the requirement of batch training
        """

        # list for the tensor part
        list_t = []
        # list for the vector part
        list_v = []
        
        # get a representative matrix's dimensions
        n_rows = self.list_ego_cars[0].observation.rows
        n_cols = self.list_ego_cars[0].observation.cols

        for car in self.list_ego_cars:
            
            # construct observation tensor from the
            # matrices in each agent's observation object
            tensor_obs = np.zeros([n_rows,n_cols,2])
            tensor_obs[:,:,0] = car.observation.mat_occupancy
            tensor_obs[:,:,1] = car.observation.mat_relspeed
            
            # the_rest = np.zeros(5)
            the_rest = np.zeros(3)
            the_rest[0] = car.vel / 29.0
            the_rest[1] = car.delta_sublane / float(car.final_edge_num_lanes * self.sim.sublanes_per_lane)
            the_rest[2] = car.dist_to_goal

            list_t.append( tensor_obs )
            list_v.append( the_rest )

        return list_t, list_v


    def step(self, actions):
        """
        Main function to be called by the RL algorithm.
        
        actions - vector of actions for all agents.

        Returns global state vector,
        one observation per agent (list of (tensor, vec) tuples),
        global reward, and done indicator.
        """

        self.execute_action(actions)
        
        self.sim.step()

        # Construct map from each vehicle to quantities 
        # observed by all other vehicles.
        # More efficient than calling the same TraCI functions
        # within each car's instantiation.
        map_car_values = {}
        for cid in self.sim.set_present:
            x, y = self.sim.traci.vehicle.getPosition(cid)
            lane = self.sim.traci.vehicle.getLaneIndex(cid)
            speed = self.sim.traci.vehicle.getSpeed(cid)
            vtype_ID = self.sim.traci.vehicle.getTypeID(cid)
            signal = self.sim.traci.vehicle.getSignals(cid)
            length = self.sim.traci.vehicle.getLength(cid)
            edge = self.sim.traci.vehicle.getRoadID(cid)
            map_car_values[cid] = (x,y,lane,speed,vtype_ID,signal,length,edge)

        # Update state of all agents
        local_rewards = np.zeros(self.num_ego)
        reward = 0 # global reward is sum of subset of local reward # HERE
        all_terminal = True
        collision = False
        count_reached_end = 0
        for idx, car in enumerate(self.list_ego_cars):
            if not car.removed:
                terminal, local_rewards[idx], use_for_global_reward = car.update_state(map_car_values, actions[idx])
                if terminal and not car.collision:
                    count_reached_end += 1
                if terminal and not car.removed:
                    car.remove()
                    self.sim.set_present.discard(car.id)
                if car.collision:
                    collision = True
                # if any car did not terminate, then
                # all_terminal is false
                all_terminal &= terminal

        global_state = self.get_global_state()

        reward = np.sum(local_rewards)

        local_t, local_v = self.get_local_observations()

        if self.sim.set_colliding or collision:
            # Need the second condition because SUMO prevents
            # turning into adjacent car, but if our agent takes
            # that action, we count as collision anyway
            self.done = True
            for car in self.list_ego_cars:
                if not car.removed:
                    car.remove()
        elif all_terminal:
            # no collisions occurred all cars reached destination
            self.done = True

        return global_state, local_t, local_v, reward, local_rewards, self.done


    def reset(self):
        """
        Reinstantiate everything for a new episode
        """
        self.list_ego_cars = []
        for idx in range(self.num_ego):
            car = egocar_simple.EgoCar(self.sim, self.goal_lanes[idx], self.goal_positions[idx], self.routes[idx], self.speeds[idx], self.lanes[idx], self.init_positions[idx], self.list_ids[idx], self.vtypes[idx], self.depart_times[idx], safety=self.safety)
            self.list_ego_cars.append(car)

        # Wait for all controlled vehicles to be on the road
        all_ready = False
        need_to_restart = False
        list_ID_arrived = []
        while not all_ready:
            self.sim.step()
            all_ready = True
            for car in self.list_ego_cars:
                if car.id in self.sim.list_arrived:
                    # This case occurs when SUMO traffic is too
                    # dense and one controlled car arrives before
                    # last controlled car departs
                    need_to_restart = True
                    list_ID_arrived.append( car.id )
                else:
                    car.updateState()
                    if car.lane < 0:
                        all_ready = False
            if need_to_restart:
                break

        if need_to_restart:
            for car in self.list_ego_cars:
                if car.id not in list_ID_arrived:
                    # remove all controlled cars except those that
                    # have arrived (they are already removed by SUMO)
                    car.remove()
            return None, None, None, True

        self.done = False

        # Get number of junctions along car's route
        # Assumes that all cars have same number of junctions
        self.n_junctions = len(self.list_ego_cars[0].list_edges) - 1
        self.len_global_state = self.num_ego * (7 + self.n_junctions) + 4

        # Take an initial step with action = do nothing
        # so that local observations for all agents will be
        # populated with meaningful values
        global_state, local_t, local_v, reward, local_rewards, done = self.step(np.zeros(self.num_ego))

        if done:
            # raise ValueError("multicar.py reset(): Done is True")
            with open('log_error.txt', 'a') as f:
                f.write('multicar.py reset(): done is True\n')

        return global_state, local_t, local_v, done

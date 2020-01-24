"""
Maintains local observation for each agent
Includes occlusion effects
See report for full definition of observation space
"""

import numpy as np
import matplotlib.pyplot as plt


class Observation(object):

    def __init__(self, sim, eid, left, right, front, back, res_forward, res_side=0.8, occlusion=False):
        """__init__
        
        Initialize observation channels
        sim - shared simulation object
        eid - ID of ego-car that receives this observation
        left - number of visible sublanes on the left
        right - number of visible sublanes on the right
        front - distance (m) visible in front of vehicle
        back - distance (m) visible behind vehicle
        res_forward - grid-size resolution in the forward direction
        res_side - lateral grid-size resolution
        occlusion - False: no occlusion
        """
        self.sim = sim
        self.traci = sim.traci
        self.eid = eid
        self.left = left
        self.right = right
        self.res_forward = res_forward
        self.res_side = res_side
        # number of grid cells in front and back
        self.front = int(round(front/self.res_forward))
        self.back = int(round(back/self.res_forward))
        # number of cells occupied by ego-car in forward direction
        self.num_ego_cells = int(round(self.traci.vehicle.getLength(self.eid)/self.res_forward))
        # number of rows
        self.rows = self.front + self.back + 1
        # number of columns
        self.cols = self.left + self.right + 1
        self.occlusion = occlusion
        self.reset_matrices()


    def reset_matrices(self, e_vel=30):
        """
        Each grid is a matrix centered on the ego-car
        position. Row indices increase along the ego-car's
        direction of travel, while column indices increase
        from left to right (in the the ego-car's reference
        frame). This means the (0,0) grid cell is on the
        left of the ego-car and behind the ego-car.

        e_vel - speed of self
        """
        # occupancy grid (1:present, 0:else)
        self.mat_occupancy = np.zeros((self.rows, self.cols))
        # relative speed (normalized to 25m/s)
        # blank locations are negative, relative to car
        self.mat_relspeed = np.ones((self.rows, self.cols)) * (-e_vel) / 25.0
        # vehicle type (1:is of type, 0: else)
        self.mat_type_car = np.zeros((self.rows, self.cols))
        self.mat_type_truck = np.zeros((self.rows, self.cols))

    
    def update(self, ex, ey, e_abs_sublane, num_lanes, e_vel,
               map_car_values):
        """update(float, int)
        
        Update grid with new location of the ego car
        ex - ego-car's x position (assumed to be direction of travel)
        ey - ego-car's y position
        e_abs_sublane - ego-car's absolute sublane number
        num_lanes - number of lanes on current edge
        e_vel - speed of ego-car
        map_car_values - map from each car ID to tuples
                         (x,y,lane,speed,vtype_ID,signal,length,edgeID)
        """
        self.reset_matrices(e_vel)
        car_on_left = False
        car_on_right = False
        # parse all cars present on the road
        # for each vehicle, populate its location
        # relative speed, type, left, and right
        # into matrices
        for cid, tup in map_car_values.items():
            x = tup[0]
            y = tup[1]
            lane = tup[2]
            speed = tup[3]
            vtype_ID = tup[4]
            signal = tup[5]
            num_cells = int(round(tup[6] / self.res_forward))

            if lane < 0:
                continue
            # fill cells for car if inside grid
            r_range, c = self.loc2cell(ex, ey, x, y, num_cells)
            if c >= 0 and c < self.cols:
                for r in r_range:
                    if r >= 0 and r < self.rows:
                        self.mat_occupancy[r,c] = 1
                        # normalized by 25m/s
                        self.mat_relspeed[r,c] = (speed - e_vel) / 25.0

                    if r == self.back or r == self.front:
                        if self.left-3 < c and c < self.left:
                            car_on_left = True
                        elif self.left < c and c < self.left + 3:
                            car_on_right = True

        if self.occlusion:
            self.occlude()

        # fill columns that are outside road
        for c in range(self.cols):
            l = self.col2sublane(e_abs_sublane, c) 
            if (l <= 0) or (l >= num_lanes*self.sim.sublanes_per_lane):
                self.mat_occupancy[:,c] = 1

        return car_on_left, car_on_right


    def display(self):
        """ display occupancy """
        # pad columns to display lane length
        colmul = int(round(3.3/self.res_forward))
        fgrid = np.zeros((self.rows, self.cols*colmul))
        for i in range(self.cols):
            for j in range(colmul):
                fgrid[:,i*colmul+j] = self.mat_occupancy[:,i]
        # plot
        plt.figure(1)
        plt.clf()
        plt.axis('off')
        plt.imshow(np.flipud(fgrid), cmap='jet')
        plt.show(block=False)
        plt.pause(1e-10)


    def loc2cell(self, ex, ey, x, y, num_cells):
        """
        Assumes that ego-car is traveling along the 
        increasing x direction, so that increasing x 
        corresponds to increasing row index, and 
        increasing y corresponds to decreasing column index

        (ex, ey) - ego-car's position
        (x, y) - other car's position
        num_cells - other car's length in number of cells
        """
        c = int(round((ey - y)/self.res_side)) + self.left

        r_high = int(round((x - ex)/self.res_forward)) + self.back + 1
        r_low = r_high - num_cells

        return range(r_low, r_high), c


    def col2sublane(self, e_sublane, c):
        """ column number to sublane """
        return e_sublane + (self.left - c)


    def occlude_cell(self, r, c):
        """
        -1 for occluded cells in mat_occupancy
        0 for corresponding cells of all other matrices
        """
        self.mat_occupancy[r,c] = -1
        self.mat_relspeed[r,c] = 0
        self.mat_type_car[r,c] = 0
        self.mat_type_truck[r,c] = 0
        self.mat_signal_left[r,c] = 0
        self.mat_signal_right[r,c] = 0


    def occlude(self):
        """
        Top left quadrant of ego-car's grid:
        occluded cells are those to the left and northwest
        diagonal.
        Top right quadrant: occluded cells are those on the
        right and northeast diagonal.
        Bottom left quadrant: left and southwest diagonal
        Bottom right quadrant: right and southeast diagonal
        """

        # Occlusion: -1 for occluded occupancy cells,
        # 0 for all other matrices
        # r_high_self = self.back + self.num_ego_cells
        # cell immediately above car
        r_high_self = self.back + 1 
        # r_low_self = self.back
        # cell immediately below car
        r_low_self = self.back - self.num_ego_cells
        
        c_self = self.left

        # Forward along ego-car's column
        prev = 0
        found = False
        for r in range(r_high_self, self.rows):
            if found:
                self.occlude_cell(r, c_self)
            else:
                # transition from 1 to 0, meaning crossed a
                # full vehicle length
                if prev==1 and self.mat_occupancy[r,c_self]==0:
                    found = True
                    self.occlude_cell(r, c_self)
                prev = self.mat_occupancy[r,c_self]

        # Backward along ego-car's column
        prev = 0
        found = False
        for r in range(r_low_self, -1, -1):
            if found:
                self.occlude_cell(r, c_self)
            else:
                if prev==1 and self.mat_occupancy[r,c_self]==0:
                    found = True
                    self.occlude_cell(r, c_self)
                prev = self.mat_occupancy[r,c_self]

        # Right along rows where ego-car has occupancy
        for r in range(r_low_self+1, r_high_self):
            found = False
            for c in range(c_self+1, self.cols):
                if found:
                    self.occlude_cell(r, c)
                else:
                    if self.mat_occupancy[r,c]==1:
                        # detected another car, so all cells
                        # on the right should be occluded
                        found = True

        # Left along rows where ego-car has occupancy
        for r in range(r_low_self+1, r_high_self):
            found = False
            for c in range(c_self-1, -1, -1):
                if found:
                    self.occlude_cell(r,c)
                else:
                    if self.mat_occupancy[r,c]==1:
                        # detected another car, so all cells
                        # on the left should be occluded
                        found = True

        # Top right
        for r in range(r_high_self, self.rows):
            found = False
            for c in range(c_self+1, self.cols):
                if found:
                    # use r+1 to occlude the row above
                    self.occlude_cell(r, c)
                else:
                    if self.mat_occupancy[r,c]==1:
                        # everything to the right and one row
                        # above should be occluded
                        found = True
                    elif self.mat_occupancy[r,c]==0 and self.mat_occupancy[r-1,c]==1 and r != r_high_self:
                        found = True
                    
        # Top left
        for r in range(r_high_self, self.rows):
            found = False
            for c in range(c_self-1, -1, -1):
                if found:
                    self.occlude_cell(r, c)
                else:
                    if self.mat_occupancy[r,c]==1:
                        found = True
                    elif self.mat_occupancy[r,c]==0 and self.mat_occupancy[r-1,c]==1 and r != r_high_self:
                        found = True

        # Bottom right
        for r in range(r_low_self, -1, -1):
            found = False
            for c in range(c_self+1, self.cols):
                if found:
                    self.occlude_cell(r, c)
                else:
                    if self.mat_occupancy[r,c]==1:
                        # everything to the right and one row
                        # below should be occluded
                        found = True
                    elif self.mat_occupancy[r,c]==0 and self.mat_occupancy[r+1,c]==1 and r != r_low_self:
                        found = True

        # Bottom left
        for r in range(r_low_self-1, -1, -1):
            found = False
            for c in range(c_self-1, -1, -1):
                if found:
                    self.occlude_cell(r, c)
                else:
                    if self.mat_occupancy[r,c]==1:
                        found = True
                    elif self.mat_occupancy[r,c]==0 and self.mat_occupancy[r+1,c]==1 and r != r_low_self:
                        found = True

"""Adapted from code by Mustafa Mukadam."""

import argparse, os, sys, subprocess
import traci
import sumolib

class Simulator(object):
  
  def __init__(self, port=8841, sid='ego_car',
               list_id=[],
               other_lc_mode=0b1000000000,
               sublane_res=None, seed=123456):
    """__init__(int)
    Initialize and start simulator with command-line arguments
    port - for SUMO simulator
    sid - string ID for a single ego-car
    list_id - list of IDs in the case of multiple 
              controlled cars
    other_lc_mode - lane change mode of other cars
    sublane_res - if not None, activates sublane mode with
                  --lateral-resolution <sublane>
    seed - RNG seed
    """
    # parse arguments
    parser = argparse.ArgumentParser(description='Sumo Simulator')
    parser.add_argument('--gui', default=False, action='store_true', help='Pass if GUI needed')
    parser.add_argument('--env', help='Load this environment')
    args = parser.parse_args()
    self.gui = args.gui
    self.env = args.env
    self.port = port

    # path to the sumo binary
    if self.gui:
      sumo_bin = "/usr/local/bin/sumo-gui"
    else:
      sumo_bin = "/usr/local/bin/sumo"

    # call sumo simulator
    if sublane_res:
      self.process = subprocess.Popen([
        sumo_bin, "--lateral-resolution", "%f" % sublane_res,
        "--collision.action", "warn",
        "-c", self.env,
        "--remote-port", str(self.port),
        "--no-step-log", str(True),
        "--no-warnings", str(True),
        "--seed", str(seed)
        ],
        stdout=sys.stdout, 
        stderr=sys.stderr)
    else:
      self.process = subprocess.Popen([
        sumo_bin, 
        "-c", self.env,
        "--remote-port", str(self.port),
        "--no-step-log", str(True),
        "--no-warnings", str(True),
        "--seed", str(seed)
        ],
        stdout=sys.stdout, 
        stderr=sys.stderr)

    # traci handle
    traci.init(self.port)
    self.traci = traci

    # other variables
    self.dt = self.traci.simulation.getDeltaT()
    self.id = sid
    self.list_id = list_id
    self.other_lc_mode = other_lc_mode
    # set of all cars present on the road
    self.set_present = set()
    self.sublane_res = sublane_res
    if sublane_res:
      self.sublanes_per_lane = int(3.2 / self.sublane_res)
    else:
      self.sublanes_per_lane = None
    # list of IDs of cars colliding at the current time step
    self.list_colliding = []

    # get number of lanes
    for edge in sumolib.output.parse(self.env.replace('.sumocfg','.edg.xml'), ['edge']):
      self.num_lanes = int(edge.numLanes)


  def step(self):
    """ Step simulator; set other cars """
    self.traci.simulationStep()
    for cid in self.traci.simulation.getLoadedIDList():
      if cid == self.id or cid in self.list_id:
        continue
      self.traci.vehicle.setLaneChangeMode(cid, self.other_lc_mode)

    # Update set of IDs of vehicles that are still
    # present on road
    list_departed = self.traci.simulation.getDepartedIDList()
    for ID in list_departed:
      self.set_present.add(ID)

    self.list_arrived = self.traci.simulation.getArrivedIDList()
    for ID in self.list_arrived:
      self.set_present.discard(ID)

    list_teleporting = self.traci.simulation.getStartingTeleportIDList()
    for ID in list_teleporting:
      self.set_present.discard(ID)

    # may contain duplicates if option --collision.action is set to "warn"
    self.set_colliding = set(self.traci.simulation.getCollidingVehiclesIDList())


  def close(self):
    """ Close simulator """
    traci.close()
    sys.stdout.flush()

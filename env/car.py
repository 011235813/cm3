"""Adapted from code by Mustafa Mukadam."""

class Car(object):

  def __init__(self, sim, vel, lane, pos=0, cid='car'):
    """__init__(obj, float, int, float, string)
    
    Initialize a car in sumo
    """
    self.sim = sim
    self.traci = sim.traci
    self.vel = vel
    self.lane = lane
    self.pos = pos
    self.x = 0
    self.y = 0
    self.id = cid
    self.dt_ms = self.traci.simulation.getDeltaT()
    self.dt = self.dt_ms/1000.0
    self.sublane_idx = 0
    self.abs_sublane = 0


  def __str__(self):
    """ print overload """
    return "<%s: pos=%.2f, vel=%.2f, lane=%d>" % (self.id, self.pos, self.vel, self.lane) 

  def updateState(self):
    """ update state of car """
    self.pos = self.traci.vehicle.getLanePosition(self.id)
    self.vel = self.traci.vehicle.getSpeed(self.id)
    self.lane = self.traci.vehicle.getLaneIndex(self.id)
    self.x, self.y = self.traci.vehicle.getPosition(self.id)
    if self.sim.sublane_res:
      self.sublane_idx = int(round(self.traci.vehicle.getLateralLanePosition(self.id) / self.sim.sublane_res))
      # absolute sublane, counting from sublane 0 of lane 0
      self.abs_sublane = int((0.5 + self.lane)*self.sim.sublanes_per_lane) + self.sublane_idx

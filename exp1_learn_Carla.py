# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-0.9.11-py3.6-linux-x86_64.egg')[0])
except IndexError:
    pass
import carla

from carla import VehicleLightState as vls

import argparse
import logging
from numpy import random

client=carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
t = world.get_spectator().get_transform()
coordinate_str = "(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z)
print (coordinate_str)
time.sleep(2)
# map = world.get_map()
blueprint_library = world.get_blueprint_library()
vehicle = blueprint_library.filter('vehicle.*')
# model_3 = blueprint_library.filter("model3")[0]
# actor_list = []
# transform =  carla.Transform(carla.Location(x=38.714229583740234,y=3.2649950981140137, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
# vehicle = world.spawn_actor(model_3, transform)
waypoint = world.get_waypoint(vehicle.get_location())
# physics_control = vehicle.get_physics_control()

# For each Wheel Physics Control, print maximum steer angle
# for wheel in physics_control.wheels:
#     print (wheel.max_steer_angle)
lane_id = waypoint.lane_id
road_id  = waypoint.road_id


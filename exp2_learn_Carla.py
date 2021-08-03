#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:42:09 2021

@author: nsh1609
"""

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
import vector3d
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow as tf1
import tensorflow.python.keras.backend as backend
from threading import Thread
from math import sqrt, asin
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-0.9.11-py3.6-linux-x86_64.egg')[0])
except IndexError:
    pass
import carla

# class Vector3D:
#     def __init__(self, x, y, z):
#         self._x = x
#         self._y = y
#         self._z = z
#     def __mul__(self, other):
#         return Vector3D(self._x*other._x, self._y*other._y, self._z*other._z)
#     def mag(self):
#         return sqrt((self._x)^2 + (self._y)^2 + (self._z)^2)
#     def dot(self, other):
#         temp = self * other
#         return temp._x + temp._y + temp._z
#     def cos_theta(self):
#         #vector's cos(angle) with the z-axis
#         return self.dot(Vector3D(0,0,1)) / self.mag() #(0,0,1) is the z-axis unit vector
#     def phi(self):
#         #vector's 
#         return asin( self.dot(Vector3D(0,0,1)) / self.mag() )
#     def __repr__(self):
#         return "({x}, {y}, {z})".format(x=self._x, y=self._y, z=self._z)

client=carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
model_3 = blueprint_library.filter("model3")[0]
collision_hist = []
actor_list = []

# transform =  carla.Transform(carla.Location(x=33.74591064453125,y=182.09759521484375, z=0.300000), carla.Rotation(pitch=0.000000, yaw=-90.000000, roll=0.000000))
#transform =  carla.Transform(carla.Location(x=38.714229583740234,y=3.2649950981140137, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
transform = carla.Transform(carla.Location(x=118.2643051147461,y=3.2649950981140137, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
start_speed =1.0
vehicle =world.spawn_actor(model_3, transform)
actor_list.append(vehicle)
# vehicle.apply_control(carla.VehicleControl(throttle=1.0))
# a=vehicle.get_acceleration()
# v=vehicle.get_velocity();
# l=vehicle.get_location()
# details=vehicle.get_physics_control()
# # forward_vec = vehicle.get_transform().get_forward_vector() #forward vector will get from MPC
# v=vehicle.get_velocity()
# check=math.sqrt(v.x**2 + v.y**2 + v.z**2)
# if check==0.9794611930847168:
#     print("e")
#     v=carla.Vector3D(x=0.387271, y=0.302593, z=0.870895)
  
# forward_vec=v*(1/(math.sqrt(v.x**2 + v.y**2 + v.z**2)))
# velocity_vec = start_speed * forward_vec
# vehicle.set_target_velocity(velocity_vec)
# zz=np.array([a.x,a.y,a.z])


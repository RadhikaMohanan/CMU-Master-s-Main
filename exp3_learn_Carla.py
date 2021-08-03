#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:16:58 2021

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
import tensorflow as tf1# vehicle.set_target_velocity(carla.Vector3D(1.0, 0.0, 0.0))
import tensorflow.python.keras.backend as backend
from threading import Thread
from math import sqrt, asin
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-0.9.11-py3.6-linux-x86_64.egg')[0])
except IndexError:
    pass
import carla
start_speed=3
client=carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
model_3 = blueprint_library.filter("model3")[0]
collision_hist = []
actor_list = []
transform =  carla.Transform(carla.Location(x=38.714229583740234,y=3.2649950981140137, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
# transform =  carla.Transform(carla.Location(x=33.74591064453125,y=182.09759521484375, z=0.300000), carla.Rotation(pitch=0.000000, yaw=-90.000000, roll=0.000000))
vehicle = world.spawn_actor(model_3, transform)
# vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0))
# v= carla.Vector3D(x=0.387271, y=0, z=0)
# forward_vec=v*(1/(math.sqrt(v.x**2 + v.y**2 + v.z**2)))
# velocity_vec = start_speed * forward_vec
# vehicle.set_target_velocity(velocity_vec)
vehicle.set_target_velocity(carla.Vector3D(10.0, 0.0, 0.0))


v=vehicle.get_velocity()
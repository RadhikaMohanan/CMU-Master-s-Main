#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:59:54 2021

@author: nsh1609
"""
#######MPC framework without details###################
from cvxpy import *
import numpy as np
import scipy as sp
from scipy import sparse
from scipy import signal
import glob
import os
import sys
import random
import time
import timeit
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.layers import Input,merge,LSTM
from keras.models import Model
from keras.callbacks import TensorBoard

from keras.models import Sequential

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow as tf1
import tensorflow.python.keras.backend as backend
from threading import Thread
import casadi as ca
import numpy as np
import time
from os import system
#
# from high_mpc.common.quad_index import *

#

from tqdm import tqdm

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-0.9.11-py3.6-linux-x86_64.egg')[0])
# except IndexError:
#     pass
# import carla

# client = carla.Client("localhost", 2000)
# client.set_timeout(2.0)
# world = client.get_world()
# blueprint_library = world.get_blueprint_library()
# settings = world.get_settings()
# dt=0.05
# settings.fixed_delta_seconds = dt
# # settings.synchronous_mode = True
# world.apply_settings(settings)
# model_3 = blueprint_library.filter("model3")[0]
# actor_list = []
# transform =  carla.Transform(carla.Location(x=38.714229583740234,y=3.2649950981140137, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
# vehicle = world.spawn_actor(model_3, transform)
# actor_list.append(vehicle)
# # t3= carla.Transform(carla.Location(x=125,y=-2, z=0.300000), carla.Rotation(pitch=0.000000, yaw=-180.000000, roll=0.000000))
# # vehicle3 = world.spawn_actor(model_3, t3)
# # actor_list.append(vehicle3)
# # vehicle3.set_target_velocity(carla.Vector3D(10.0, 0.0, 0.0))

# # t4=carla.Transform(carla.Location(x=74.87380981445312,y=3.2649950981140137, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
# # vehicle4 = world.spawn_actor(model_3, t4)
# # actor_list.append(vehicle4)
# # vehicle4.set_target_velocity(carla.Vector3D(10.0, 0.0, 0.0))

# #t1 vehicle is at right 
# #t2 vehicle is at left
# t1=carla.Transform(carla.Location(x=105,y=43.455867767333984, z=0.300000), carla.Rotation(pitch=0.000000, yaw=-90.000000, roll=0.000000))
# vehicle1 = world.spawn_actor(model_3, t1)
# actor_list.append(vehicle1)
# vehicle1.set_target_velocity(carla.Vector3D(0.0, -5.0, 0.0))

# t2=carla.Transform(carla.Location(x=97.56910705566406,y=-44.676116943359375, z=0.300000), carla.Rotation(pitch=0.000000, yaw=90.000000, roll=0.000000))
# vehicle2 = world.spawn_actor(model_3, t2)
# actor_list.append(vehicle2)
# vehicle2.set_target_velocity(carla.Vector3D(0.0, 5.0, 0.0))

# #States of MPC
# pe = vehicle.get_location()
# ve = vehicle.get_velocity()
# ae = vehicle.get_acceleration()
# p1=vehicle1.get_location()
# v1=vehicle1.get_velocity()
# ac1=vehicle1.get_acceleration()
# p2=vehicle2.get_location()
# v2=vehicle2.get_velocity()
# ac2=vehicle2.get_acceleration()
# xe=np.array([pe.x,pe.y,pe.z,ve.x,ve.y,ve.z,ae.x,ae.y,ae.z])
# # # x1=np.array([p1.x,p1.y,p1.z,v1.x,v1.y,v1.z,ac1.x,ac1.y,ac1.z]) 
# # # x2=np.array([p2.x,p2.y,p2.z,v2.x,v2.y,v2.z,ac2.x,ac2.y,ac2.z])
# x1=np.array([p1.x,p1.y,p1.z]) 

x1=np.array([105,43.455867767333984, 0.300000])
x2=np.array([97.56910705566406,-44.676116943359375, 0.300000])
# x2=np.array([p2.x,p2.y,p2.z])
Ac = np.array ([
  [0., 0., 0., 1., 0., 0., 0., 0., 0.],
  [0., 0., 0., 0., 1., 0., 0., 0., 0.],
  [0., 0., 0., 0., 0., 1., 0., 0., 0.],
  [0., 0., 0., 0., 0., 0., 1., 0., 0.],
  [0., 0., 0., 0., 0., 0., 0., 1., 0.],
  [0., 0., 0., 0., 0., 0., 0., 0., 1.],
  [0., 0., 0., 0., 0., 0., 0., 0., 0.],
  [0., 0., 0., 0., 0., 0., 0., 0., 0.],
  [0., 0., 0., 0., 0., 0., 0., 0., 0.]])

Bc=np.array ([
  [0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.],
  [1., 0., 0.],
  [0., 1., 0.],
  [0., 0., 1.]])

C1=np.array ([
  [1., 0., 0.],
  [0., 1., 0.],
  [0., 0., 1.],
  [0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.]])
Cc=C1.T
Dc=np.zeros((3,3))
dt1=0.05 #Time step taken as 0.5
Sysd= sp.signal.cont2discrete((Ac,Bc,Cc,Dc), dt1, method='zoh', alpha=None)
Ad=sparse.csc_matrix(Sysd[0])
Bd=sparse.csc_matrix(Sysd[1])
[nx, nu] = Bd.shape
#Objective Function
Q=sparse.diags([0,0,0,1,1,1,1,1,1])
QN=sparse.diags([1,1,1,1,1,1,1,1,1])
R=sparse.eye(3)

# #Initial and Reference States and input
u0= np.array([2,0,0])
x0=np.array([38.714229583740234,3.2649950981140137,0.300000,0,0,0,0,0,0])
xr=np.array([118.2643051147461,3.2649950981140137,0.300000,5,0,0,0,0,0])
ur=np.array([0,0,0])
# # Prediction horizon
N = 10
x1=np.array([105,43.455867767333984, 0.300000])
x2=np.array([97.56910705566406,-44.676116943359375, 0.300000])

def states(const_vel,init_state, dt):#Obstacle Prediction
    final_state=np.zeros(3) 
    final_state[0]=init_state[0]
    final_state[1]=init_state[1]+dt*const_vel
    # print('const',const_vel,'xy',final_state[1])
    final_state[2]=init_state[2]
    return final_state
def intersect_point(x1,x2):
    cross=np.zeros(3)
    cross[0]=x2[0]
    cross[1]=x1[1]
    cross[2]=x2[2]
    return cross
#Constraints:
d1e=20
d2e=20  

# Define problem
u = Variable((nu, N))
x = Variable((nx, N+1))
x_init = Parameter(nx)
objective = 0
constraints = [x[:,0] == x_init]
# x1=np.array([p1.x,p1.y,p1.z]) 
# x2=np.array([p2.x,p2.y,p2.z])
# x_init.value = x0
cross_1_e= intersect_point(x0,x1)
cross_2_e= intersect_point(x0,x2)
# x_init.value = x0
for k in range(N):
    dist1=np.linalg.norm(x1-cross_1_e)
    dist2=np.linalg.norm(x2-cross_2_e)
    
    x2=states(1,x2,2.5)
    x1=states(-1,x1,2.5)
    if np.min((dist1,dist2))==dist1:
        cross=cross_1_e
    else:
        cross=cross_2_e
        
    # diste=np.linalg.norm(x2-cross_2_e)   
   
    objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k]-ur, R)
    print("this")
    constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k]]
    
    # constraints += [x_init==x0]
    constraints += [x[1,k]<=cross[1]+d1e]
#     constraints += [d1e <= np.linalg.norm(x[:3,k] - states(10,x1,0.1)), d2e <= np.linalg.norm(x[:3,k] - states(10,x2,0.1))]
objective += quad_form(x[:,N] - xr, QN)
prob = Problem(Minimize(objective), constraints)
# prob.solve(solver=OSQP, warm_start=True)
# x_init.value = x0
# prob.solve(solver=OSQP, warm_start=True)
# u_val=np.linalg.norm(u[:,0].value)
# print(x.value.shape)
# x0 = Ad.dot(x0) + Bd.dot(u[:,0].value)

# Simulate in closed loop
nsim = 15
for i in range(nsim):
    x_init.value = x0
    prob.solve(solver=OSQP, warm_start=True)
    # u_val=np.linalg.norm(u[:,0].value)
    print(x[:,0].value)
    # vehicle.apply_control(carla.VehicleControl(throttle=u_val, steer=0))
    x0 = Ad.dot(x0) + Bd.dot(u[:,0].value)


#[105.01462555 -72.16513824   0.15422072]
#[105.0215683  -67.34603882   0.11152906]
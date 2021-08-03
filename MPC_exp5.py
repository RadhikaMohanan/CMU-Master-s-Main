#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:16:00 2021

@author: nsh1609
"""

from cvxpy import *
import matplotlib.pyplot as plt
from matplotlib import animation, rc
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
from tqdm import tqdm
#Target Vehicles
const_vel=10
d1e=10
d2e=10
#Triple integrator system
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
dt1=0.05 #Time step for continuous to discrete conversion taken as 0.5
Sysd= sp.signal.cont2discrete((Ac,Bc,Cc,Dc), dt1, method='zoh', alpha=None)
# Ad=sparse.csc_matrix(Sysd[0])
# Bd=sparse.csc_matrix(Sysd[1])
Ad=sparse.csc_matrix(Sysd[0])
Bd=sparse.csc_matrix(Sysd[1])
[nx, nu] = Bd.shape
# Objective function
Q=sparse.diags([0,0,0,1,1,1,1,1,1])
QN=sparse.diags([1,1,1,1,1,1,1,1,1])
R=sparse.eye(3)
#Function1 for future states Prediction
def states(const_vel,init_state, dt):#Obstacle Prediction
    final_state=np.zeros(3) 
    final_state[0]=init_state[0]
    final_state[1]=init_state[1]+dt*const_vel
    final_state[2]=init_state[2]
    return final_state

#Function2 for Intersection Coordinates
def intersect_point(x0,x):
    cross=np.zeros(3)
    cross[0]=x[0]
    cross[1]=x0[1]
    cross[2]=x[2]
    return cross

#Function3 for MPC calculation    
def calmpc(x0,xr,u0,ur,x1,x2):    
    # Prediction horizon
    N = 10
   
    # Define problem
    u = Variable((nu, N))
    x = Variable((nx, N+1))
    x_init = Parameter(nx)
    objective = 0
    x_init.value = x0
    constraints = [x[:,0] == x_init]
    """
    Constraints:
        1) collision Avoidance.
        A)Take Way and Give Way
    """
    cross_1_e= intersect_point(x0,x1)
    cross_2_e= intersect_point(x0,x2)
    # x_init.value = x0
    for k in range(N):
   
        x1_new=states(-const_vel,x1,k*0.1)
        x2_new=states(const_vel,x2,k*0.1)
        print(x1_new[1])
        dist1=np.linalg.norm(x1-cross_1_e)
        dist2=np.linalg.norm(x2-cross_2_e)
        
        if np.min((dist1,dist2))==dist1:
            cross=cross_1_e
        else:
            cross=cross_2_e
            
        # diste=np.linalg.norm(x2-cross_2_e)   
       
        objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k]-ur, R)
        constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k]]
        
        # constraints += [x_init==x0]
        constraints += [x[1,k]<=cross[1]+d1e]
    objective += quad_form(x[:,N] - xr, QN)

    prob = Problem(Minimize(objective), constraints)
    # [cp.norm(
    # cp.hstack([
    #     y_hat[col] - cp.trace(
    #         np.transpose((B_hat_star[:,col][:,np.newaxis]*np.sqrt(L)*C_hat[col,:])) @ X)
    #     for col in range(L)
    # ]), 2) <= delta]
    start = time.time()
    prob.solve()
    elapsed_time = time.time() - start
    # print("calc time:{0}".format(elapsed_time) + "[sec]")
    # print(prob.value)
    return u,x,prob.value,x1_new,x2_new


def GetListFromMatrix(x):
    return np.array(x).flatten().tolist()

def Main():
    fig=plt.figure(num=None, figsize=(12, 12))    
    #Starting point of the two vehicles
    x1=np.array([105,43.455867767333984, 0.300000])
    x2=np.array([97.56910705566406,-44.676116943359375, 0.300000])
    
    # Initial and Reference States and input
    u0= np.array([2,0,0])
    x0=np.array([38.714229583740234,3.2649950981140137,0.300000,0,0,0,0,0,0])
    xr=np.array([118.2643051147461,3.2649950981140137,0.300000,5,0,0,0,0,0])
    ur=np.array([0,0,0])
    ego=[]
    veh1=[]
    veh2=[]
    for i in range(1000):
        ustar,xstar,cost,x1,x2=calmpc(x0,xr,u0,ur,x1,x2) 
        # u_val=np.linalg.norm(ustar[:,0].value)
       
        # vehicle.apply_control(carla.VehicleControl(throttle=u_val, steer=0))
        x0 = Ad.dot(x0) + Bd.dot(ustar[:,0].value)
        # print(xstar.value.shape)
        # xt=GetListFromMatrix(xstar.value)
        # print(len(xt))
        ego.append(xstar.value[:3, 0])
        veh1.append(x1)
        veh2.append(x2)
    ego=np.array(ego)
    veh1=np.array(veh1)
    veh2=np.array(veh2) 
    # print(ego.shape)
    #Working with unchanged axes limits
    # plt.subplot(3, 1, 1)
    # plt.plot(ego[:,0], '.r')
    # plt.plot(veh1[:,0], '.b')
    # plt.plot(veh2[:,0], '.g')
    # plt.axis("equal")
    # plt.xlabel("Iteration")
    # plt.ylabel("x_position")
    # plt.grid(True)

    # plt.subplot(3, 1, 2)
    # plt.plot(ego[:,1], '.r')
    # plt.plot(veh1[:,1], '.b')
    # plt.plot(veh2[:,1], '.g')
    # plt.axis("equal")
    # plt.xlabel("Iteration")
    # plt.ylabel("y_position")
    
    # plt.xlim([-10000, 10000])
    # plt.grid(True)

    # plt.subplot(3, 1, 3)
    # plt.plot(ego[:,2], '.r')
    # plt.plot(veh1[:,2], '.b')
    # plt.plot(veh2[:,2], '.g')
    # plt.axis("equal")
    # plt.xlabel("Iteration")
    # plt.ylabel("z_position")
    # plt.grid(True)
    #Working with changed axes limits
    ax=fig.add_subplot(3, 1, 1)
    ax.plot(ego[:,0], '.r')
    ax.plot(veh1[:,0], '.b')
    ax.plot(veh2[:,0], '.g')
    ax.axis("equal")
    plt.xlabel("Iteration")
    plt.ylabel("x_position")
    ax.grid(True)

    ax1=fig.add_subplot(3, 1, 2)
    ax1.plot(ego[:,1], '.r')
    ax1.plot(veh1[:,1], '.b')
    ax1.plot(veh2[:,1], '.g')
    ax1.axis("equal")
    plt.xlabel("Iteration")
    plt.ylabel("y_position")
    
    ax1.set_xlim([-1000, 1000])
    ax1.set_ylim(-1000,1000)
    ax1.grid(True)

    ax2=fig.add_subplot(3, 1, 3)
    ax2.plot(ego[:,2], '.r')
    ax2.plot(veh1[:,2], '.b')
    ax2.plot(veh2[:,2], '.g')
    ax2.axis("equal")
    plt.xlabel("Iteration")
    plt.ylabel("z_position")
    ax2.grid(True)
    
    plt.show()

    

if __name__ == '__main__':
    Main()

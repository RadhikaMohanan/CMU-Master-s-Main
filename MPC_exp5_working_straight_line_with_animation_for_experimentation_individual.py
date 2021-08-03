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
action=1
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
    N = 100
   
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
    it1=intersect_point(x0,x1)
    # print('it1',it1)
    it2=intersect_point(x0,x2)
    itmaxx=np.max((int(it1[0]),int(it2[0])))
    itminx=np.min((int(it1[0]),int(it2[0])))
    xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                 -np.inf,-np.inf,-np.inf])
    xmax = np.array([ 1000, 2,33 ,2, 1, 1,
                  1, 1, 1])
    d1= np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                 -np.inf,-np.inf,-np.inf])

    for k in range(N):
        x1_new=states(-const_vel,x1,k*0.00005)
        x2_new=states(const_vel,x2,k*0.00005)
        # x1_new=states(-const_vel,x1,k*0.00005)
        # x2_new=states(const_vel,x2,k*0.00005)

        objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k]-ur, R)
        constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k]]
        if math.isclose(x1_new[1], it1[1],abs_tol=3) or math.isclose(x2_new[1], it2[1],abs_tol=3):
            # print("check this out",x1_new[1], it1[1])
            # print("check this out",x2_new[1], it2[1])
            if action==1:
                
                constraints += [itmaxx+40<=x[0,k]]
            # if action==2:
            #     constraints += [x[0,k]<=itminx-4]

    objective += quad_form(x[:,N] - xr, QN)
    prob = Problem(Minimize(objective), constraints)
    start = time.time()
    prob.solve(solver=OSQP, warm_start=True)
    elapsed_time = time.time() - start
    # print("calc time:{0}".format(elapsed_time) + "[sec]")
    # print(prob.value)
    return u,x,prob.value,x1_new,x2_new

def GetListFromMatrix(x):
    return np.array(x).flatten().tolist()

def Main():
    # fig=plt.figure(num=None, figsize=(12, 12))    
    #Starting point of the two vehicles
    # x1=np.array([105,10, 0.300000])
    # x2=np.array([97.56910705566406,-10, 0.300000])
    #Original_used_for_everything
    x1=np.array([105,43.455867767333984, 0.300000])
    x2=np.array([97.56910705566406,-44.676116943359375, 0.300000])
    
    # Initial and Reference States and input
    u0= np.array([2,0,0])
    x0=np.array([40,3.2649950981140137,0.300000,0,0,0,0,0,0]) 
    # x0=np.array([38.714229583740234,3.2649950981140137,0.300000,0,0,0,0,0,0]) #for normal case 
    xr=np.array([118.2643051147461,3.2649950981140137,0.300000,5,0,0,0,0,0])
    ur=np.array([0,0,0])
    ego=[]
    veh1=[]
    veh2=[]
    iwant=0;
    ego_max=0;
    for i in range(400):
        ustar,xstar,cost,x1,x2=calmpc(x0,xr,u0,ur,x1,x2) 
        # u_val=np.linalg.norm(ustar[:,0].value)
        # vehicle.apply_control(carla.VehicleControl(throttle=u_val, steer=0))
        x0 = Ad.dot(x0) + Bd.dot(ustar[:,0].value)
        # print(xstar.value[:3,0])
    
        # xt=GetListFromMatrix(xstar.value)
        # print(len(xt))
        if xstar.value[3, 0]>ego_max:
            ego_max=xstar.value[3, 0]
            iwant=i
        ego.append(xstar.value[:, 0])
        veh1.append(x1)
        veh2.append(x2)
    print("max iteration",iwant," egomax: ",ego_max)
    ego=np.array(ego)
    veh1=np.array(veh1)
    veh2=np.array(veh2) 
    return ego,veh1,veh2
    

def animate(i,xego,yego,x11,y11,x22,y22):
    line.set_ydata(xego[:i])
    line.set_xdata(yego[:i])
    line2.set_ydata(x11[:i])
    line2.set_xdata(y11[:i])
    line3.set_ydata(x22[:i])
    line3.set_xdata(y22[:i])
    return line,line2,line3
    

if __name__ == '__main__':
    ego,veh1,veh2=Main()
    xego=ego[:,0]
    yego=ego[:,1]
    x11=veh1[:,0]
    y11=veh1[:,1]
    x22=veh2[:,0]
    y22=veh2[:,1]
    fig=plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([],[], 'bo')
    line2, = ax.plot([],[],'ro')
    line3, = ax.plot([],[],'ro')
    ax.set_xlim(-80, 80)
    ax.set_ylim(0, 250)
    ax.set_xlabel('X-Axis Position')
    ax.set_ylabel('Y-Axis Position')
    ax.legend((line, line2), ('Ego Vehicle', 'Target Vehicle'))
    
    ani = animation.FuncAnimation(fig, animate, frames=len(x11), fargs=( xego,yego,x11,y11,x22,y22),
                              interval=100, blit=True)
    plt.show()
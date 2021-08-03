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
Q=sparse.diags([1,1,1,1,1,1,1,1,1])
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
def calmpc(x0,xr,u0,ur,x1,x2,xx):    
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
    for k in range(N):
   
        x1_new=states(-const_vel,x1,k*0.01)
        x2_new=states(const_vel,x2,k*0.01)
    
        objective += quad_form(x[:,k] - xr, Q) + quad_form(u[:,k]-ur, R)
        constraints += [x[:,k+1] == Ad@x[:,k] + Bd@u[:,k]]
        if xx>=85 :
            pp=np.sqrt(258.57-np.square(xx-89.04))-12
            constraints += [x[1,k]== pp]
        # if xx>=104.1:
        #     pp=pp+1
        #     constraints += [x[1,k]== pp]
        # constraints += [x_init==x0]
        # constraints += [x[1,k]<=cross[1]+d1e]
    objective += quad_form(x[:,N] - xr, QN)

    prob = Problem(Minimize(objective), constraints)
    start = time.time()
    prob.solve()
    elapsed_time = time.time() - start
    # print("calc time:{0}".format(elapsed_time) + "[sec]")
    # print(prob.value)
    return u,x,prob.value,x1_new,x2_new


def GetListFromMatrix(x):
    return np.array(x).flatten().tolist()

def Main():
    #Starting point of the two vehicles
    x1=np.array([105,43.455867767333984, 0.300000])
    x2=np.array([97.56910705566406,-44.676116943359375, 0.300000])
    
    # Initial and Reference States and input
    u0= np.array([2,0,0])
    # x0=np.array([38.714229583740234,3.2649950981140137,0.300000,0,0,0,0,0,0])
    x0=np.array([70,3.2649950981140137,0.300000,0,0,0,0,0,0])
    xr=np.array([139.53480529785156,3.2649950981140137,0.300000,5,0,0,0,0,0]) #Longer straight Line
    ur=np.array([0,0,0])
    ego=[]
    # veh1=[]
    # veh2=[]
    xx=38.714229583740234
    for i in range(100):
        ustar,xstar,cost,x1,x2=calmpc(x0,xr,u0,ur,x1,x2,xx) 
        xx=xstar.value[0,0]
        print(i," ",xstar.value[0,0],' ',xstar.value[1,0])
        x0 = Ad.dot(x0) + Bd.dot(ustar[:,0].value)
        ego.append(xstar.value[:3, 0])
        # veh1.append(x1)
        # veh2.append(x2)
    ego=np.array(ego)
    # veh1=np.array(veh1)
    # veh2=np.array(veh2) 
    # return ego,veh1,veh2
    return ego
  
def animate(i,xego,yego):
    line.set_data(xego[:i],yego[:i])
    return line,
    

if __name__ == '__main__':
    ego=Main()
    xego=ego[:,0]
    yego=ego[:,1]
    np.save('xego',xego)
    np.save('yego',yego)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(-50, 10)
    line, = ax.plot([],[], 'bo')
    ani = animation.FuncAnimation(fig, animate, frames=len(xego), fargs=(xego,yego),
                              interval=10, blit=True)
    plt.show()
    
#Start of intersection (x,y,z) = (85.46109008789062,3.66758394241333,3.8200066089630127)
#end of intersection (x,y,z) = (105.68384552001953,-8.370306015014648,3.444342613220215)
# (x - 88.32)²  +  (y + 14.535)²  =  339.5
## (x−89.04)2+(y+12)2=258.57 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:59:54 2021

@author: nsh1609
"""
#######MPC framework without details###################
import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import scipy.signal as ssig
from scipy.linalg import block_diag
#States of MPC
# pe = vehicle.get_location()
# ve = vehicle.get_velocity()
# ae = vehicle.get_acceleration()
# xe=np.array([pe.x,pe.y,pe.z,ve.x,ve.y,ve.z,ae.x,ae.y,ae.z])
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
dt=0.2 #Time step taken as 0.5
Sysd= ssig.cont2discrete((Ac,Bc,Cc,Dc), dt, method='zoh', alpha=None)
Ad=sparse.csc_matrix(Sysd[0])
Bd=sparse.csc_matrix(Sysd[1])
[nx, nu] = Bd.shape
#Objective Function
Q=sparse.diags([0,0,0,1,1,1,1,1,1])
QN=sparse.diags([1,1,1,1,1,1,1,1,1])
R=sparse.eye(3)

#Initial and Reference States and input
x0=np.array([38.714229583740234,3.2649950981140137,0.300000,0,0,0,0,0,0])
xr=np.array([118.2643051147461,3.2649950981140137,0.300000,0,0,0,0,0,0])
ur=np.array([0,0,0])
# Prediction horizon
N = 10

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])
# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
# lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
# uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

# Simulate in closed loop
nsim = 15
for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    x0 = Ad.dot(x0) + Bd.dot(ctrl)

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)




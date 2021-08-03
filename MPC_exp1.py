#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:57:43 2021

@author: nsh1609
"""
import numpy as np
from scipy.linalg import block_diag

class MPC(object):
    def MPC(self):
        self.N=100
        self.Q=block_diag(0,1,1)
        self.R=1
        self.S=0
        self.P=block_diag(1,1,1,1,1,1,1,1,1)
        # self.A=np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.A=block_diag(1,1,1,1,1,1,1,1,1)
        self.B=np.array([[0],[0],[0],[0],[0],[0],[1],[1],[1]])
        self.N=100
        self._initDynamics()
        ###############input states#####################
        pe = self.vehicle.get_location()
        ve = self.vehicle.get_velocity()
        ae = self.vehicle.get_acceleration()
        self.xe=np.array([pe.x,pe.y,pe.z,ve.x,ve.y,ve.z,ae.x,ae.y,ae.z])
        jerk=ae-aold #jerk
        aold=ae
        #####Control Command###
        self.u=np.array([jerk.x,jerk.y,jerk.z])
        Qbar = [];
        Rbar = [];
        RbarD = [];
        Sx = [];
        CAB = [];
        
        SU=[];
        Su=[];
        for ii in range(1, N):
            Qbar = block_diag(Qbar,Q);
            Rbar = block_diag(Rbar,R);
            RbarD = block_diag(RbarD,RD);
            Sx = np.array([[Sx][C*A^ii]]);
            CAB = np.array([[CAB] [C*A^(ii-1)*B]]);
        for ii in range(1, N):
            for jj in range(1, ii):  
                Su(ii,jj)=sum(CAB(1:ii-jj+1));
        Su;
        Su1=  Su(:,1);
        %%
        LL = tril(ones(N));
        H = 2*(LL'*Rbar*LL+RbarD+Su'*Qbar*Su);
        Fu = 2*(diag(LL'*Rbar')'+Su1'*Qbar*Su)';  %Note the trick on Rbar - u(-1) is really a scalar
        Fr = -2*(Qbar*Su)';
        Fx = 2*(Sx'*Qbar*Su)';
        %%
        G =  [tril(ones(N));-tril(ones(N))];
        W0 = 12*ones(2*N,1);
        S =  zeros(2*N,4);
        %%
        %Initial state
        X = [0;0;0;0];
        T =  70;
        %Considering xobj to be traveling in a straight line(r=xobj)
        r= 5*ones(1,1000);%:0.01:50;
        %r = square([1:T+N+1]/6);
        Z = zeros(4,1);
        U = 0;
        %%
        options = optimoptions('quadprog');
        options.Display = 'none';
        Xact=[];
        for ii = 1:T-1
            Xact(ii,:) = X;
            f = Fx*X+Fu*U+Fr*(r(ii:ii+N-1))';  %Subtracting offset value 5 from xobj
            W = W0+[ones(N,1)*-U;ones(N,1)*U];
            Z = quadprog(H,f,G,W+S*X,[],[],[],[],[],options);
            Uopt(ii) = U + Z(1);    
            U = Uopt(ii);
            X = A*X+B*U;
        end
        Xact(ii+1,:) = X;
        %%
        y = C*Xact';
                
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:34:00 2020

@author: vittorio
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class TwoRooms:
    class Environment:
        def __init__(self):
            self.Nc = 10 #Time steps required to bring drone to base when it crashes
            self.P_WIND = 0.1 #Gust of wind probability
            #IDs of elements in map
            self.FREE = 0
            self.TREE = 1
            self.SHOOTER = 2
            self.REWARD = 3
            self.BASE = 4
            #Actions index
            self.NORTH = 0
            self.SOUTH = 1
            self.EAST = 2
            self.WEST = 3
            self.HOVER = 4
            
            def GenerateMap(self):
                
                mapsize = [3, 7]
                grid = np.zeros((mapsize[0], mapsize[1]))
            
                #define obstacles
                grid[0,3] = self.TREE
                grid[mapsize[0]-1,3]= self.TREE

                #count trees
                ntrees=0;
                trees = np.empty((0,2),int)
                for i in range(0,mapsize[0]):
                    for j in range(0,mapsize[1]):
                        if grid[i,j]== self.TREE:
                            trees = np.append(trees, [[j, i]], 0)
                            ntrees += 1

                #R1
                reward = np.array([mapsize[1]-1, mapsize[0]-1])
                grid[reward[1],reward[0]] = self.REWARD

                #base
                base = np.array([0, 0])
                grid[base[1],base[0]] = self.BASE
                        
                return grid
            
            self.map = GenerateMap(self)
        
            def GenerateStateSpace(self):

                stateSpace = np.empty((0,2),int)

                for m in range(0,self.map.shape[0]):
                    for n in range(0,self.map.shape[1]):
                        if self.map[m,n] != self.TREE:
                            stateSpace = np.append(stateSpace, [[m, n]], 0)
                        
                return stateSpace
            
            self.stateSpace = GenerateStateSpace(self)
            

        def BaseStateIndex(self):

            K = self.stateSpace.shape[0];
    
            for i in range(0,self.map.shape[0]):
                for j in range(0,self.map.shape[1]):
                    if self.map[i,j]==self.BASE:
                        m=i
                        n=j
                        break
            
            for i in range(0,K):
                if self.stateSpace[i,0]==m and self.stateSpace[i,1]==n:
                    stateIndex = i
                    break
    
            return stateIndex

        def RStateIndex(self):
        
            K = self.stateSpace.shape[0];
    
            for i in range(0,self.map.shape[0]):
                for j in range(0,self.map.shape[1]):
                    if self.map[i,j]==self.REWARD:
                        m=i
                        n=j
                        break
            
            for i in range(0,K):
                if self.stateSpace[i,0]==m and self.stateSpace[i,1]==n:
                    stateIndex = i
                    break
    
            return stateIndex

        def FindStateIndex(self, value):
    
            K = self.stateSpace.shape[0];
            stateIndex = 0
    
            for k in range(0,K):
                if self.stateSpace[k,0]==value[0] and self.stateSpace[k,1]==value[1]:
                    stateIndex = k
    
            return stateIndex
        
        def ComputeTransitionProbabilityMatrix(self):
            action_space=5
            K = self.stateSpace.shape[0]
            P = np.zeros((K,K,action_space))
            [M,N]=self.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = TwoRooms.Environment.FindStateIndex(self,array_temp)

                    if self.map[i,j] != self.TREE:

                        for u in range(0,action_space):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == self.EAST and self.map[i,j+1]!=self.TREE:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                                elif j==N-1 and u==self.EAST:
                                    comp_no=1
                            #west case
                            if j!=0:
                                if u==self.WEST and self.map[i,j-1]!=self.TREE:
                                    r=i
                                    s=j-1
                                    comp_no=0
                                elif j==0 and u==self.WEST:
                                    comp_no=1
                            #south case
                            if i!=0:
                                if u==self.SOUTH and self.map[i-1,j]!=self.TREE:
                                    r=i-1
                                    s=j
                                    comp_no=0
                                elif i==0 and u==self.SOUTH:
                                    comp_no=1
                            #north case
                            if i!=M-1:
                                if u==self.NORTH and self.map[i+1,j]!=self.TREE:
                                    r=i+1
                                    s=j
                                    comp_no=0
                                elif i==M-1 and u==self.NORTH:
                                    comp_no=1
                            #hover case
                            if u==self.HOVER:
                                r=i
                                s=j
                                comp_no=0

                            if comp_no==0:
                                array_temp = [r, s]
                                t = TwoRooms.Environment.FindStateIndex(self,array_temp)

                                # No wind case
                                P[k,t,u] = P[k,t,u]+(1-self.P_WIND)
                                base0 = TwoRooms.Environment.BaseStateIndex(self)

                                # case wind

                                #north wind
                                if s+1>N-1 or self.map[r,s+1]==self.TREE:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s+1]
                                    t = TwoRooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #north wind no hit

                                #South Wind
                                if s-1<0 or self.map[r,s-1]==self.TREE:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s-1]
                                    t=TwoRooms.Environment.FindStateIndex(self,array_temp)                                 
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #south wind no hit

                                #East Wind
                                if r+1>M-1 or self.map[r+1,s]==self.TREE:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r+1, s]
                                    t=TwoRooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #east wind no hit

                                #West Wind
                                if r-1<0 or self.map[r-1,s]==self.TREE:
                                    P[k,base0,u]=P[k,base0,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r-1, s]
                                    t=TwoRooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #west wind no hit

                            elif comp_no == 1:
                                base0=TwoRooms.Environment.BaseStateIndex(self)
                                P[k,base0,u]=1

            return P
      
    class Expert:
        def __init__(self):
            self.Environment = TwoRooms.Environment()
            self.R_STATE_INDEX = self.Environment.RStateIndex()
            self.P = self.Environment.ComputeTransitionProbabilityMatrix()
        
        def ComputeStageCostsR(self):
            action_space=5
            K = self.Environment.stateSpace.shape[0]
            G = np.zeros((K,action_space))
            [M,N]=self.Environment.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = self.Environment.FindStateIndex(array_temp)

                    if self.Environment.map[i,j] != self.Environment.TREE:

                        if k == self.R_STATE_INDEX:
                            dummy=0 #no cost
                        else:
                            for u in range(0,action_space):
                                comp_no=1;
                                # east case
                                if j!=N-1:
                                    if u == self.Environment.EAST and self.Environment.map[i,j+1]!=self.Environment.TREE:
                                        r=i
                                        s=j+1
                                        comp_no = 0
                                elif j==N-1 and u==self.Environment.EAST:
                                    comp_no=1

                                if u == self.Environment.EAST:
                                    if j==N-1 or self.Environment.map[i,j+1]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #west case
                                if j!=0:
                                    if u==self.Environment.WEST and self.Environment.map[i,j-1]!=self.Environment.TREE:
                                        r=i
                                        s=j-1
                                        comp_no=0
                                elif j==0 and u==self.Environment.WEST:
                                    comp_no=1

                                if u==self.Environment.WEST:
                                    if j==0 or self.Environment.map[i,j-1]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #south case
                                if i!=0:
                                    if u==self.Environment.SOUTH and self.Environment.map[i-1,j]!=self.Environment.TREE:
                                        r=i-1
                                        s=j
                                        comp_no=0
                                elif i==0 and u==self.Environment.SOUTH:
                                    comp_no=1

                                if u==self.Environment.SOUTH:
                                    if i==0 or self.Environment.map[i-1,j]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #north case
                                if i!=M-1:
                                    if u==self.Environment.NORTH and self.Environment.map[i+1,j]!=self.Environment.TREE:
                                        r=i+1
                                        s=j
                                        comp_no=0
                                elif i==M-1 and u==self.Environment.NORTH:
                                    comp_no=1

                                if u==self.Environment.NORTH:
                                    if i==M-1 or self.Environment.map[i+1,j]==self.Environment.TREE:
                                        G[k,u]=np.inf

                                #hover case
                                if u==self.Environment.HOVER:
                                    r=i
                                    s=j
                                    comp_no=0

                                if comp_no==0:
                                    array_temp = [r, s]

                                    G[k,u] = G[k,u]+(1-self.Environment.P_WIND) #no shot and no wind

                                    # case wind

                                    #north wind
                                    if s+1>N-1 or self.Environment.map[r,s+1]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r, s+1]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25


                                    #South Wind
                                    if s-1<0 or self.Environment.map[r,s-1]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r, s-1]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #south wind no hit

                                    #East Wind
                                    if r+1>M-1 or self.Environment.map[r+1,s]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r+1, s]

                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #east wind no hit

                                    #West Wind
                                    if r-1<0 or self.Environment.map[r-1,s]==self.Environment.TREE:
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25*self.Environment.Nc #wind causes crash
                                    else:
                                        array_temp = [r-1, s]
                                    
                                        G[k,u]=G[k,u]+self.Environment.P_WIND*0.25 #west wind no hit

                                elif comp_no == 1:
                                    dummy=0

            for l in range(0,action_space):
                G[self.R_STATE_INDEX,l]=0

            return G            

        def ValueIteration(self, G, TERMINAL_INDEX):
            action_space=5
            tol=10**(-5)
            K = G.shape[0]
            V=np.zeros((K,action_space))
            VV=np.zeros((K,2))
            I=np.zeros((K))
            Err=np.zeros((K))

            #initialization
            VV[:,0]=50
            VV[TERMINAL_INDEX,0]=0
            n=0
            Check_err=1

            while Check_err==1:
                n=n+1
                Check_err=0
                for k in range(0,K):
                    if n>1:
                        VV[:,0]=VV[0:,1]

                    if k==TERMINAL_INDEX:
                        VV[k,1]=0
                        V[k,:]=0
                    else:
                        CTG=np.zeros((action_space)) #cost to go
                        for u in range(0,action_space):
                            for j in range(0,K):
                                CTG[u]=CTG[u] + self.P[k,j,u]*VV[j,1]

                            V[k,u]=G[k,u]+CTG[u]

                        VV[k,1]=np.amin(V[k,:])
                        flag = np.where(V[k,:]==np.amin(V[k,:]))
                        I[k]=flag[0][0]

                    Err[k]=abs(VV[k,1]-VV[k,0])

                    if Err[k]>tol:
                        Check_err=1

            J_opt=VV[:,1]
            I[TERMINAL_INDEX]=self.Environment.HOVER
            u_opt = I[:]

            return J_opt,u_opt
        
        def PlotPolicy(self, u, name):
            mapsize = self.Environment.map.shape
            #count trees
            ntrees=0;
            trees = np.empty((0,2),int)
            shooters = np.empty((0,2),int)
            nshooters=0
            for i in range(0,mapsize[0]):
                for j in range(0,mapsize[1]):
                    if self.Environment.map[i,j]==self.Environment.TREE:
                        trees = np.append(trees, [[j, i]], 0)
                        ntrees += 1
                    if self.Environment.map[i,j]==self.Environment.SHOOTER:
                        shooters = np.append(shooters, [[j, i]], 0)
                        nshooters+=1

            #Reward
            RIndex=self.R_STATE_INDEX
            i_R = self.Environment.stateSpace[RIndex,0]
            j_R = self.Environment.stateSpace[RIndex,1]
            R = np.array([j_R, i_R])
            #base
            BaseIndex=self.Environment.BaseStateIndex()
            i_base = self.Environment.stateSpace[BaseIndex,0]
            j_base = self.Environment.stateSpace[BaseIndex,1]
            base = np.array([j_base, i_base])

            # Plot
            plt.figure()
            plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
            plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                     [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
            plt.plot([R[0], R[0], R[0]+1, R[0]+1, R[0]],
                     [R[1], R[1]+1, R[1]+1, R[1], R[1]],'k-')

            for i in range(0,nshooters):
                plt.plot([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                         [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'k-')

            for i in range(0,ntrees):
                plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

            plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                     [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
            plt.fill([R[0], R[0], R[0]+1, R[0]+1, R[0]],
                     [R[1], R[1]+1, R[1]+1, R[1], R[1]],'y')

            for i in range(0,nshooters):
                plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                         [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

            for i in range(0,ntrees):
                plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k')

            plt.text(base[0]+0.5, base[1]+0.5, 'B')
            plt.text(R[0]+0.5, R[1]+0.5, 'R')
            for i in range(0,nshooters):
                plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

            for s in range(0,u.shape[0]):
                if u[s] == self.Environment.NORTH:
                    txt = u'\u2191'
                elif u[s] == self.Environment.SOUTH:
                    txt = u'\u2193'
                elif u[s] == self.Environment.EAST:
                    txt = u'\u2192'
                elif u[s] == self.Environment.WEST:
                    txt = u'\u2190'
                elif u[s] == self.Environment.HOVER:
                    txt = u'\u2715'
                plt.text(self.Environment.stateSpace[s,1]+0.3, self.Environment.stateSpace[s,0]+0.5, txt, fontsize=20)
            
            plt.savefig(name, format='eps')
            
        
        def ComputeFlatPolicy(self):
# =============================================================================
# Compute the expert's policy using value iteration and plot the result
# =============================================================================
            GR = TwoRooms.Expert.ComputeStageCostsR(self)
            [JR,UR] = TwoRooms.Expert.ValueIteration(self,GR,self.R_STATE_INDEX)
            UR = UR.reshape(len(UR),1)
            TwoRooms.Expert.PlotPolicy(self, UR, 'Figures/FiguresExpert/Expert_R.eps')
            
            return UR
        
        def StateSpace(self):
            stateSpace = np.empty((0,3),int)

            for m in range(0,self.Environment.map.shape[0]):
                for n in range(0,self.Environment.map.shape[1]):
                    for k in range(1):
                        if self.Environment.map[m,n] != self.Environment.TREE:
                            stateSpace = np.append(stateSpace, [[m, n, k]], 0)
                        
            return stateSpace
        
        def FindStateIndex(self, value):
            
            stateSpace = TwoRooms.Expert.StateSpace(self)
            K = stateSpace.shape[0];
            stateIndex = 0
    
            for k in range(0,K):
                if stateSpace[k,0]==value[0,0] and stateSpace[k,1]==value[0,1] and stateSpace[k,2]==value[0,2]:
                    stateIndex = k
    
            return stateIndex
        
        def generate_pi_hi(self):
            stateSpace = TwoRooms.Expert.StateSpace(self)
            pi_hi = np.empty((0,1),int)
            for i in range(len(stateSpace)):
                if stateSpace[i,1]<3:
                    pi_hi = np.append(pi_hi, [[0]], 0)
                elif stateSpace[i,1]>=3:
                    pi_hi = np.append(pi_hi, [[1]], 0)
                 
            pi_hi_encoded = np.zeros((len(pi_hi), pi_hi.max()+1))
            pi_hi_encoded[np.arange(len(pi_hi)),pi_hi[:,0]] = 1
            
            return pi_hi_encoded
        
        def generate_pi_lo(self, Uopt, pi_hi, n_op):
            stateSpace = TwoRooms.Expert.StateSpace(self)
            pi_lo = np.empty((0,1),int)

            for i in range(len(stateSpace)):
                if pi_hi[i,n_op]==1:
                    pi_lo = np.append(pi_lo, [[int(Uopt[i,0])]], 0)
                else:
                    pi_lo = np.append(pi_lo, [[self.Environment.HOVER]], 0)
                        
            pi_lo_encoded = np.zeros((len(pi_lo), pi_lo.max()+1,1))
            pi_lo_encoded[np.arange(len(pi_lo)),pi_lo[:,0],0] = 1
            
            return pi_lo_encoded
        
        def generate_pi_b(self, pi_hi, n_op):
            stateSpace = TwoRooms.Expert.StateSpace(self)
            pi_b = np.empty((0,1),int)
            for i in range(len(stateSpace)):
                if pi_hi[i,n_op]==1:
                    pi_b = np.append(pi_b, [[0]], 0)
                else:
                    pi_b = np.append(pi_b, [[1]], 0)

            pi_b_encoded = np.zeros((len(pi_b), pi_b.max()+1, 1))
            pi_b_encoded[np.arange(len(pi_b)),pi_b[:,0],0] = 1
            
            return pi_b_encoded
        
        def HierarchicalPolicy(self):
# =============================================================================
# This function generates a hierarchical policy for expert starting from 
# the solution obtained using value-iteration. The policy is arbitrarily obtained using 
# functions already defined.
# =============================================================================
            U = TwoRooms.Expert.ComputeFlatPolicy(self)
            pi_hi = TwoRooms.Expert.generate_pi_hi(self)
            pi_lo1 = TwoRooms.Expert.generate_pi_lo(self, U, pi_hi, 0)
            pi_lo2 = TwoRooms.Expert.generate_pi_lo(self, U, pi_hi, 1)
            pi_lo = np.concatenate((pi_lo1, pi_lo2), 2)
            pi_b1 = TwoRooms.Expert.generate_pi_b(self, pi_hi, 0)
            pi_b2 = TwoRooms.Expert.generate_pi_b(self, pi_hi, 1)
            pi_b = np.concatenate((pi_b1, pi_b2), 2)
                        
            return pi_hi, pi_lo, pi_b
        
        def PlotOptions(self, pi_hi, name):
            mapsize = self.Environment.map.shape
            #count trees
            ntrees=0;
            trees = np.empty((0,2),int)
            shooters = np.empty((0,2),int)
            nshooters=0
            for i in range(0,mapsize[0]):
                for j in range(0,mapsize[1]):
                    if self.Environment.map[i,j]==self.Environment.TREE:
                        trees = np.append(trees, [[j, i]], 0)
                        ntrees += 1
                    if self.Environment.map[i,j]==self.Environment.SHOOTER:
                        shooters = np.append(shooters, [[j, i]], 0)
                        nshooters+=1

            #Reward
            RIndex=self.R_STATE_INDEX
            i_R = self.Environment.stateSpace[RIndex,0]
            j_R = self.Environment.stateSpace[RIndex,1]
            R = np.array([j_R, i_R])
            #base
            BaseIndex=self.Environment.BaseStateIndex()
            i_base = self.Environment.stateSpace[BaseIndex,0]
            j_base = self.Environment.stateSpace[BaseIndex,1]
            base = np.array([j_base, i_base])

            # Plot
            plt.figure()
            plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
            plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                     [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
            plt.plot([R[0], R[0], R[0]+1, R[0]+1, R[0]],
                     [R[1], R[1]+1, R[1]+1, R[1], R[1]],'k-')


            for i in range(0,nshooters):
                plt.plot([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                         [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'k-')

            for i in range(0,ntrees):
                plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

            plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                     [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
            plt.fill([R[0], R[0], R[0]+1, R[0]+1, R[0]],
                     [R[1], R[1]+1, R[1]+1, R[1], R[1]],'y')

            for i in range(0,nshooters):
                plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                         [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

            for i in range(0,ntrees):
                plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                         [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k')

            plt.text(base[0]+0.5, base[1]+0.5, 'B')
            plt.text(R[0]+0.5, R[1]+0.5, 'R')
            for i in range(0,nshooters):
                plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')
                
            for s in range(0,len(pi_hi)):
                if pi_hi[s]==0:
                    c = 'c'
                elif pi_hi[s]==1:
                    c = 'lime'
                elif pi_hi[s]==2:
                    c = 'y'    
                plt.fill([self.Environment.stateSpace[s,1], self.Environment.stateSpace[s,1], self.Environment.stateSpace[s,1]+0.9, 
                          self.Environment.stateSpace[s,1]+0.9, self.Environment.stateSpace[s,1]],
                         [self.Environment.stateSpace[s,0], self.Environment.stateSpace[s,0]+0.9, self.Environment.stateSpace[s,0]+0.9, 
                          self.Environment.stateSpace[s,0], self.Environment.stateSpace[s,0]],c)            
 
            
            plt.savefig(name, format='eps')
       
            
        class Plot:
            def __init__(self, pi_hi, pi_lo, pi_b):
                self.pi_hi = pi_hi
                self.pi_lo = pi_lo
                self.pi_b = pi_b
                self.expert = TwoRooms.Expert()
                
            def PlotHierachicalPolicy(self, NameFilePI_HI, NameFilePI_LO, NameFilePI_B):
                pi_hi = np.argmax(self.pi_hi,1)
                pi_b = np.argmax(self.pi_b,1)
                pi_lo = np.argmax(self.pi_lo,1)
                option_space = pi_lo.shape[1]
                psi_space = 1
                PI_HI = np.empty((0,psi_space,1),int)
                PI_B = []
                PI_LO = []
                for i in range(option_space):
                    PI_LO.append(np.empty((0,psi_space,1),int))
                for i in range(2):
                    PI_B.append(np.empty((0,psi_space,1),int))
                for i in range(0,len(pi_lo),psi_space):
                    pi_hi_temp = pi_hi[i:i+psi_space].reshape(1,psi_space,1)
                    PI_HI = np.append(PI_HI, pi_hi_temp, 0)
                    for option in range(option_space):
                        pi_b_temp = pi_b[i:i+psi_space,option].reshape(1,psi_space,1)
                        PI_B[option] = np.append(PI_B[option], pi_b_temp, 0)
                        pi_lo_temp = pi_lo[i:i+psi_space,option].reshape(1,psi_space,1)
                        PI_LO[option] = np.append(PI_LO[option], pi_lo_temp, 0)
            
                for option in range(option_space):
                    for psi in range(psi_space):
                        U = PI_LO[option]
                        self.expert.PlotPolicy(U[:,psi,0], NameFilePI_LO.format(option, psi))
                        B = PI_B[option]
                        self.expert.PlotOptions(B[:,psi,0], NameFilePI_B.format(option, psi))
                        
                for psi in range(psi_space):
                    self.expert.PlotOptions(PI_HI[:,psi,0], NameFilePI_HI.format(psi))
                                 
                
        class Simulation:
            def __init__(self, pi_hi, pi_lo, pi_b):
                option_space = pi_hi.shape[1]
                self.option_space = option_space
                self.mu = np.ones(option_space)*np.divide(1,option_space)
                self.zeta = 0.0001
                self.Environment = TwoRooms.Environment()
                self.initial_state = self.Environment.BaseStateIndex()
                self.P = self.Environment.ComputeTransitionProbabilityMatrix()
                self.stateSpace = self.Environment.stateSpace
                self.R_STATE_INDEX = self.Environment.RStateIndex()
                self.pi_hi = TwoRooms.PI_HI(pi_hi)
                self.pi_lo = TwoRooms.PI_LO(pi_lo)
                self.pi_b = TwoRooms.PI_B(pi_b)
                                
            def HierarchicalStochasticSampleTrajMDP(self, max_epoch_per_traj, number_of_trajectories, seed):
                traj = [[None]*1 for _ in range(number_of_trajectories)]
                control = [[None]*1 for _ in range(number_of_trajectories)]
                Option = [[None]*1 for _ in range(number_of_trajectories)]
                Termination = [[None]*1 for _ in range(number_of_trajectories)]
                reward = np.empty((0,0),int)
                psi_evolution = [[None]*1 for _ in range(number_of_trajectories)]
                np.random.seed(seed)
    
                for t in range(0,number_of_trajectories):
        
                    x = np.empty((0,0),int)
                    x = np.append(x, self.initial_state)
                    u_tot = np.empty((0,0))
                    o_tot = np.empty((0,0),int)
                    b_tot = np.empty((0,0),int)
                    psi_tot = np.empty((0,0),int)
                    psi = 0
                    psi_tot = np.append(psi_tot, psi)
                    r=0
        
                    # Initial Option
                    prob_o = self.mu
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[0]):
                        prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled))
                    o_tot = np.append(o_tot,o)
        
                    # Termination
                    state_partial = self.stateSpace[x[0],:].reshape(1,len(self.stateSpace[x[0],:]))
                    state = np.concatenate((state_partial,[[psi]]),1)
                    prob_b = self.pi_b.Policy(state, o)
                    prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                    for i in range(1,prob_b_rescaled.shape[1]):
                        prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                    draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                    b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                    b_tot = np.append(b_tot,b)
                    if b == 1:
                        b_bool = True
                    else:
                        b_bool = False
        
                    o_prob_tilde = np.empty((1,self.option_space))
                    if b_bool == True:
                        o_prob_tilde = self.pi_hi.Policy(state)
                    else:
                        o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                        o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                    prob_o = o_prob_tilde
                    prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                    for i in range(1,prob_o_rescaled.shape[1]):
                        prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                    draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                    o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                    o_tot = np.append(o_tot,o)
        
                    for k in range(0,max_epoch_per_traj):
                        state_partial = self.stateSpace[x[k],:].reshape(1,len(self.stateSpace[x[k],:]))
                        state = np.concatenate((state_partial,[[psi]]),1)
                        # draw action
                        prob_u = self.pi_lo.Policy(state,o)
                        prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                        for i in range(1,prob_u_rescaled.shape[1]):
                            prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                        draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                        u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
                        # given action, draw next state
                        x_k_possible=np.where(self.P[x[k],:,int(u)]!=0)
                        prob = self.P[x[k],x_k_possible[0][:],int(u)]
                        prob_rescaled = np.divide(prob,np.amin(prob))
            
                        for i in range(1,prob_rescaled.shape[0]):
                            prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
                        draw=np.divide(np.random.rand(),np.amin(prob))
                        index_x_plus1=np.amin(np.where(draw<prob_rescaled))
                        x = np.append(x, x_k_possible[0][index_x_plus1])
                        u_tot = np.append(u_tot,u)
            
                        if (x[k+1] == self.R_STATE_INDEX):
                            r = r + 1 
                            x[k+1] = np.random.randint(0,len(self.stateSpace))
            
                        # Randomly update the reward
                        psi_tot = np.append(psi_tot, psi)
            
                        # Select Termination
                        # Termination
                        state_plus1_partial = self.stateSpace[x[k+1],:].reshape(1,len(self.stateSpace[x[k+1],:]))
                        state_plus1 = np.concatenate((state_plus1_partial,[[psi]]),1)
                        prob_b = self.pi_b.Policy(state_plus1,o)
                        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
                        for i in range(1,prob_b_rescaled.shape[1]):
                            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
                        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
                        b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
                        b_tot = np.append(b_tot,b)
                        if b == 1:
                            b_bool = True
                        else:
                            b_bool = False
        
                        o_prob_tilde = np.empty((1,self.option_space))
                        if b_bool == True:
                            o_prob_tilde = self.pi_hi.Policy(state_plus1)
                        else:
                            o_prob_tilde[0,:] = self.zeta/self.option_space*np.ones((1,self.option_space))
                            o_prob_tilde[0,o] = 1 - self.zeta + self.zeta/self.option_space
            
                        prob_o = o_prob_tilde
                        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
                        for i in range(1,prob_o_rescaled.shape[1]):
                            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
                        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
                        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
                        o_tot = np.append(o_tot,o)
            
        
                    traj[t] = x
                    control[t]=u_tot
                    Option[t]=o_tot
                    Termination[t]=b_tot            
                    reward = np.append(reward,r)
                    psi_evolution[t] = psi_tot  

        
                return traj, control, Option, Termination, psi_evolution, reward
            
            def HILVideoSimulation(self,u,states,o,name_video):
                
                mapsize = self.Environment.map.shape
                #count trees
                ntrees=0;
                trees = np.empty((0,2),int)
                shooters = np.empty((0,2),int)
                nshooters=0
                for i in range(0,mapsize[0]):
                    for j in range(0,mapsize[1]):
                        if self.Environment.map[i,j]==self.Environment.TREE:
                            trees = np.append(trees, [[j, i]], 0)
                            ntrees += 1
                        if self.Environment.map[i,j]==self.Environment.SHOOTER:
                            shooters = np.append(shooters, [[j, i]], 0)
                            nshooters+=1

                #Reward
                RIndex=self.R_STATE_INDEX
                i_R = self.Environment.stateSpace[RIndex,0]
                j_R = self.Environment.stateSpace[RIndex,1]
                R = np.array([j_R, i_R])
                #base
                BaseIndex=self.Environment.BaseStateIndex()
                i_base = self.Environment.stateSpace[BaseIndex,0]
                j_base = self.Environment.stateSpace[BaseIndex,1]
                base = np.array([j_base, i_base])

                # Plot
                fig = plt.figure()
                plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
                plt.plot([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                         [base[1], base[1]+1, base[1]+1, base[1], base[1]],'k-')
                plt.plot([R[0], R[0], R[0]+1, R[0]+1, R[0]],
                         [R[1], R[1]+1, R[1]+1, R[1], R[1]],'k-')

                for i in range(0,nshooters):
                    plt.plot([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                             [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'k-')

                for i in range(0,ntrees):
                    plt.plot([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                             [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k-')

                plt.fill([base[0], base[0], base[0]+1, base[0]+1, base[0]],
                         [base[1], base[1]+1, base[1]+1, base[1], base[1]],'r')
                plt.fill([R[0], R[0], R[0]+1, R[0]+1, R[0]],
                         [R[1], R[1]+1, R[1]+1, R[1], R[1]],'y')

                for i in range(0,nshooters):
                    plt.fill([shooters[i,0], shooters[i,0], shooters[i,0]+1, shooters[i,0]+1, shooters[i,0]],
                             [shooters[i,1], shooters[i,1]+1, shooters[i,1]+1, shooters[i,1], shooters[i,1]],'c')

                for i in range(0,ntrees):
                    plt.fill([trees[i,0], trees[i,0], trees[i,0]+1, trees[i,0]+1, trees[i,0]],
                             [trees[i,1], trees[i,1]+1, trees[i,1]+1, trees[i,1], trees[i,1]],'k')

                plt.text(base[0]+0.5, base[1]+0.5, 'B')
                plt.text(R[0]+0.5, R[1]+0.5, 'R')
                for i in range(0,nshooters):
                    plt.text(shooters[i,0]+0.5, shooters[i,1]+0.5, 'S')

                ims = []
                for s in range(0,len(u)):
                    if u[s] == self.Environment.NORTH:
                        txt = u'\u2191'
                    elif u[s] == self.Environment.SOUTH:
                        txt = u'\u2193'
                    elif u[s] == self.Environment.EAST:
                        txt = u'\u2192'
                    elif u[s] == self.Environment.WEST:
                        txt = u'\u2190'
                    elif u[s] == self.Environment.HOVER:
                        txt = u'\u2715'
                    if o[s+1]==0:
                        c = 'c'
                    elif o[s+1]==1:
                        c = 'lime'
                    elif o[s+1]==2:
                        c = 'y'         
                    im1 = plt.text(self.Environment.stateSpace[states[s],1]+0.3, self.Environment.stateSpace[states[s],0]+0.1, txt, fontsize=20, backgroundcolor=c)
                    ims.append([im1])
        
                ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                                repeat_delay=2000)
                ani.save(name_video)
                

    class PI_LO:
        def __init__(self, pi_lo):
            self.pi_lo = pi_lo
            self.expert = TwoRooms.Expert()
                
        def Policy(self, state, option):
            stateID = self.expert.FindStateIndex(state)
            prob_distribution = self.pi_lo[stateID,:,option]
            prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
            return prob_distribution
            
    class PI_B:
        def __init__(self, pi_b):
            self.pi_b = pi_b
            self.expert = TwoRooms.Expert()
                
        def Policy(self, state, option):
            stateID = self.expert.FindStateIndex(state)
            prob_distribution = self.pi_b[stateID,:,option]
            prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
            return prob_distribution
            
    class PI_HI:
        def __init__(self, pi_hi):
            self.pi_hi = pi_hi
            self.expert = TwoRooms.Expert()
                
        def Policy(self, state):
            stateID = self.expert.FindStateIndex(state)
            prob_distribution = self.pi_hi[stateID,:]
            prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
            return prob_distribution
        
    
    
                 
            
            
            
            
            
            
            
            
            
            
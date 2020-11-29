#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:29:37 2020

@author: vittorio
"""

import numpy as np
import World

def ProcessData(traj,control,psi,stateSpace):
    Xtr = np.empty((1,0),int)
    inputs = np.empty((3,0),int)

    for i in range(len(traj)):
        Xtr = np.append(Xtr, control[i][:])
        inputs = np.append(inputs, np.transpose(np.concatenate((stateSpace[traj[i][:-1],:], psi[i][:-1].reshape(len(psi[i])-1,1)),1)), axis=1) 
    
    labels = Xtr.reshape(len(Xtr),1)
    TrainingSet = np.transpose(inputs) 
    
    return labels, TrainingSet

class PI_LO:
    def __init__(self, pi_lo):
        self.pi_lo = pi_lo
        self.expert = World.TwoRewards.Expert()
                
    def Policy(self, stateID, option):
        prob_distribution = self.pi_lo[stateID,:,option]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution
            
class PI_B:
    def __init__(self, pi_b):
        self.pi_b = pi_b
        self.expert = World.TwoRewards.Expert()
                
    def Policy(self, stateID, option):
        prob_distribution = self.pi_b[stateID,:,option]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution
            
class PI_HI:
    def __init__(self, pi_hi):
        self.pi_hi = pi_hi
        self.expert = World.TwoRewards.Expert()
                
    def Policy(self, stateID):
        prob_distribution = self.pi_hi[stateID,:]
        prob_distribution = prob_distribution.reshape(1,len(prob_distribution))
                
        return prob_distribution

class OnlineHIL:
    def __init__(self, TrainingSet, Labels, option_space):
        self.TrainingSet = TrainingSet
        self.Labels = Labels
        self.option_space = option_space
        self.size_input = TrainingSet.shape[1]
        self.action_space = int(np.max(Labels)+1)
        self.termination_space = 2
        self.zeta = 0.0001
        self.mu = np.ones(option_space)*np.divide(1,option_space)
        self.environment = World.TwoRewards.Environment()
        self.expert = World.TwoRewards.Expert()
        
    def FindStateIndex(self, value):
        stateSpace = self.expert.StateSpace()
        K = stateSpace.shape[0];
        stateIndex = 0
    
        for k in range(0,K):
            if stateSpace[k,0]==value[0,0] and stateSpace[k,1]==value[0,1] and stateSpace[k,2]==value[0,2]:
                stateIndex = k
    
        return stateIndex
    
    def Pi_hi(ot, Pi_hi_parameterization, state):
        Pi_hi = Pi_hi_parameterization(state)
        o_prob = Pi_hi[0,ot]
        return o_prob

    def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
        if b == True:
            o_prob_tilde = OnlineHIL.Pi_hi(ot, Pi_hi_parameterization, state)
        elif ot == ot_past:
            o_prob_tilde = 1-zeta+np.divide(zeta,option_space)
        else:
            o_prob_tilde = np.divide(zeta,option_space)
        
        return o_prob_tilde

    def Pi_lo(a, Pi_lo_parameterization, state, ot):
        Pi_lo = Pi_lo_parameterization(state, ot)
        a_prob = Pi_lo[0,int(a)]
    
        return a_prob

    def Pi_b(b, Pi_b_parameterization, state, ot):
        Pi_b = Pi_b_parameterization(state, ot)
        if b == True:
            b_prob = Pi_b[0,1]
        else:
            b_prob = Pi_b[0,0]
        return b_prob
    
    def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
        Pi_hi_eval = np.clip(OnlineHIL.Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space),0.0001,1)
        Pi_lo_eval = np.clip(OnlineHIL.Pi_lo(a, Pi_lo_parameterization, state, ot),0.0001,1)
        Pi_b_eval = np.clip(OnlineHIL.Pi_b(b, Pi_b_parameterization, state, ot_past),0.0001,1)
        output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
        return output
        
    def initialize_pi_hi(self):
        stateSpace = self.expert.StateSpace()
        pi_hi = np.empty((0,self.option_space))
        for i in range(len(stateSpace)):
            prob_temp = np.random.uniform(0,1,self.option_space)
            prob_temp = np.divide(prob_temp, np.sum(prob_temp)).reshape(1,len(prob_temp))
            pi_hi = np.append(pi_hi, prob_temp, axis=0)
            
        return pi_hi
    
    def initialize_pi_lo(self):
        stateSpace = self.expert.StateSpace()
        pi_lo = np.zeros((len(stateSpace),self.action_space, self.option_space))
        for i in range(len(stateSpace)):
            for o in range(self.option_space):
                prob_temp = np.random.uniform(0,1,self.action_space)
                prob_temp = np.divide(prob_temp, np.sum(prob_temp))
                pi_lo[i,:,o] = prob_temp
                
        return pi_lo
    
    def initialize_pi_b(self):
        stateSpace = self.expert.StateSpace()
        pi_b = np.zeros((len(stateSpace),self.termination_space, self.option_space))
        for i in range(len(stateSpace)):
            for o in range(self.option_space):
                prob_temp = np.random.uniform(0,1,self.termination_space)
                prob_temp = np.divide(prob_temp, np.sum(prob_temp))
                pi_b[i,:,o] = prob_temp
                
        return pi_b
    
    def TrainingSetID(self):
        TrainingSetID = np.empty((0,1))
        for i in range(len(self.TrainingSet)):
            ID = OnlineHIL.FindStateIndex(self,self.TrainingSet[i,:].reshape(1,self.size_input))
            TrainingSetID = np.append(TrainingSetID, [[ID]], axis=0)
            
        return TrainingSetID
    
    def UpdatePiHi(self, Old_pi_hi, phi_h):
        New_pi_hi = np.zeros((Old_pi_hi.shape[0], Old_pi_hi.shape[1]))
        stateSpace = self.expert.StateSpace()
        temp_theta = np.zeros((1,self.option_space))
            
        for stateID in range(len(stateSpace)):
            for option in range(self.option_space):
                
                if np.sum(phi_h[:,1,:,:,stateID,:,:,0])==0:
                    temp_theta[0,option] = Old_pi_hi[stateID,option]
                else:
                    temp_theta[0,option] = np.clip(np.divide(np.sum(phi_h[:,1,option,:,stateID,:,:,0]),np.sum(phi_h[:,1,:,:,stateID,:,:,0])),0,1)
                    
                    
            temp_theta = np.divide(temp_theta, np.sum(temp_theta))
            New_pi_hi[stateID,:] = temp_theta
        
        return New_pi_hi
    
    def UpdatePiLo(self, Old_pi_lo, phi_h):
        New_pi_lo = np.zeros((Old_pi_lo.shape[0], Old_pi_lo.shape[1], Old_pi_lo.shape[2]))
        stateSpace = self.expert.StateSpace()
        temp_theta = np.zeros((1,self.action_space))
        
        for option in range(self.option_space):
            for stateID in range(len(stateSpace)):
                for action in range(self.action_space):
                
                    if np.sum(phi_h[:,:,option,:,stateID,:,:,0])==0:
                        temp_theta[0,action] = Old_pi_lo[stateID,action,option]
                    else:
                        temp_theta[0,action] = np.clip(np.divide(np.sum(phi_h[:,:,option,action,stateID,:,:,0]),np.sum(phi_h[:,:,option,:,stateID,:,:,0])),0,1)
                    
                    
                temp_theta = np.divide(temp_theta, np.sum(temp_theta))
                New_pi_lo[stateID,:,option] = temp_theta
        
        return New_pi_lo
    
    def UpdatePiB(self, Old_pi_b, phi_h):
        New_pi_b = np.zeros((Old_pi_b.shape[0], Old_pi_b.shape[1], Old_pi_b.shape[2]))
        stateSpace = self.expert.StateSpace()
        temp_theta = np.zeros((1,self.termination_space))
            
        for option in range(self.option_space):
            for stateID in range(len(stateSpace)):
                for termination_boolean in range(self.termination_space):
                
                    if np.sum(phi_h[option,:,:,:,stateID,:,:,0])==0:
                        temp_theta[0,termination_boolean] = Old_pi_b[stateID,termination_boolean,option]
                    else:
                        temp_theta[0,termination_boolean] = np.clip(np.divide(np.sum(phi_h[option,termination_boolean,:,:,stateID,:,:,0]),np.sum(phi_h[option,:,:,:,stateID,:,:,0])),0,1)
                           
                temp_theta = np.divide(temp_theta, np.sum(temp_theta))
                New_pi_b[stateID,:,option] = temp_theta
        
        return New_pi_b
        
    def Online_Baum_Welch(self, T_min):
        TrainingSetID = OnlineHIL.TrainingSetID(self)
        pi_hi = OnlineHIL.initialize_pi_hi(self)
        pi_b = OnlineHIL.initialize_pi_b(self)
        pi_lo = OnlineHIL.initialize_pi_lo(self)
        stateSpace = self.expert.StateSpace()
        StateSpace_size = len(stateSpace)
        
        #Initialization 
        pi_hi_agent = PI_HI(pi_hi) 
        pi_b_agent = PI_B(pi_b)
        pi_lo_agent = PI_LO(pi_lo)
        
        zi = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, StateSpace_size, 1))
        phi_h = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, 
                         StateSpace_size, self.termination_space, self.option_space,1))
        norm = np.zeros((len(self.mu), self.action_space, StateSpace_size))
        P_option_given_obs = np.zeros((self.option_space, 1))

        State = TrainingSetID[0,0]
        Action = self.Labels[0]
        
        for a1 in range(self.action_space):
            for s1 in range(StateSpace_size):
                for o0 in range(self.option_space):
                    for b1 in range(self.termination_space):
                        for o1 in range(self.option_space):
                            state = s1
                            action = a1
                            zi[o0,b1,o1,a1,s1,0] = OnlineHIL.Pi_combined(o1, o0, action, b1, pi_hi_agent.Policy, pi_lo_agent.Policy, pi_b_agent.Policy, 
                                                                         state, self.zeta, self.option_space)
                                                       
                    norm[o0,a1,s1]=self.mu[o0]*np.sum(zi[:,:,:,a1,s1,0],(1,2))[o0]
            
                zi[:,:,:,a1,s1,0] = np.divide(zi[:,:,:,a1,s1,0],np.sum(norm[:,a1,s1]))
                if a1 == int(Action) and s1 == int(State):
                    P_option_given_obs[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi[:,:,:,a1,s1,0],1))*self.mu),0) 

        for a1 in range(self.action_space):
            for s1 in range(StateSpace_size):
                for o0 in range(self.option_space):
                    for b1 in range(self.termination_space):
                        for o1 in range(self.option_space):
                            for bT in range(self.termination_space):
                                for oT in range(self.option_space):
                                    if a1 == int(Action) and s1 == int(State):
                                        phi_h[o0,b1,o1,a1,s1,bT,oT,0] = zi[o0,b1,o1,a1,s1,0]*self.mu[o0]
                                    else:
                                        phi_h[o0,b1,o1,a1,s1,bT,oT,0] = 0
                                        
        for t in range(1,len(self.TrainingSet)):
        
            if np.mod(t,100)==0:
                print('iter', t, '/', len(self.TrainingSet))
    
            #E-step
            zi_temp1 = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, StateSpace_size, 1))
            phi_h_temp = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, StateSpace_size, 
                                  self.termination_space, self.option_space, 1))
            norm = np.zeros((len(self.mu), self.action_space, StateSpace_size))
            P_option_given_obs_temp = np.zeros((self.option_space, 1))
            prod_term = np.ones((self.option_space, self.termination_space, self.option_space, self.action_space, StateSpace_size, 
                                 self.termination_space, self.option_space))
    
            State = TrainingSetID[t,0]
            Action = self.Labels[t]
            for at in range(self.action_space):
                for st in range(StateSpace_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                state = st
                                action = at
                                zi_temp1[ot_past,bt,ot,at,st,0] = OnlineHIL.Pi_combined(ot, ot_past, action, bt, pi_hi_agent.Policy, 
                                                                                        pi_lo_agent.Policy,  pi_b_agent.Policy, state, self.zeta, 
                                                                                        self.option_space)
                
                        norm[ot_past,at,st] = P_option_given_obs[ot_past,0]*np.sum(zi_temp1[:,:,:,at,st,0],(1,2))[ot_past]
    
                    zi_temp1[:,:,:,at,st,0] = np.divide(zi_temp1[:,:,:,at,st,0],np.sum(norm[:,at,st]))
                    if at == int(Action) and st == int(State):
                        P_option_given_obs_temp[:,0] = np.sum(np.transpose(np.transpose(np.sum(zi_temp1[:,:,:,at,st,0],1))*P_option_given_obs[:,0]),0) 
            
            zi = zi_temp1
    
            for at in range(self.action_space):
                for st in range(StateSpace_size):
                    for ot_past in range(self.option_space):
                        for bt in range(self.termination_space):
                            for ot in range(self.option_space):
                                for bT in range(self.termination_space):
                                    for oT in range(self.option_space):
                                        prod_term[ot_past, bt, ot, at, st, bT, oT] = np.sum(zi[:,bT,oT,int(Action),int(State),0]*np.sum(phi_h[ot_past,bt,ot,at,st,:,:,0],0))
                                        if at == int(Action) and st == int(State):
                                            phi_h_temp[ot_past,bt,ot,at,st,bT,oT,0] = (1/t)*zi[ot_past,bt,ot,at,st,0]*P_option_given_obs[ot_past,0] 
                                            + (1-1/t)*prod_term[ot_past,bt,ot,at,st,bT,oT]
                                        else:
                                            phi_h_temp[ot_past,bt,ot,at,st,bT,oT,0] = (1-1/t)*prod_term[ot_past,bt,ot,at,st,bT,oT]
                                    
            phi_h = phi_h_temp
            P_option_given_obs = P_option_given_obs_temp
            
            #M-step 
            if t > T_min:
                pi_hi = OnlineHIL.UpdatePiHi(self, pi_hi_agent.pi_hi, phi_h)
                pi_lo = OnlineHIL.UpdatePiLo(self, pi_lo_agent.pi_lo, phi_h)
                pi_b = OnlineHIL.UpdatePiB(self, pi_b_agent.pi_b, phi_h)
                
                pi_hi_agent = PI_HI(pi_hi) 
                pi_b_agent = PI_B(pi_b)
                pi_lo_agent = PI_LO(pi_lo)
                
        return pi_hi, pi_lo, pi_b
                
                
                
                                        
                                        
                                        
                                        
                                        
                                        
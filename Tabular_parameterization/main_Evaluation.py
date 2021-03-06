#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:17:59 2020

@author: vittorio
"""
import World
import numpy as np
import BatchBW_HIL 
import matplotlib.pyplot as plt

with open('Models/Saved_Model_Expert/pi_hi.npy', 'rb') as f:
    pi_hi_expert = np.load(f)
    
with open('Models/Saved_Model_Expert/pi_lo.npy', 'rb') as f:
    pi_lo_expert = np.load(f)

with open('Models/Saved_Model_Expert/pi_b.npy', 'rb') as f:
    pi_b_expert = np.load(f)   
    
with open('Models/Saved_Model_Batch/pi_hi.npy', 'rb') as f:
    pi_hi_batch = np.load(f)
    
with open('Models/Saved_Model_Batch/pi_lo.npy', 'rb') as f:
    pi_lo_batch = np.load(f)

with open('Models/Saved_Model_Batch/pi_b.npy', 'rb') as f:
    pi_b_batch = np.load(f)   

with open('Models/Saved_Model_Batch/pi_hi_evolution.npy', 'rb') as f:
    pi_hi_evolution = np.load(f)
    
with open('Models/Saved_Model_Batch/pi_lo_evolution.npy', 'rb') as f:
    pi_lo_evolution = np.load(f)

with open('Models/Saved_Model_Batch/pi_b_evolution.npy', 'rb') as f:
    pi_b_evolution = np.load(f)   
        

# %% Expert
expert = World.TwoRooms.Expert()
ExpertSim = expert.Simulation(pi_hi_expert, pi_lo_expert, pi_b_expert)
max_epoch = 100 #max iterations in the simulation per trajectory
nTraj = 100 #number of trajectories generated
[trajExpert, controlExpert, OptionsExpert, 
 TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj,1)
best = np.argmax(rewardExpert)   
ExpertSim.HILVideoSimulation(controlExpert[best][:], trajExpert[best][:], 
                             OptionsExpert[best][:],"Videos/VideosExpert/sim_HierarchExpert.mp4")

# %% Batch Agent
Batch_Plot = expert.Plot(pi_hi_batch, pi_lo_batch, pi_b_batch)
Batch_Plot.PlotHierachicalPolicy('Figures/FiguresBatch/Batch_High_policy_psi{}.eps','Figures/FiguresBatch/Batch_Action_option{}_psi{}.eps','Figures/FiguresBatch/Batch_Termination_option{}_psi{}.eps')
BatchSim = expert.Simulation(pi_hi_batch, pi_lo_batch, pi_b_batch)
[trajBatch, controlBatch, OptionsBatch, 
 TerminationBatch, psiBatch, rewardBatch] = BatchSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj,1)
best = np.argmax(rewardBatch)  
BatchSim.HILVideoSimulation(controlBatch[best][:], trajBatch[best][:], 
                            OptionsBatch[best][:],"Videos/VideosBatchAgent/sim_BatchBW.mp4")

# %% Comparison
AverageRewardExpert = np.sum(rewardExpert)/nTraj
AverageRewardBatch = np.sum(rewardBatch)/nTraj

# %% L2 norm 
L2_norm_error = np.empty((0)) 
for i in range(len(pi_hi_evolution)):
    error_hi = np.sqrt(np.sum((pi_hi_evolution[i] - pi_hi_expert)**2))
    error_lo = np.sqrt(np.sum((pi_lo_evolution[i] - pi_lo_expert)**2))
    error_b = np.sqrt(np.sum((pi_b_evolution[i] - pi_b_expert)**2))
    error = error_hi + error_lo + error_b
    L2_norm_error = np.append(L2_norm_error, error)

# %%
iterations = np.linspace(0,len(L2_norm_error),len(L2_norm_error))    
plt.figure()
plt.plot(iterations, L2_norm_error, '-ok')
plt.savefig('Figures/L2_norm_error.eps', format='eps')


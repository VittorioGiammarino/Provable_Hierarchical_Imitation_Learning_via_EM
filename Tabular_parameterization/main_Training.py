#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:02:52 2020

@author: vittorio
"""
import World 
import BatchBW_HIL 
import numpy as np

# %% Expert Policy Generation and simulation
expert = World.TwoRooms.Expert()
theta_hi_star = 0.6
theta_lo_star = 0.7
theta_b_star = 0.8
pi_hi_expert, pi_lo_expert, pi_b_expert = expert.HierarchicalPolicy(theta_hi_star,theta_lo_star,theta_b_star)
ExpertPlot = expert.Plot(pi_hi_expert, pi_lo_expert, pi_b_expert)
ExpertPlot.PlotHierachicalPolicy('Figures/FiguresExpert/Hierarchical/PIHI_{}', 'Figures/FiguresExpert/Hierarchical/PILO_option{}_{}', 'Figures/FiguresExpert/Hierarchical/PIB_option{}_{}')
ExpertSim = expert.Simulation(pi_hi_expert, pi_lo_expert, pi_b_expert)
max_epoch = 200 #max iterations in the simulation per trajectory
nTraj = 100 #number of trajectories generated
seed = 1 # random seed
[trajExpert, controlExpert, OptionsExpert, 
 TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch, nTraj, seed)
best = np.argmax(rewardExpert)   
ExpertSim.HILVideoSimulation(controlExpert[best][:], trajExpert[best][:], 
                             OptionsExpert[best][:], "Videos/VideosExpert/sim_HierarchExpert.mp4")

# %% Batch BW for HIL with tabular parameterization: Training
ss = expert.Environment.stateSpace
Labels, TrainingSet = BatchBW_HIL.ProcessData(trajExpert, controlExpert, psiExpert, ss)
option_space = 2
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space)
N=50 #number of iterations for the BW algorithm
theta_hi_init = 0.5
theta_lo_init = 0.6
theta_b_init = 0.7
pi_hi_batch, pi_lo_batch, pi_b_batch, pi_hi_evolution, pi_lo_evolution, pi_b_evolution = Agent_BatchHIL.Baum_Welch(N, pi_hi_expert, pi_lo_expert, pi_b_expert, theta_hi_init, theta_lo_init, theta_b_init)

# %% Save Model
with open('Models/Saved_Model_Expert/pi_hi.npy', 'wb') as f:
    np.save(f, pi_hi_expert)
    
with open('Models/Saved_Model_Expert/pi_lo.npy', 'wb') as f:
    np.save(f, pi_lo_expert)

with open('Models/Saved_Model_Expert/pi_b.npy', 'wb') as f:
    np.save(f, pi_b_expert)    
    
with open('Models/Saved_Model_Batch/pi_hi.npy', 'wb') as f:
    np.save(f, pi_hi_batch)
    
with open('Models/Saved_Model_Batch/pi_lo.npy', 'wb') as f:
    np.save(f, pi_lo_batch)

with open('Models/Saved_Model_Batch/pi_b.npy', 'wb') as f:
    np.save(f, pi_b_batch)    
    
with open('Models/Saved_Model_Batch/pi_hi_evolution.npy', 'wb') as f:
    np.save(f, pi_hi_evolution)
    
with open('Models/Saved_Model_Batch/pi_lo_evolution.npy', 'wb') as f:
    np.save(f, pi_lo_evolution)

with open('Models/Saved_Model_Batch/pi_b_evolution.npy', 'wb') as f:
    np.save(f, pi_b_evolution)        

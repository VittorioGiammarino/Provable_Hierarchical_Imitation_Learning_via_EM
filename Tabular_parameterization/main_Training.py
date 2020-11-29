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
pi_hi_expert, pi_lo_expert, pi_b_expert = expert.HierarchicalPolicy()
ExpertPlot = expert.Plot(pi_hi_expert, pi_lo_expert, pi_b_expert)
ExpertPlot.PlotHierachicalPolicy('Figures/FiguresExpert/Hierarchical/PIHI_{}', 'Figures/FiguresExpert/Hierarchical/PILO_option{}_{}', 'Figures/FiguresExpert/Hierarchical/PIB_option{}_{}')
ExpertSim = expert.Simulation(pi_hi_expert, pi_lo_expert, pi_b_expert)
max_epoch = 100 #max iterations in the simulation per trajectory
nTraj = 10 #number of trajectories generated
[trajExpert, controlExpert, OptionsExpert, 
 TerminationExpert, psiExpert, rewardExpert] = ExpertSim.HierarchicalStochasticSampleTrajMDP(max_epoch,nTraj,1)
best = np.argmax(rewardExpert)   
ExpertSim.HILVideoSimulation(controlExpert[best][:], trajExpert[best][:], 
                             OptionsExpert[best][:], "Videos/VideosExpert/sim_HierarchExpert.mp4")

# %% Batch BW for HIL with tabular parameterization: Training
ss = expert.Environment.stateSpace
Labels, TrainingSet = BatchBW_HIL.ProcessData(trajExpert, controlExpert, psiExpert, ss)
option_space = 2
Agent_BatchHIL = BatchBW_HIL.BatchHIL(TrainingSet, Labels, option_space)
N=10 #number of iterations for the BW algorithm
pi_hi_batch, pi_lo_batch, pi_b_batch = Agent_BatchHIL.Baum_Welch(N)


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
    
    

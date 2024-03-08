#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:46:34 2020

@author: mirjamheinemans
"""
#%%
import scipy.stats as ss
import csv 
import numpy as np
import os, glob # Operating System
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/loom/conditioning/'
#%%

def CondFreezLoom(file_names):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name + file_names, usecols = [1,2,3])
        base = dataset.iloc[:,].diff()[dataset.iloc[:,0].diff() != 0].index.values
        baseline = base[1]
        freeze_base = (sum(dataset.iloc[:baseline,1]))/baseline*100
        
        name = file_names.replace('.csv',"")

        stim = dataset.iloc[:,1].diff()[dataset.iloc[:,1].diff() != 0].index.values
        end_stim = stim[-1]
        freeze_stim = (sum(dataset.iloc[baseline:end_stim,1]))/(end_stim-baseline)*100
        
        
        freezing_percent = pd.DataFrame(data =[freeze_base,freeze_stim], index =['base','stim'],columns = [name])
       
        return(freezing_percent)

#%%
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/loom/conditioning/'
columns = ['xpos']
index = range(1)
freeze_loom = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = CondFreezLoom(file_names)
    freeze_loom = pd.concat([freeze_loom, animal], axis=1, sort = True)

freeze_loom = freeze_loom.drop(columns = ['xpos'])
freeze_loom = freeze_loom.drop(index = 0)
columns = ['xpos']
index = range(1)
df_loom_high_freezer = pd.DataFrame(index = index, columns = columns)
for rat in freeze_loom:
    if ('26' in rat) | ('34' in rat) |('43' in rat) |('44' in rat) |('48' in rat) |('49' in rat) |('53' in rat) |('58' in rat) | ('time' in rat):
        animal = freeze_loom[rat]
        df_loom_high_freezer = pd.concat([df_loom_high_freezer, animal], axis=1, sort = True)
df_loom_high_freezer = df_loom_high_freezer.drop(columns = ['xpos'])


loom = df_loom_high_freezer.T
#%%

def CondFreezShock(file_names):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name + file_names, usecols = [1,2,3])
        base = dataset.iloc[:,].diff()[dataset.iloc[:,0].diff() != 0].index.values
        baseline = base[1]
        freeze_base = (sum(dataset.iloc[:baseline,1]))/baseline*100
        
        name = file_names.replace('.csv',"")

        stim = dataset.iloc[:,1].diff()[dataset.iloc[:,1].diff() != 0].index.values
        end_stim = stim[-1]
        freeze_stim = (sum(dataset.iloc[baseline:end_stim,1]))/(end_stim-baseline)*100
    
        
        freezing_percent = pd.DataFrame(data =[freeze_base,freeze_stim], index =['base','stim'],columns = [name])
       
        return(freezing_percent)    
 #%%
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/shock/conditioning/'
columns = ['xpos']
index = range(1)
freeze_shock = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = CondFreezShock(file_names)
    freeze_shock = pd.concat([freeze_shock, animal], axis=1, sort = True)

freeze_shock = freeze_shock.drop(columns = ['xpos'])
freeze_shock = freeze_shock.drop(index = 0)     
shock = freeze_shock.T
#%%
# shock pre and post
before_s = shock['base']
after_s = shock['stim']
fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):
    
    # plotting the points
    plt.scatter(np.zeros(len(before_s)), before_s, color = 'crimson', s = 80, zorder = 2)
    plt.scatter(np.ones(len(after_s)), after_s, color = 'crimson',s =80, zorder = 2)  
    # plotting the lines
    for i in range(len(before_s)):
        plt.plot( [0,1], [before_s[i], after_s[i]], c='orange', zorder =1)
        
    plt.xticks([0,1], ['Baseline', 'Stimulus'])
    plt.tick_params(labelsize=20)
    plt.ylim([-5,105])
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2
    
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    
    plt.tick_params(labelsize=20)
    sns.despine()
    os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/graphs') # to change directory Read csv files with Pandas  
    plt.savefig('HIGHFREEZEcond_s_before_after.jpeg',bbox_inches='tight', dpi=1800)
#%%

# loom pre and post
before_l = loom['base']
after_l = loom['stim']
fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):
    
    # plotting the points
    plt.scatter(np.zeros(len(before_l)), before_l, color = 'royalblue', s = 80, zorder = 2)
    plt.scatter(np.ones(len(after_l)), after_l, color = 'royalblue',s =80, zorder = 2)  
    # plotting the lines
    for i in range(len(before_s)):
        plt.plot( [0,1], [before_l[i], after_l[i]], c='orange', zorder =1)
        
    plt.xticks([0,1], ['Baseline', 'Stimulus'])
    plt.tick_params(labelsize=20)
    plt.ylim([-5,105])
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2
    
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    
    plt.tick_params(labelsize=20)
    sns.despine()
    os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/graphs') # to change directory Read csv files with Pandas  
    plt.savefig('HIGHFREEZEcond_l_before_after.jpeg',bbox_inches='tight', dpi=1800)
#%%

shock['condition'] = "shock"
loom['condition'] = "loom"
cond_freeze = pd.concat([shock, loom], axis = 0)

cond_freeze['increase'] = cond_freeze['stim']-cond_freeze['base']
#%%
'''freezing during stimulus'''
fig = plt.figure(figsize=(2,4))
total = list(cond_freeze['increase'])
condition = list(cond_freeze['condition'])

ax = sns.swarmplot(x="condition", y="increase", 
                 data=cond_freeze, palette = ['crimson', 'royalblue'], s = 8)

#ax.set_title('Freezing increase to conditioned tone\n', fontsize=20)
ax = sns.boxplot(x=condition, y=total, color = 'lightgray',  width = 0.6, showfliers = False)
ax.set_ylabel('', fontsize=30)
ax.set_xlabel('', fontsize=50)
ax.tick_params(labelsize=20)
ax.set_ylim([-8,105])
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 2
ax.yaxis.set_tick_params(size=5, width=2)
ax.xaxis.set_tick_params(size=5, width=2)
ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.01)) 
plt.savefig('HIGHFREEZEcond_increase.jpeg',bbox_inches='tight', dpi=1800)
#%%
'''freezing during stimulus'''
#
#total = list(cond_freeze['stim'])
#condition = list(cond_freeze['condition'])
#ax = sns.swarmplot(x="condition", y="stim", 
#                 data=cond_freeze, palette = ['crimson', 'royalblue'], s = 7)
#ax.set_title('Exp1 Training \nFreezing during stimulus')
#ax = sns.boxplot(x=condition, y=total, color = 'lightgray', showfliers = False)
#ax.set_ylim([0,100])
#




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:27:23 2019

@author: mirjamheinemans
"""

import scipy.stats as ss
import csv 
import numpy as np
import os, glob # Operating System
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout') # to change directory Read csv files with Pandas
#%%

#path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/loom/conditioning'


def FreezeLoom(file_names):

    if file_names == '.DS_Store':
        next
    else:
        if 'ML0' in file_names:
            dataset = pd.read_csv(path_name + file_names, usecols = [1,2,3])
            data_sec = dataset.groupby(np.arange(len(dataset))//15).mean()
            file_names = file_names.replace('0','')
            
            
        else:
            dataset = pd.read_csv(path_name + file_names, usecols = [1,2,3])
            data_sec = dataset.groupby(np.arange(len(dataset))//60).mean()
        
        
        rat = file_names.replace('.csv', '')
        data_sec[rat+str('diff')] = data_sec[rat+str('_baseline')].diff()
        
        first_stim = data_sec[data_sec[rat+str('diff')].values<-.5].index[0]
        dataset_correct = data_sec.loc[(first_stim-610):,:]
        
        
        
        dataset_correct = dataset_correct.reset_index(drop=True)
        return(dataset_correct)

#%%
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/loom/conditioning/'
columns = ['xpos']
index = range(1)
freeze_loom = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = FreezeLoom(file_names)
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
#%%
loom_15_sec = df_loom_high_freezer.groupby(np.arange(len(df_loom_high_freezer))//15).mean()

loom_15_sec['average'] = loom_15_sec.loc[:, [x for x in loom_15_sec.columns if x.endswith('freeze')]].mean(axis=1)
loom_15_sec['percent'] = loom_15_sec['average']*100

loom_15_sec['stdev'] = loom_15_sec.loc[:, [x for x in loom_15_sec.columns if x.endswith('freeze')]].std(axis=1)
loom_15_sec['SEM'] = (loom_15_sec['stdev']/np.sqrt(16))*100 #to make it percent instead of proportion


loom_15_sec['lower_bound'] = loom_15_sec['percent'] - loom_15_sec['SEM'] 
loom_15_sec['upper_bound'] = loom_15_sec['percent'] + loom_15_sec['SEM'] 

loom_15_sec['15sec']=  range(len(loom_15_sec.average))
loom_15_sec['min']=  (loom_15_sec['15sec']*15/60) 


#%%

ax = sns.lineplot(x='min', y='percent', data =loom_15_sec, color = 'dodgerblue').set_title('Tone-Loom conditioning\n', fontsize=20)#T-Shock
ax = plt.fill_between(loom_15_sec['min'],loom_15_sec['lower_bound'] , loom_15_sec['upper_bound'], color ='dodgerblue',alpha=.3)
ax = sns.lineplot(x=[10,10], y =[0,1.0], color = 'gray')
ax.axvspan(0,30, alpha=0.3, color='gray')
ax.set_ylim([-0, 100])
ax.set_xlim([-10,28])
ax.set_ylabel('Freezing %', fontsize=20)
ax.set_xlabel('Minutes', fontsize=20)
ax.tick_params(labelsize=15)
#%%

path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/shock/conditioning'


def FreezeShock(file_names):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name + file_names, usecols = [1,2,3])
        # baseline_sec = (dataset.iloc[:,].diff()[dataset.iloc[:,0].diff() <0].index.values[0])//25
        data_sec = dataset.groupby(np.arange(len(dataset))//25).mean()

        rat = file_names.replace('.csv', '')
        data_sec[rat+str('diff')] = data_sec[rat+str('_baseline')].diff()
        
        first_stim = data_sec[data_sec[rat+str('diff')].values<-.5].index[0]
        dataset_correct = data_sec.loc[(first_stim-610):,:]
        
        
        
        dataset_correct = dataset_correct.reset_index(drop=True)
        return(dataset_correct)

       
#%%
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/shock/conditioning/'
columns = ['xpos']
index = range(1)
freeze_shock= pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = FreezeShock(file_names)
    freeze_shock = pd.concat([freeze_shock, animal], axis=1, sort = True)

freeze_shock = freeze_shock.drop(columns = ['xpos'])
freeze_shock = freeze_shock.drop(index = 0)
#df_s = freeze_shock.T
#df_s.loc['condition',:] = 'Shock'


#%%
shock_15_sec = freeze_shock.groupby(np.arange(len(freeze_shock))//15).mean()

shock_15_sec['average'] = shock_15_sec.loc[:, [x for x in shock_15_sec.columns if x.endswith('freeze')]].mean(axis=1)
shock_15_sec['percent'] = shock_15_sec['average']*100

shock_15_sec['stdev'] = shock_15_sec.loc[:, [x for x in shock_15_sec.columns if x.endswith('freeze')]].std(axis=1)
shock_15_sec['SEM'] = shock_15_sec['stdev']/np.sqrt(8)*100

shock_15_sec['lower_bound'] = shock_15_sec['percent'] - shock_15_sec['SEM'] 
shock_15_sec['upper_bound'] = shock_15_sec['percent'] + shock_15_sec['SEM'] 

shock_15_sec['15sec']=  range(len(shock_15_sec.average))
shock_15_sec['min']=  (shock_15_sec['15sec']*15/60) 


#%%
fig = plt.figure(figsize=(5,4))
ax = sns.lineplot(x='min', y='percent', data =loom_15_sec, color = 'royalblue')#.set_title('Tone-Loom conditioning\n', fontsize=20)#T-Shock
ax = plt.fill_between(loom_15_sec['min'],loom_15_sec['lower_bound'] , loom_15_sec['upper_bound'], color ='royalblue',alpha=.3)
ax = sns.lineplot(x=[10,10], y =[0,1.0], color = 'gray')

ax = sns.lineplot(x='min', y='percent', data =shock_15_sec, color = 'crimson')#.set_title('Freezing levels during conditioning\n', fontsize=20)#T-Shock
ax = plt.fill_between(shock_15_sec['min'],shock_15_sec['lower_bound'] , shock_15_sec['upper_bound'], color ='crimson',alpha=.3)
ax = sns.lineplot(x=[10,10], y =[0,1.0], color = 'lightgray')
ax.axvspan(10,40, alpha=0.2, color='gray')
ax.axvline(10, alpha=0.9, ls='--', color='k')

ax.set_ylim([0, 100])
ax.set_xlim([0,40])

ax.yaxis.set_ticks(np.arange(0, 101, 25))
ax.set_ylabel('', fontsize=20)
ax.set_xlabel('', fontsize=20)
ax.tick_params(labelsize=20)

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 2
ax.yaxis.set_tick_params(size=5, width=2)
ax.xaxis.set_tick_params(size=5, width=2)
ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.03)) 

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/graphs') # to change directory Read csv files with Pandas

plt.savefig('cond_lineHIGHFREEZERS.jpeg',bbox_inches='tight', dpi=1800)



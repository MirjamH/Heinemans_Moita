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
#%%

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

      
stimbystim_loom = pd.DataFrame(index = index, columns = columns)
baseline_loom = pd.DataFrame(index = index, columns = columns)

baseline_times = [50,110,170,230,290,350,410,470,530,590]
loom_onset = [10,610,802,1063,1225,1296,1578,1699,1921,2232,2319]
tone_onset = [value - 10 for value in loom_onset]

for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')
    for i in range(10):
        baseline_freezing =  freeze_loom.loc[(baseline_times[i]):(baseline_times[i]+10),rat_name+str('_freeze')].mean()
        freeze_proportion =  freeze_loom.loc[(tone_onset[i]):(loom_onset[i]-1),rat_name+str('_freeze')].sum()/(10)
        print(rat_name)
        print(freeze_proportion)
        stimbystim_loom.loc[i+1,rat_name+str('_freeze')]=freeze_proportion
        baseline_loom.loc[i,rat_name+str('baseline')]=baseline_freezing

stimbystim_loom = stimbystim_loom.drop(columns = ['xpos'])
baseline_loom = baseline_loom.drop(columns = ['xpos'])

for rat in range(len(stimbystim_loom.columns)):
    baseline_median = baseline_loom.iloc[:,rat].median()
    print(baseline_median)
    stimbystim_loom.iloc[0,rat]=baseline_median

stimbystim_loom = stimbystim_loom.T
stimbystim_loom['baseline'] = stimbystim_loom[0]*100
stimbystim_loom['stimulus'] = stimbystim_loom.iloc[:,1:9].mean(axis=1)*100

result_loom,p_loom= ss.wilcoxon(x=stimbystim_loom['baseline'],y=stimbystim_loom['stimulus'])


#%%
'''For the tone-shock condition'''

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


#%%
baseline_times = [50,110,170,230,290,350,410,470,530,590]
stimbystim_shock = pd.DataFrame(index = index, columns = columns)
baseline_shock = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')
    shock_onset = freeze_shock.loc[freeze_shock[rat_name+str('_stim')]>0.27].index.values.astype(int)
    
    first_loom = np.array([10])
    shock_onset = np.concatenate((first_loom, shock_onset))
    tone_onset = [value - 10 for value in shock_onset]
    print(len(tone_onset))

    if rat_name == 'ML42':
        loom_number = 8
    else:
        loom_number = 10
        
    for i in range (loom_number):
        # print(tone_onset[i])

        baseline_freezing =  freeze_shock.loc[(baseline_times[i]):(baseline_times[i]+10),rat_name+str('_freeze')].mean()  
        freeze_proportion =  freeze_shock.loc[(tone_onset[i]):(shock_onset[i]-1),rat_name+str('_freeze')].sum()/(10)
        # print(rat_name)
        # print(freeze_proportion)
        stimbystim_shock.loc[i+1,rat_name+str('_freeze')]=freeze_proportion
        baseline_shock.loc[i,rat_name+str('baseline')]=baseline_freezing
        
stimbystim_shock = stimbystim_shock.drop(columns = ['xpos'])
baseline_shock = baseline_shock.drop(columns = ['xpos'])

for rat in range(len(baseline_shock.columns)):
    baseline_median = baseline_shock.iloc[:,rat].median()
    print(baseline_median)
    stimbystim_shock.iloc[0,rat]=baseline_median

stimbystim_shock = stimbystim_shock.T
stimbystim_shock['baseline'] = stimbystim_shock[0]*100
stimbystim_shock['stimulus'] = stimbystim_shock.iloc[:,1:9].mean(axis=1)*100

result_shock,p_shock= ss.wilcoxon(x=stimbystim_shock['baseline'],y=stimbystim_shock['stimulus'])

#%%
# shock pre and post
before_s = stimbystim_shock['baseline']
after_s = stimbystim_shock['stimulus']
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
    plt.ylim([-7,105])
    plt.xlim([-0.1,1.1])
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2
    
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    
    plt.tick_params(labelsize=20)
    sns.despine()
    os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/graphs') # to change directory Read csv files with Pandas  
    plt.savefig('DURINGTONE_cond_s_before_after.jpeg',bbox_inches='tight', dpi=1800)
#%%

# loom pre and post
before_l = stimbystim_loom['baseline']
after_l = stimbystim_loom['stimulus']
fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):
    
    # plotting the points
    plt.scatter(np.zeros(len(before_l)), before_l, color = 'royalblue', s = 80, zorder = 2)
    plt.scatter(np.ones(len(after_l)), after_l, color = 'royalblue',s =80, zorder = 2)  
    # plotting the lines
    for i in range(len(before_l)):
        plt.plot( [0,1], [before_l[i], after_l[i]], c='orange', zorder =1)
        
    plt.xticks([0,1], ['Baseline', 'Stimulus'])
    plt.tick_params(labelsize=20)
    plt.ylim([-7,105])
    plt.xlim([-0.1,1.1])
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2
    
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    
    plt.tick_params(labelsize=20)
    sns.despine()
    os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/graphs') # to change directory Read csv files with Pandas  
    plt.savefig('DURINGTONE_cond_l_before_after.jpeg',bbox_inches='tight', dpi=1800)
#%%

stimbystim_shock['condition'] = "shock"
stimbystim_loom['condition'] = "loom"
cond_freeze = pd.concat([stimbystim_shock, stimbystim_loom], axis = 0)

cond_freeze['increase'] = cond_freeze['stimulus']-cond_freeze['baseline']
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
plt.savefig('DURINGTONE_cond_increase.jpeg',bbox_inches='tight', dpi=1800)
#%%
'''freezing during stimulus'''
stimbystim_shock['increase'] = stimbystim_shock['baseline'] - stimbystim_shock['stimulus']
stimbystim_loom['increase'] = stimbystim_loom['baseline'] - stimbystim_loom['stimulus']

ss.mannwhitneyu(x=stimbystim_shock['increase'],y=stimbystim_loom['increase'])




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
df_loom = pd.read_csv('/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/loom/test/Loom_test_freeze.csv')        
df_shock = pd.read_csv('/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/shock/test/Shock_test_freeze.csv')        

#%%
columns = ['xpos']
index = range(1)
freeze_loom = pd.DataFrame(index = index, columns = columns)


for col in df_loom:
    if col.startswith("ML"):
        base1 = df_loom.loc[9,col]
        base2 = df_loom.loc[19,col]
        base3 = df_loom.loc[29,col]
        base = (base1+base2+base3)/3
        
        stim1 = df_loom.loc[30,col]
        stim2 = df_loom.loc[48,col]
        stim3 = df_loom.loc[66,col]
        stim = (stim1+stim2+stim3)/3
        
        animal = pd.DataFrame(data =[base,stim], index =['base','stim'],columns = [col])
        freeze_loom = pd.concat([freeze_loom, animal], axis=1, sort = True)

freeze_loom = freeze_loom.drop(columns = ['xpos'])
freeze_loom = freeze_loom.drop(index = 0)

#%%

freeze_loom = freeze_loom.drop(columns = ['ML2','ML8', 'ML33'])
#%%
columns = ['xpos']
index = range(1)
freeze_shock = pd.DataFrame(index = index, columns = columns)


for col in df_shock:
    if col.startswith("ML"):
        base1 = df_shock.loc[9,col]
        base2 = df_shock.loc[19,col]
        base3 = df_shock.loc[29,col]
        base = (base1+base2+base3)/3
        
        stim1 = df_shock.loc[30,col]
        stim2 = df_shock.loc[48,col]
        stim3 = df_shock.loc[66,col]
        stim = (stim1+stim2+stim3)/3
        animal = pd.DataFrame(data =[base,stim], index =['base','stim'],columns = [col])
        freeze_shock = pd.concat([freeze_shock, animal], axis=1, sort = True)

freeze_shock = freeze_shock.drop(columns = ['xpos'])
freeze_shock = freeze_shock.drop(index = 0)



#%%
shock = freeze_shock.T
loom = freeze_loom.T


result_shock,p_shock= ss.wilcoxon(x=shock['base'],y=shock['stim'])

result_loom,p_loom= ss.wilcoxon(x=loom['base'],y=loom['stim'])
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
    plt.savefig('TONEBYTONE_ttest_s_before_after.jpeg',bbox_inches='tight', dpi=1800)
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
    plt.savefig('TONEBYTONE_ttest_l_before_after.jpeg',bbox_inches='tight', dpi=1800)

#%%

shock['condition'] = "shock"
loom['condition'] = "loom"
test_freeze = pd.concat([shock, loom], axis = 0)

test_freeze['increase'] = test_freeze['stim']-test_freeze['base']

#%%
'''freezing during stimulus'''
fig = plt.figure(figsize=(2,4))
total = list(test_freeze['increase'])
condition = list(test_freeze['condition'])

ax = sns.swarmplot(x="condition", y="increase", 
                 data=test_freeze, palette = ['crimson', 'royalblue'], s = 8)

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
plt.savefig('TONEBYTONE_test_increase.jpeg',bbox_inches='tight', dpi=1800)

#%%
'''freezing during stimulus'''
shock['increase'] = shock['base'] - shock['stim']
loom['increase'] = loom['base'] - loom['stim']

ss.mannwhitneyu(x=shock['increase'],y=loom['increase'])






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
        base = sum(df_loom.loc[:29,col])/30
        stim = sum(df_loom.loc[30:,col])/40
        animal = pd.DataFrame(data =[base,stim], index =['base','stim'],columns = [col])
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
columns = ['xpos']
index = range(1)
freeze_shock = pd.DataFrame(index = index, columns = columns)


for col in df_shock:
    if col.startswith("ML"):
        base = sum(df_shock.loc[:29,col])/30
        stim = sum(df_shock.loc[30:,col])/40
        animal = pd.DataFrame(data =[base,stim], index =['base','stim'],columns = [col])
        freeze_shock = pd.concat([freeze_shock, animal], axis=1, sort = True)

freeze_shock = freeze_shock.drop(columns = ['xpos'])
freeze_shock = freeze_shock.drop(index = 0)

#%%
shock = freeze_shock.T
loom = df_loom_high_freezer.T


#%%
result_shock,p_shock= ss.wilcoxon(x=shock['base'],y=shock['stim'])

result_loom,p_loom= ss.wilcoxon(x=loom['base'],y=loom['stim'])


#%%
# shock pre and post
before_s = shock['base']
after_s = shock['stim']
fig = plt.figure(figsize=(2,4))
# with sns.axes_style("ticks"):

# plotting the points
ax = plt.scatter(np.zeros(len(before_s)), before_s, color = 'crimson', s = 80, zorder = 2)
ax = plt.scatter(np.ones(len(after_s)), after_s, color = 'crimson',s =80, zorder = 2)  
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
plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/test_HIGHFREEZERS__s_before_after.jpeg',bbox_inches='tight', dpi=1800)
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
    plt.ylim([-7,105])
    plt.xlim([-0.1,1.1])
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2

    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    
    plt.tick_params(labelsize=20)
    sns.despine()
    plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/test_HIGHFREEZERS__l_before_after.jpeg',bbox_inches='tight', dpi=1800)

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
# plt.rcParams['axes.position'] = -0.1
ax.yaxis.set_tick_params(size=5, width=2)
ax.xaxis.set_tick_params(size=5, width=2)

plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/test_HIGHFREEZERS_dots.jpeg',bbox_inches='tight', dpi=1800)

#%%
shock['increase'] = shock['stim'] -shock['base'] 
loom['increase'] = loom['stim'] - loom['base'] 

ss.mannwhitneyu(x=shock['increase'],y=loom['increase'])




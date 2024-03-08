#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:23:12 2020

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

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/conditioning71_102')
path = os.getcwd() #short for get Current Work Directory

#%%
df_loom = pd.read_csv('/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/loom/test/Loom_test_freeze.csv')        
df_shock = pd.read_csv('/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/shock/test/Shock_test_freeze.csv')        

# df_loom.drop('ML2', axis=1, inplace=True)
# df_loom.drop('ML8', axis=1, inplace=True)
df_loom.drop('ML14', axis=1, inplace=True)
#%%

columns = ['xpos']
index = range(1)
freeze_loom = pd.DataFrame(index = index, columns = columns)


for col in df_loom:
    if col.startswith("ML"):
        base = df_loom.loc[29,col]

        
        stim1 = df_loom.loc[30,col]
        stim2 = df_loom.loc[48,col]
        stim3 = df_loom.loc[66,col]
        
        freeze_loom.loc[0,col] = base
        freeze_loom.loc[1,col] = stim1
        freeze_loom.loc[2,col] = stim2
        freeze_loom.loc[3,col] = stim3
       

freeze_loom = freeze_loom.drop(columns = ['xpos'])

freeze_loom['percent'] = freeze_loom.loc[:, [x for x in freeze_loom.columns if x.startswith('ML')]].mean(axis=1)
freeze_loom['stdev'] = freeze_loom.loc[:, [x for x in freeze_loom.columns if x.startswith('ML')]].std(axis=1)
freeze_loom['SEM'] = freeze_loom['stdev']/np.sqrt(8)

freeze_loom['lower_bound'] = freeze_loom['percent'] - freeze_loom['SEM'] 
freeze_loom['upper_bound'] = freeze_loom['percent'] + freeze_loom['SEM'] 

freeze_loom['loom_nr'] = range(len(freeze_loom))
#%%
columns = ['xpos']
index = range(1)
freeze_shock = pd.DataFrame(index = index, columns = columns)


for col in df_shock:
    if col.startswith("ML"):
        base = df_shock.loc[29,col]
        stim1 = df_shock.loc[30,col]
        stim2 = df_shock.loc[48,col]
        stim3 = df_shock.loc[66,col]

        freeze_shock.loc[0,col] = base
        freeze_shock.loc[1,col] = stim1
        freeze_shock.loc[2,col] = stim2
        freeze_shock.loc[3,col] = stim3

freeze_shock = freeze_shock.drop(columns = ['xpos'])

freeze_shock['percent'] = freeze_shock.loc[:, [x for x in freeze_shock.columns if x.startswith('ML')]].mean(axis=1)
freeze_shock['stdev'] = freeze_shock.loc[:, [x for x in freeze_shock.columns if x.startswith('ML')]].std(axis=1)
freeze_shock['SEM'] = freeze_shock['stdev']/np.sqrt(8)

freeze_shock['lower_bound'] = freeze_shock['percent'] - freeze_shock['SEM'] 
freeze_shock['upper_bound'] = freeze_shock['percent'] + freeze_shock['SEM'] 

freeze_shock['loom_nr'] = range(len(freeze_shock))

#%%
ax = sns.lineplot(x='loom_nr', y='percent', data =freeze_loom, color = 'royalblue',label='')
# ax = plt.fill_between(df_loom['time_sec'],df_loom['lower_bound'] , df_loom['upper_bound'], color ='royalblue',alpha=.3)
plt.errorbar(x='loom_nr', y='percent', data =freeze_loom, yerr='SEM',color = 'dodgerblue')


ax = sns.lineplot(x='loom_nr', y='percent', data =freeze_shock, color = 'crimson',label='')
# ax = plt.fill_between(df_shock['time_sec'],df_shock['lower_bound'] , df_shock['upper_bound'], color ='crimson',alpha=.3)
plt.errorbar(x='loom_nr', y='percent', data =freeze_shock, yerr='SEM', color = 'crimson')#%%

ax = plt.gca()
ax.legend(bbox_to_anchor=(0.5, 1))
ax.set_ylim([-5, 100])
# ax.set_xlim([0,10])
ax.set_xticks([0,1,2,3])
ax.set_ylabel('Freezing %', fontsize=20)
ax.set_xlabel('tone number', fontsize=20)
ax.tick_params(labelsize=15)
ax.tick_params('both', length=5, width=2, which='major')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('conditioning: during 10-second tone')
plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/test_by_tone.jpeg',bbox_inches='tight', dpi=1800)




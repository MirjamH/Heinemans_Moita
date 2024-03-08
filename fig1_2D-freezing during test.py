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

#%%

df_shock['percent'] = df_shock.loc[:, [x for x in df_shock.columns if x.startswith('ML')]].mean(axis=1)
df_shock['stdev'] = df_shock.loc[:, [x for x in df_shock.columns if x.startswith('ML')]].std(axis=1)
df_shock['SEM'] = df_shock['stdev']/np.sqrt(8)

df_shock['lower_bound'] = df_shock['percent'] - df_shock['SEM'] 
df_shock['upper_bound'] = df_shock['percent'] + df_shock['SEM'] 



#%%
'''only high freezers included: 26, 34, 43, 44, 48, 49, 53, 58'''
columns = ['xpos']
index = range(1)
df_loom_high_freezer = pd.DataFrame(index = index, columns = columns)
for rat in df_loom:
    if ('26' in rat) | ('34' in rat) |('43' in rat) |('44' in rat) |('48' in rat) |('49' in rat) |('53' in rat) |('58' in rat) | ('time' in rat):
        animal = df_loom[rat]
        df_loom_high_freezer = pd.concat([df_loom_high_freezer, animal], axis=1, sort = True)
df_loom_high_freezer = df_loom_high_freezer.drop(columns = ['xpos'])


df_loom_high_freezer['percent'] = df_loom_high_freezer.loc[:, [x for x in df_loom_high_freezer.columns if x.startswith('ML')]].mean(axis=1)
df_loom_high_freezer['stdev'] = df_loom_high_freezer.loc[:, [x for x in df_loom_high_freezer.columns if x.startswith('ML')]].std(axis=1)
df_loom_high_freezer['SEM'] = df_loom_high_freezer['stdev']/np.sqrt(8)

df_loom_high_freezer['lower_bound'] = df_loom_high_freezer['percent'] - df_loom_high_freezer['SEM'] 
df_loom_high_freezer['upper_bound'] = df_loom_high_freezer['percent'] + df_loom_high_freezer['SEM'] 


#%%
ax = sns.lineplot(x='time_sec', y='percent', data =df_loom_high_freezer, color = 'royalblue')
ax = plt.fill_between(df_loom_high_freezer['time_sec'],df_loom_high_freezer['lower_bound'] , df_loom_high_freezer['upper_bound'], color ='royalblue',alpha=.3)

ax = sns.lineplot(x=[0,0], y =[0,100], color = 'white')
#ax.axvspan(0, 10, alpha=0.3, color='black')
#ax.axvspan(180, 190, alpha=0.3, color='black')
#ax.axvspan(360, 370, alpha=0.3, color='black')
#ax.set_title('Tone-Loom Test\n', fontsize=20)

ax.set_ylabel('% Freezing', fontsize=20)
ax.set_xlabel('Seconds', fontsize=20)
ax.tick_params(labelsize=15)
ax.set_ylim([0,100])
ax.set_xlim([-300,400])


ax = sns.lineplot(x='time_sec', y='percent', data =df_shock, color = 'crimson')
ax = plt.fill_between(df_shock['time_sec'],df_shock['lower_bound'] , df_shock['upper_bound'], color ='crimson',alpha=.3)

ax = sns.lineplot(x=[0,0], y =[0,100], color = 'white')
# ax.axvspan(0, 10, alpha=0.3, color='dimgrey')
# ax.axvspan(180, 190, alpha =0.3, color='dimgrey')
# ax.axvspan(360, 370, alpha=0.3, color='dimgrey')
ax.set_title('Tone-Shock Test\n', fontsize=20)

ax.axvline(0, alpha=0.9, ls='--', color='k')
ax.axvline(180, alpha=0.9, ls='--', color='k')
ax.axvline(360, alpha=0.9, ls='--', color='k')
ax.set_ylabel('% Freezing', fontsize=20)
ax.set_xlabel('Seconds', fontsize=20)
ax.tick_params(labelsize=15)
ax.set_ylim([0,100])
ax.set_xlim([-300,400])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.03)) 

ax.set_title('test high freezers')

plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/test_HIGHFREEZERS.jpeg',bbox_inches='tight', dpi=1800)




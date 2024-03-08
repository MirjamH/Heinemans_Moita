#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:31:00 2023

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

'''Excluded animals: LOOM: ML2, ML8, ML14,'''

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



#%%
columns = ['xpos']
index = range(41)
df_first_loom = pd.DataFrame(index = index, columns = columns)

loom_onset = 610
tone_onset = 600

for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')
    
    freeze_proportion =  freeze_loom.loc[tone_onset-10:loom_onset+20,rat_name+str('_freeze')].reset_index(drop=True)
    
    df_first_loom.loc[:,rat_name+str('_freeze')]=freeze_proportion

df_first_loom = df_first_loom.drop(columns = ['xpos'])
                
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
columns = ['xpos']
index = range(41)
df_first_shock = pd.DataFrame(index = index, columns = columns)

shock_onset = 610
tone_onset = 600

for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')
    
    freeze_proportion =  freeze_shock.loc[tone_onset-10:shock_onset+20,rat_name+str('_freeze')].reset_index(drop=True)
    
    df_first_shock.loc[:,rat_name+str('_freeze')]=freeze_proportion

df_first_shock = df_first_shock.drop(columns = ['xpos'])

#%%
'''loom'''
df_first_loom['average'] = df_first_loom.loc[:, [x for x in df_first_loom.columns if x.endswith('freeze')]].mean(axis=1)
df_first_loom['percent'] = df_first_loom['average']*100

df_first_loom['stdev'] = df_first_loom.loc[:, [x for x in df_first_loom.columns if x.endswith('freeze')]].std(axis=1)
df_first_loom['SEM'] = df_first_loom['stdev']/np.sqrt(16)*100

df_first_loom['lower_bound'] = df_first_loom['percent'] - df_first_loom['SEM'] 
df_first_loom['upper_bound'] = df_first_loom['percent'] + df_first_loom['SEM'] 
df_first_loom['seconds'] = range(len(df_first_loom))

'''shock'''
df_first_shock['average'] = df_first_shock.loc[:, [x for x in df_first_shock.columns if x.endswith('freeze')]].mean(axis=1)
df_first_shock['percent'] = df_first_shock['average']*100

df_first_shock['stdev'] = df_first_shock.loc[:, [x for x in df_first_shock.columns if x.endswith('freeze')]].std(axis=1)
df_first_shock['SEM'] = df_first_shock['stdev']/np.sqrt(8)*100

df_first_shock['lower_bound'] = df_first_shock['percent'] - df_first_shock['SEM'] 
df_first_shock['upper_bound'] = df_first_shock['percent'] + df_first_shock['SEM'] 
df_first_shock['seconds'] = range(len(df_first_loom))
#%%
'''plots'''

ax = sns.lineplot(x='seconds', y='percent', data =df_first_loom, color = 'dodgerblue').set_title('first loom\n', fontsize=20)#T-Shock
ax = plt.fill_between(df_first_loom['seconds'],df_first_loom['lower_bound'] , df_first_loom['upper_bound'], color ='dodgerblue',alpha=.3)


ax = sns.lineplot(x='seconds', y='percent', data =df_first_shock, color = 'crimson').set_title('conditioning\n', fontsize=20)#T-Shock
ax = plt.fill_between(df_first_shock['seconds'],df_first_shock['lower_bound'] , df_first_shock['upper_bound'], color ='crimson',alpha=.3)



ax = plt.gca()

ax.set_ylim([-5, 100])
# ax.set_xlim([0,10])
# ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax.set_ylabel('Freezing %', fontsize=20)
ax.set_xlabel('seconds', fontsize=20)
ax.tick_params(labelsize=15)

ax.axvspan(10, 20, alpha =0.3, color='dimgrey')
ax.axvspan(20, 21, alpha =0.6, color='black')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('zoom-in first loom')

plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/zoom_loom1.jpeg',bbox_inches='tight', dpi=1800)







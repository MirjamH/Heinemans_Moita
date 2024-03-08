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
stimbystim_loom = pd.DataFrame(index = index, columns = columns)

loom_onset = [10,610,802,1063,1225,1296,1578,1699,1921,2232,2319]
tone_onset = [value - 10 for value in loom_onset]

for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')
    for i in range(10):
        
        freeze_proportion =  freeze_loom.loc[(tone_onset[i]):(loom_onset[i]-1),rat_name+str('_freeze')].sum()/(10)
        print(rat_name)
        print(freeze_proportion)
        stimbystim_loom.loc[i,rat_name+str('_freeze')]=freeze_proportion

stimbystim_loom = stimbystim_loom.drop(columns = ['xpos'])
                
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
stimbystim_shock = pd.DataFrame(index = index, columns = columns)
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

        
        freeze_proportion =  freeze_shock.loc[(tone_onset[i]):(shock_onset[i]-1),rat_name+str('_freeze')].sum()/(10)
        # print(rat_name)
        # print(freeze_proportion)
        stimbystim_shock.loc[i,rat_name+str('_freeze')]=freeze_proportion

stimbystim_shock = stimbystim_shock.drop(columns = ['xpos'])


#%%
'''loom'''
stimbystim_loom['average'] = stimbystim_loom.loc[:, [x for x in stimbystim_loom.columns if x.endswith('freeze')]].mean(axis=1)
stimbystim_loom['percent'] = stimbystim_loom['average']*100

stimbystim_loom['stdev'] = stimbystim_loom.loc[:, [x for x in stimbystim_loom.columns if x.endswith('freeze')]].std(axis=1)
stimbystim_loom['SEM'] = stimbystim_loom['stdev']/np.sqrt(16)*100

stimbystim_loom['lower_bound'] = stimbystim_loom['percent'] - stimbystim_loom['SEM'] 
stimbystim_loom['upper_bound'] = stimbystim_loom['percent'] + stimbystim_loom['SEM'] 

stimbystim_loom['loom_nr'] = range(len(stimbystim_loom))
#%%
'''shock'''
stimbystim_shock['average'] = stimbystim_shock.loc[:, [x for x in stimbystim_shock.columns if x.endswith('freeze')]].mean(axis=1)
stimbystim_shock['percent'] = stimbystim_shock['average']*100

stimbystim_shock['stdev'] = stimbystim_shock.loc[:, [x for x in stimbystim_shock.columns if x.endswith('freeze')]].std(axis=1)
stimbystim_shock['SEM'] = stimbystim_shock['stdev']/np.sqrt(8)*100

stimbystim_shock['lower_bound'] = stimbystim_shock['percent'] - stimbystim_shock['SEM'] 
stimbystim_shock['upper_bound'] = stimbystim_shock['percent'] + stimbystim_shock['SEM'] 

stimbystim_shock['loom_nr'] = range(len(stimbystim_loom))

#%%
'''plots'''

ax = sns.lineplot(x='loom_nr', y='percent', data =stimbystim_loom, color = 'dodgerblue').set_title('conditioning\n', fontsize=20)#T-Shock
plt.errorbar(x='loom_nr', y='percent', data =stimbystim_loom, yerr='SEM',color = 'dodgerblue')


ax = sns.lineplot(x='loom_nr', y='percent', data =stimbystim_shock, color = 'crimson').set_title('conditioning\n', fontsize=20)#T-Shock
plt.errorbar(x='loom_nr', y='percent', data =stimbystim_shock, yerr='SEM',color = 'crimson')



ax = plt.gca()

ax.set_ylim([-0, 100])
# ax.set_xlim([0,10])
ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax.set_ylabel('Freezing %', fontsize=20)
ax.set_xlabel('freezing during tone', fontsize=20)
ax.tick_params(labelsize=15)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('conditioning: during 10-second tone')

plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/training_by_tone.jpeg',bbox_inches='tight', dpi=1800)



#%%


'''Statistics freezing first 43looms and last 3 inter loom intervals:'''
columns = ['xpos']
index = range(1)
df_loom1_3 = stimbystim_loom.loc[1:3,:]
df_loom7_10 = stimbystim_loom.loc[7:,:]


#%%
df_loom1_3.loc['average_loom1_3',:] = df_loom1_3.mean(axis=0)
df_loom7_10.loc['average_loom7_10',:] = df_loom7_10.mean(axis=0)

#%%

df_mean_freezing_start = df_loom1_3.loc['average_loom1_3',:]
df_mean_freezing_end = df_loom7_10.loc['average_loom7_10',:] 


#%%
''' statistics shock'''

columns = ['xpos']
index = range(1)
df_shock1_3 = stimbystim_shock.loc[1:3,:]
df_shock7_10 = stimbystim_shock.loc[7:,:]


#%%
df_shock1_3.loc['average_loom1_3',:] = df_shock1_3.mean(axis=0)
df_shock7_10.loc['average_loom7_10',:] = df_shock7_10.mean(axis=0)

#%%

df_mean_shock_start = df_shock1_3.loc['average_loom1_3',:]
df_mean_shock_end = df_shock7_10.loc['average_loom7_10',:] 









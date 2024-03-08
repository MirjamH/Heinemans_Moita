#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:20:58 2023

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
'''Tone-Loom'''
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


firstloom_train_test = pd.DataFrame(index = index, columns = columns)

loom_onset = [610]
tone_onset = [value - 10 for value in loom_onset]

for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')

    
    freeze_proportion =  (freeze_loom.loc[(tone_onset[0]):(loom_onset[0]-1),rat_name+str('_freeze')].mean() -freeze_loom.loc[(tone_onset[0]-10):(tone_onset[0]-1),rat_name+str('_freeze')].mean())*100
    print(rat_name)
    print(freeze_proportion)
    firstloom_train_test.loc['train_loom',rat_name]=freeze_proportion

firstloom_train_test = firstloom_train_test.drop(columns = ['xpos'])

#%%
df_loom = pd.read_csv('/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/loom/test/Loom_test_freeze.csv')        

df_loom.drop('ML2', axis=1, inplace=True)
df_loom.drop('ML8', axis=1, inplace=True)
# df_loom.drop('ML14', axis=1, inplace=True)


for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')

    
    freeze_proportion =  df_loom.loc[30,rat_name]
    print(rat_name)
    print(freeze_proportion)
    firstloom_train_test.loc['test_loom',rat_name]=freeze_proportion


#%%
'''Tone-Shock'''

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

firstshock_train_test = pd.DataFrame(index = index, columns = columns)
for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')
    shock_onset = freeze_shock.loc[freeze_shock[rat_name+str('_stim')]>0.27].index.values.astype(int)
    
 
    tone_onset = [value - 10 for value in shock_onset]
    print(len(tone_onset))

    
    freeze_proportion =  (freeze_shock.loc[(tone_onset[0]):(shock_onset[0]-1),rat_name+str('_freeze')].mean() -freeze_shock.loc[(tone_onset[0]-10):(tone_onset[0]-1),rat_name+str('_freeze')].mean())*100
    print(rat_name)
    # print(rat_name)
    # print(freeze_proportion)
    firstshock_train_test.loc['train_shock',rat_name]=freeze_proportion


#%%

df_shock = pd.read_csv('/Users/mirjamheinemans/Desktop/Annotator python tryout/experiment_1_freezing/shock/test/Shock_test_freeze.csv')        
for file_names in sorted(os.listdir(path_name)): 
    if file_names == '.DS_Store':
        continue
    rat_name = file_names.replace('.csv', '')

    
    freeze_proportion =  df_shock.loc[30,rat_name]
    print(rat_name)
    print(freeze_proportion)
    firstshock_train_test.loc['test_shock',rat_name]=freeze_proportion

#%%
loom_train_test = firstloom_train_test.T
shock_train_test = firstshock_train_test.T
#%%
#loom pre and post
before_l = loom_train_test['train_loom']
after_l =loom_train_test['test_loom']
fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):
    
    # plotting the points
    plt.scatter(np.zeros(len(before_l)), before_l, color = 'royalblue', s = 80, zorder = 2)
    plt.scatter(np.ones(len(after_l)), after_l, color = 'royalblue', s = 80, zorder = 2)
 
    # plotting the lines
    for i in range(len(before_l)):
        plt.plot( [0,1], [before_l[i], after_l[i]], c='orange', zorder =1)
        
    plt.xticks([0,1], ['tr_s', 'te_s'])
    plt.tick_params(labelsize=20)
    plt.ylim([-60,100])
    plt.xlim([-0.1,1.1])
    plt.tick_params('both', length=5, width=2, which='major')
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2
    
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    
    plt.tick_params(labelsize=20)
    sns.despine()
    plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/LoomtoneIncrease_train_test.jpeg',bbox_inches='tight', dpi=1800)
    plt.show()

# shock pre and post

before_s = shock_train_test['train_shock']
after_s =shock_train_test['test_shock']
fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):
    
    # plotting the points
    plt.scatter(np.zeros(len(before_s)), before_s, color = 'crimson', s = 80, zorder = 2)
    plt.scatter(np.ones(len(after_s)), after_s, color = 'crimson', s = 80, zorder = 2)
 
    # plotting the lines
    for i in range(len(before_s)):
        plt.plot( [0,1], [before_s[i], after_s[i]], c='orange', zorder =1)
        
    plt.xticks([0,1], ['tr_s', 'te_s'])
    plt.tick_params(labelsize=20)
    plt.ylim([-60,100])
    plt.xlim([-0.1,1.1])
    plt.tick_params('both', length=5, width=2, which='major')
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2
    
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    
    plt.tick_params(labelsize=20)
    sns.despine()
    plt.savefig('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/graphs/ShocktoneIncrease_train_test.jpeg',bbox_inches='tight', dpi=1800)
    plt.show()

#%%

result_loom,p_loom= ss.wilcoxon(x=loom_train_test['train_loom'],y=loom_train_test['test_loom'])

result_shock,p_shock= ss.wilcoxon(x=shock_train_test['train_shock'],y=shock_train_test['test_shock'])

print('For loom \n  W is:',result_loom,' p-value:',p_loom)
print('For shock \n  W is:',result_shock,' p-value:',p_shock)

#%%
result_training, p_training = ss.mannwhitneyu(x=loom_train_test['train_loom'],y=shock_train_test['train_shock'])
print('For training U=',result_training,'p-value=',p_training)


result_test, p_test = ss.mannwhitneyu(x=loom_train_test['test_loom'],y=shock_train_test['test_shock'])
print('For test U=',result_test,'p-value=',p_test)



result_trainShock_testLoom, p_trainShocktestLoom = ss.mannwhitneyu(x=loom_train_test['test_loom'],y=shock_train_test['train_shock'])
print('For training shock vs test loom: U=',result_trainShock_testLoom,'p-value=',p_trainShocktestLoom)


result_trainLoom_testShock, p_trainShocktestLoom = ss.mannwhitneyu(x=loom_train_test['train_loom'],y=shock_train_test['test_shock'])
print('For training loom vs test shock: U=',result_trainLoom_testShock,'p-value=',p_trainShocktestLoom)



#%%
loom_train_test['increase'] = loom_train_test['test_loom'] -loom_train_test['train_loom']
median_increase_loom = loom_train_test['increase'].median()

shock_train_test['increase'] = shock_train_test['test_shock'] -shock_train_test['train_shock']
median_increase_shock = shock_train_test['increase'].median()




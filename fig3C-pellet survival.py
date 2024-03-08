#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:45:57 2019

@author: mirjamheinemans

TRAINING columns:
    MH33,
    MH33_in_shelter
    MH33_doorway
    MH33_with_pellet
    MH33_eat_pellet
    MH33_freeze
    MH33_reaching
    MH33_scanning
    MH33_new_pellet

TEST columns:
    0. ''
    1.x-value33,
    2.MH33_in_shelter
    3.MH33_doorway
    4.MH33_with_pellet
    5.MH33_eat_pellet
    6.MH33_freeze
    7.MH33_reaching
    8.MH33_scanning
    9.MH33_stim
"""
#%%
import scipy.stats as ss
import csv 
import numpy as np
import os, glob # Operating System
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import json
sns.set_style("ticks")
#%%

'''Here I turn the test survival into a dataframe'''

path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'
def PelletTest(file_names):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name +'/' + file_names +'/' + 'test.csv', usecols = [2,4,9])# 2 = in_shelter, 4 = with pellet, 9 = stimulation
        stim = dataset.loc[dataset.iloc[:,-1] == 1].index.values.astype(int)[0]  
        dataset_stim = dataset.iloc[int(stim-5):,:].reset_index(drop=True) #-5 frames to ensure no animal took it at 0 seconds
        
        number =file_names.replace('MH',"")
        
        if int(number) <39: # 60 FPS
            dataset_sec = dataset_stim.groupby(np.arange(len(dataset_stim))//60).mean()

        elif 39 < int(number) <71: #  90 FPS videos
            dataset_sec = dataset_stim.groupby(np.arange(len(dataset_stim))//90).mean()
            
        else: # 60 FPS videos
            dataset_sec = dataset_stim.groupby(np.arange(len(dataset_stim))//60).mean()
        
        dataset_sec =  dataset_sec.iloc[:700,:]
        dataset_sec.iloc[-1:,1] = 1
        pellet = dataset_sec.loc[dataset_sec.iloc[:,1]  > 0].index.values.astype(int)[0]
                
        dataset_sec[file_names+'taken'] = 1    
        
        dataset_sec.iloc[int(pellet):,3] = 0
        
        survival = dataset_sec.iloc[:,3]
        return(survival)
     


#%%
'''Tone-Loom'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_T_Loom'
columns = ['xpos']
index = range(10)
shelter_loom_test = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = PelletTest(file_names)  
    shelter_loom_test = pd.concat([shelter_loom_test, animal], axis=1)
shelter_loom_test = shelter_loom_test.drop(columns = ['xpos'])
#shelter_loom_test = shelter_loom_test.drop(index = 0)
shelter_loom_test = shelter_loom_test.apply(pd.to_numeric, errors='coerce')
shelter_loom_test = shelter_loom_test.fillna(0)

loom_survival_test = []
for i in shelter_loom_test.iterrows():
    survived = sum(i[1]) / len(i[1])
    loom_survival_test.append(survived)
survival_loom_test = pd.DataFrame(loom_survival_test, columns = ['TL_test'])   
survival_loom_test['t'] = range(len(survival_loom_test))

#%%
'''Tone-Shock'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_T_Shock'
columns = ['xpos']
index = range(10)
shelter_shock_test = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = PelletTest(file_names)  
    shelter_shock_test = pd.concat([shelter_shock_test, animal], axis=1)
shelter_shock_test = shelter_shock_test.drop(columns = ['xpos'])
#shelter_shock_test = shelter_shock_test.drop(index = 0)
shelter_shock_test = shelter_shock_test.apply(pd.to_numeric, errors='coerce')
shelter_shock_test = shelter_shock_test.fillna(0)

survival_shock_test = []
for i in shelter_shock_test.iterrows():
    survived = sum(i[1]) / len(i[1])
    survival_shock_test.append(survived)
survival_shock_test = pd.DataFrame(survival_shock_test, columns = ['TS_test'])   
survival_shock_test['t'] = range(len(survival_shock_test))

#%%
'''Tone'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Tone'
columns = ['xpos']
index = range(10)
shelter_tone_test = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = PelletTest(file_names)  
    shelter_tone_test = pd.concat([shelter_tone_test, animal], axis=1)
shelter_tone_test = shelter_tone_test.drop(columns = ['xpos'])
# shelter_tone_test = shelter_tone_test.drop(index = 0)
shelter_tone_test = shelter_tone_test.apply(pd.to_numeric, errors='coerce')
shelter_tone_test = shelter_tone_test.fillna(0)

survival_tone_test = []
for i in shelter_tone_test.iterrows():
    survived = sum(i[1]) / len(i[1])
    survival_tone_test.append(survived)
survival_tone_test = pd.DataFrame(survival_tone_test, columns = ['T_test'])   
survival_tone_test['t'] = range(len(survival_tone_test))

#%%
# '''Here I turn the training survival into a dataframe'''

# path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'
# def PelletTrain(file_names):

#     if file_names == '.DS_Store':
#         next
#     else:
#         dataset = pd.read_csv(path_name +'/' + file_names +'/' + 'training.csv', usecols = [2,4,9])# 2 = in_shelter
#         dataset.iloc[-10:,-1] = 1
#         end_exp = dataset.iloc[:,2].diff()[dataset.iloc[:,2].diff() == 1].index.values[-1] 
#         dataset_end = dataset.iloc[:int(end_exp - 1),:].reset_index(drop=True)      
        
#         pellet3 = dataset_end.iloc[:,2].diff()[dataset_end.iloc[:,2].diff() == -1].index.values[-1]
#         dataset_cut = dataset.iloc[int(pellet3):,:].reset_index(drop=True)  
        
#         out_shelter = dataset_cut.iloc[:,0].diff()[dataset_cut.iloc[:,0].diff() == -1].index.values[0] 
#         df_final = dataset_cut.iloc[int(out_shelter):,:].reset_index(drop=True)  
        
#         number =file_names.replace('MH',"")
        
#         if int(number) <39: # 30 FPS
#             dataset_sec = df_final.groupby(np.arange(len(df_final))//30).mean()

#         elif 39 < int(number) < 45: # 60 FPS videos
#             dataset_sec = df_final.groupby(np.arange(len(df_final))//60).mean()
 
#         elif 45 < int(number) <71: #  90 FPS videos
#             dataset_sec = df_final.groupby(np.arange(len(df_final))//90).mean()
            
#         else: # 60 FPS videos
#             dataset_sec = df_final.groupby(np.arange(len(df_final))//60).mean()
            
             
#         pellet = dataset_sec.loc[dataset_sec.iloc[:,1] > 0].index.values.astype(int)[0]
        
#         dataset_sec[file_names+'taken'] = 1    
        
#         pellet = dataset_sec.loc[dataset_sec.iloc[:,1] > 0].index.values.astype(int)[0]
        
#         dataset_sec.iloc[int(pellet):,3] = 0
        
#         survival = dataset_sec.iloc[:,3]
#         return(survival)
     
# #%%
# '''Tone-Loom'''
# path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_T_Loom'
# columns = ['xpos']
# index = range(10)
# shelter_loom_train = pd.DataFrame(index = index, columns = columns)

# for file_names in sorted(os.listdir(path_name)): 
#     print(file_names)
#     animal = PelletTrain(file_names)  
#     shelter_loom_train = pd.concat([shelter_loom_train, animal], axis=1)
# shelter_loom_train = shelter_loom_train.drop(columns = ['xpos'])
# shelter_loom_train = shelter_loom_train.drop(index = 0)
# shelter_loom_train = shelter_loom_train.apply(pd.to_numeric, errors='coerce')
# shelter_loom_train = shelter_loom_train.fillna(0)

# loom_survival_train = []
# for i in shelter_loom_train.iterrows():
#     survived = sum(i[1]) / len(i[1])
#     loom_survival_train.append(survived)
# survival_loom_train = pd.DataFrame(loom_survival_train, columns = ['TL_train'])   
# survival_loom_train['t'] = range(len(survival_loom_train))

# #%%
# '''Tone-Shock'''
# path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_T_Shock'
# columns = ['xpos']
# index = range(10)
# shelter_shock_train = pd.DataFrame(index = index, columns = columns)

# for file_names in sorted(os.listdir(path_name)): 
#     print(file_names)
#     animal = PelletTrain(file_names)  
#     shelter_shock_train = pd.concat([shelter_shock_train, animal], axis=1)
# shelter_shock_train = shelter_shock_train.drop(columns = ['xpos'])
# shelter_shock_train = shelter_shock_train.drop(index = 0)
# shelter_shock_train = shelter_shock_train.apply(pd.to_numeric, errors='coerce')
# shelter_shock_train = shelter_shock_train.fillna(0)

# survival_shock_train = []
# for i in shelter_shock_train.iterrows():
#     survived = sum(i[1]) / len(i[1])
#     survival_shock_train.append(survived)
# survival_shock_train = pd.DataFrame(survival_shock_train, columns = ['TS_train'])   
# survival_shock_train['t'] = range(len(survival_shock_train))

# #%%
# '''Tone'''
# path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Tone'
# columns = ['xpos']
# index = range(10)
# shelter_tone_train = pd.DataFrame(index = index, columns = columns)

# for file_names in sorted(os.listdir(path_name)): 
#     print(file_names)
#     animal = PelletTrain(file_names)  
#     shelter_tone_train = pd.concat([shelter_tone_train, animal], axis=1)
# shelter_tone_train = shelter_tone_train.drop(columns = ['xpos'])
# shelter_tone_train = shelter_tone_train.drop(index = 0)
# shelter_tone_train = shelter_tone_train.apply(pd.to_numeric, errors='coerce')
# shelter_tone_train = shelter_tone_train.fillna(0)
# survival_tone_train = []
# for i in shelter_tone_train.iterrows():
#     survived = sum(i[1]) / len(i[1])
#     survival_tone_train.append(survived)
# survival_tone_train = pd.DataFrame(survival_tone_train, columns = ['T_train'])   
# survival_tone_train['t'] = range(len(survival_tone_train))

#%%
'''Putting together the test and training dataframes together'''


cond_train_test = pd.concat([survival_loom_test, survival_shock_test, survival_tone_test], axis = 1)
cond_train_test['sec'] =  range(len(cond_train_test))
#%%

#sns.set(rc={'figure.figsize':(4.5,6)})
sns.set_style(style='ticks')
fig = plt.figure(figsize=(3,4))
#ax = sns.lineplot(x ='sec', y='TL_train', data =cond_train_test, color = 'blue')
#ax = sns.lineplot(x ='sec', y='TS_train', data =cond_train_test, color = 'red')
#ax = sns.lineplot(x ='sec', y='T_train', data =cond_train_test, color = 'green')

ax = sns.lineplot(x ='sec', y='T_test', data =cond_train_test, color = 'green',  linewidth=2.5)
ax = sns.lineplot(x ='sec', y='TL_test', data =cond_train_test, color = 'blue',  linewidth=2.5)
ax = sns.lineplot(x ='sec', y='TS_test', data =cond_train_test, color = 'red',  linewidth=2.5)

#ax.set_title('Survival of pellet')#
ax.set_ylabel('', fontsize = 20)
#ax.lines[0].set_linestyle("--")
#ax.lines[1].set_linestyle("--")
#ax.lines[2].set_linestyle("--")
ax.set_ylim([-0.01,1.01])
ax.set_xlim([-6,600]) # the animals with 90 FPS are only saved until 600 seconds
ax.legend(loc='upper right', labels=['Tone-Loom', 'Tone-Shock','Tone'])
ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.03)) 

ax.tick_params(labelsize=20)
ax.set_xlabel('', fontsize = 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.yaxis.set_tick_params(size=5, width=2)
ax.xaxis.set_tick_params(size=5, width=2, color='k')
ax.xaxis.set_ticks(np.arange(-0, 601, 200))
os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

plt.savefig('cond_survival.png',bbox_inches='tight', dpi=1800, transparent=True)
# #%%
# '''Tone-Loom'''
# sns.set_style(style='white')
# ax = sns.lineplot(x ='sec', y='TL_test', data =cond_train_test, color = 'blue')
# ax = sns.lineplot(x ='sec', y='TL_train', data =cond_train_test, color = 'blue')

# ax.set_title('Tone-Loom\nSurvival of pellet')#
# ax.set_ylabel('fraction of animals without pellet')#

# ax.lines[1].set_linestyle("--")
# ax.set_ylim([0.0,1.1])
# ax.set_xlim([0,599]) # the animals with 90 FPS are only saved until 600 seconds
# ax.legend(loc='upper right', labels=['Test', 'Training'])

# #%%
# ax = sns.lineplot(x ='sec', y='TS_test', data =cond_train_test, color = 'red')
# ax = sns.lineplot(x ='sec', y='TS_train', data =cond_train_test, color = 'red')

# ax.set_title('Tone-Shock\nSurvival of pellet')#
# ax.set_ylabel('fraction of animals without pellet')#

# ax.lines[1].set_linestyle("--")
# ax.set_ylim([0.0,1.1])
# ax.set_xlim([0,599]) # the animals with 90 FPS are only saved until 600 seconds
# ax.legend(loc='upper right', labels=['Test', 'Training'])

# #%%
# ax = sns.lineplot(x ='sec', y='T_test', data =cond_train_test, color = 'green')
# ax = sns.lineplot(x ='sec', y='T_train', data =cond_train_test, color = 'green')# this is training, comes from LoomPelletSurvivalTrain.py
# ax.set_title('Tone\nSurvival of pellet')#
# ax.set_ylabel('fraction of animals without pellet')#

# ax.lines[1].set_linestyle("--")
# ax.set_ylim([0.0,1.1])
# ax.set_xlim([0,599]) # the animals with 90 FPS are only saved until 600 seconds
# ax.legend(loc='upper right', labels=['Test', 'Training'])

# #%%
# '''
# # Try-out with Kaplan Meyer statistics
# # '''
# # from lifelines import KaplanMeierFitter
# # pellet = pd.read_csv('/Users/mirjamheinemans/Desktop/Annotator python tryout/pellet_survival.csv',delimiter=',', low_memory=False, index_col=False)        

# # group1=pellet.drop('Training', axis=1)
# # group1 = group1.dropna()
# # group2=pellet.drop('Test', axis=1)
# # group2 = group2.dropna()

# # T=group1['Seconds']
# # E=group1['Test']


# # T1=group2['Seconds']
# # E1=group2['Training']



# # kmf = KaplanMeierFitter()

# # ax = plt.subplot(111)
# # ax = kmf.fit(T, E, label="Group 1-Test").plot(ax=ax)
# # ax = kmf.fit(T1, E1, label="Group 2-Training").plot(ax=ax)
# # ax.set_title('Pellet survival time')
# # #%%
# # #logrank_test
# # from lifelines.statistics import logrank_test
# # #%%
# # results=logrank_test(T,T1,event_observed_A=E, event_observed_B=E1)
# # results.print_summary()


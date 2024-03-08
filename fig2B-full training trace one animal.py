#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:53:27 2019

@author: mirjamheinemans

Individual traces of the animals during the whole test


TRAINING columns:
    0.
    1.MH33,
    2.MH33_in_shelter
    3.MH33_doorway
    4.MH33_with_pellet
    5.MH33_eat_pellet
    6.MH33_freeze
    7.MH33_reaching
    8.MH33_scanning
    9.MH33_new_pellet

TEST columns:
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

path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'


def Trace(file_names):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name +'/' + file_names +'/' + 'training.csv', usecols = [1,4,9])
        if int(file_names[2:]) >45:
           dataset = dataset.groupby(np.arange(len(dataset.index))//2).mean()
             
        dataset['took_pell'+file_names] = -1000
        
        took_pell = dataset.iloc[:,1].diff()[dataset.iloc[:,1].diff() > 0].index.values#[0:4]
        dataset.iloc[took_pell,-1] = dataset.iloc[took_pell,0]
        
        return(dataset)
        

#%%
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'
columns = ['xpos']
index = range(900)
shelter_loom = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = Trace(file_names)  
    shelter_loom = pd.concat([shelter_loom, animal], axis=1)
shelter_loom = shelter_loom.drop(columns = ['xpos'])
shelter_loom = shelter_loom.drop(index = 0)
shelter_loom['frames']=  range(len(shelter_loom))
shelter_loom['sec'] =shelter_loom['frames']/30 

export_csv = shelter_loom.to_csv ('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1/example_traceMH85.csv', header=True)  

                
        
#%%
'''Loom'''
fig = plt.figure(figsize=(5,4))
#ax = sns.lineplot(x='sec', y='x-value33', data =shelter_loom, color = 'black', zorder=1).set_title('Loom MH33')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH033', data = shelter_loom, color = 'c',s =150,zorder = 2)
##
##
#ax = sns.lineplot(x='sec', y='x-value34', data =shelter_loom, color = 'black', zorder=1).set_title('Loom MH34')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH034', data = shelter_loom, color = 'c',s =150,zorder = 2)

#
#ax = sns.lineplot(x='sec', y='x-value46', data =shelter_loom, color = 'black', zorder=1).set_title('Loom MH46')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH046', data = shelter_loom, color = 'c',s =150,zorder = 2)
#
#
#ax = sns.lineplot(x='sec', y='x-value59', data =shelter_loom, color = 'black', zorder=1).set_title('Loom MH59')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH059', data = shelter_loom, color = 'c',s =150,zorder = 2)
#
##
##ax = sns.lineplot(x='sec', y='x-value60', data =shelter_loom, color = 'black', zorder=1).set_title('Loom MH60')#Loom
##ax = sns.scatterplot(x = 'sec', y = 'took_pellMH060', data = shelter_loom, color = 'c',s =150,zorder = 2)

#
#ax = sns.lineplot(x='sec', y='x-value65', data =shelter_loom, color = 'black', zorder=1).set_title('Loom MH65')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH065', data = shelter_loom, color = 'c',s =150,zorder = 2)
#
#
#ax = sns.lineplot(x='sec', y='x-value66', data =shelter_loom, color = 'orange', zorder=1).set_title('Loom MH66')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH066', data = shelter_loom, color = 'c',s =150,zorder = 2)
#
#
#'''Loom'''
ax = sns.lineplot(x='sec', y='x-value85', data =shelter_loom, color = 'k', zorder = 1).set_title('Loom MH85')#Loom
ax = sns.scatterplot(x = 'sec', y = 'took_pellMH085', data = shelter_loom, color = 'pink',s =150,zorder = 2)
#
#
#ax = sns.lineplot(x='sec', y='x-value86', data =shelter_loom, color = 'k', zorder = 1)#.set_title('Loom MH86')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH086', data = shelter_loom, color = 'c',s =150,zorder = 2)
##
#
#ax = sns.lineplot(x='sec', y='x-value87', data =shelter_loom, color = 'k', zorder = 1).set_title('Loom MH87')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH087', data = shelter_loom, color = 'c',s =150,zorder = 2)
#
#
#ax = sns.lineplot(x='sec', y='x-value88', data =shelter_loom, color = 'k', zorder = 1).set_title('Loom MH88')#Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH088', data = shelter_loom, color = 'c',s =150,zorder = 2)

#ax.set_xlim([0, 20])
#ax.set_ylim([0, 800])



ax.axhspan(10, -100, alpha=0.4, color='maroon')
ax.set_ylim([-20, 1200])
ax.set_xlim([0, 500])
ax.set_title('Position & pellet retrieval\n', fontsize = 20)
ax.tick_params(labelsize=15)
ax.set_xlabel('Seconds', fontsize = 15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel("")

#%%
'''Tone-Loom'''

#ax = sns.lineplot(x='sec', y='x-value43', data =shelter_t_loom, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH043', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH43')
#
#
#
#ax = sns.lineplot(x='sec', y='x-value44', data =shelter_t_loom, color = 'orange',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH044', data = shelter_t_loom, color = 'c',s =100,zorder = 2)#.set_title('Tone-Loom MH44')
#ax.set_xlim([0,350])
#ax.set_title('Tone-Loom MH44')
#
#ax = sns.lineplot(x='sec', y='x-value51', data =shelter_t_loom, color = 'orange',zorder = 1)#T-Loom
#ax.set_ylabel('x-position (a.u.)')
#ax.set_ylim([0, 1200])
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH051', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH51')
#
#
#ax = sns.lineplot(x='sec', y='x-value52', data =shelter_t_loom, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH052', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH52')
#
#
#ax = sns.lineplot(x='sec', y='x-value53', data =shelter_t_loom, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH053', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH53')
#
#
#ax = sns.lineplot(x='sec', y='x-value63', data =shelter_t_loom, color = 'orange',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH063', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH63')
#
#
#ax = sns.lineplot(x='sec', y='x-value64', data =shelter_t_loom, color = 'orange',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH064', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH64')
#
#
#ax = sns.lineplot(x='sec', y='x-value70', data =shelter_t_loom, color = 'orange',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH070', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH70')
#
#
#
#ax = sns.lineplot(x='sec', y='x-value71', data =shelter_t_loom, color = 'orange',zorder = 1).set_title('Tone-Loom MH71')#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH071', data = shelter_t_loom, color = 'c',s =100,zorder = 2)
#
#
#ax = sns.lineplot(x='sec', y='x-value78', data =shelter_t_loom, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH078', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH78')
##
##
#ax = sns.lineplot(x='sec', y='x-value92', data =shelter_t_loom, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH092', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH92')
#
#
#ax = sns.lineplot(x='sec', y='x-value96', data =shelter_t_loom, color = 'orange',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH096', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_title('Tone-Loom MH96')
#
#
#ax = sns.lineplot(x='sec', y='x-value101',data =shelter_t_loom, color = 'orange',zorder = 1).set_title('Tone-Loom MH101')
#ax = sns.scatterplot(x = 'sec', y = 'took_pellMH101', data = shelter_t_loom, color = 'c',s =100,zorder = 2).set_ylabel('x-position (a.u.)')#T-Loom

#ax = sns.lineplot([10 ,10], [0, 1200],color ='gray')

#ax.set_xlim([0, 20])
#ax.set_ylim([0, 800])
#ax.set_ylabel('x-position (a.u.)')
#ax.plot([10 ,10], [0, 1200],color ='gray')
#ax.plot([1 ,1], [0, 1200],color ='gray')
ax.set_ylabel('x-position (a.u.)')
ax.set_ylim([0, 1200])




#%%
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_T_Shock'
file_names = 'MH068'
#%%
dataset = pd.read_csv(path_name +'/' + file_names +'/' + 'training.csv', usecols = [1,4,9])
stimulus = dataset.loc[dataset.iloc[:,2] == 1].index.values.astype(int)  
dataset_stim = dataset[~dataset.index.isin(list(stimulus))] # this filters out the frames where I'm putting a new pellet
#dataset_stim = dataset_stim.reset_index(drop=True)

#if int(file_names[2:]) >45:
#   dataset_stim = dataset_stim.groupby(np.arange(len(dataset_stim.index))//2).mean()
     
dataset_stim['took_pell'+file_names] = 0

took_pell = dataset_stim.iloc[:,1].diff()[dataset_stim.iloc[:,1].diff() > 0].index.values#[0:4]
dataset_stim.iloc[took_pell,-1] = dataset_stim.iloc[took_pell,0]
dataset_stim['frames']=  range(len(dataset_stim))     
  #%%
fig = plt.figure(figsize=(5,4))
ax = sns.lineplot(x='frames', y='x-value68', data =dataset_stim, color = 'red',zorder = 1).set_title('Tone-Shock MH68')#T-Shock
ax = sns.scatterplot(x = 'frames', y = 'took_pellMH068', data = dataset_stim, color = 'b',s =100,zorder = 2)
ax.set_ylim([0, 1200])
ax.set_xlim([13000, 15000])




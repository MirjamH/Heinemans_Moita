#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:26:53 2019

@author: mirjamheinemans



TRAINING columns:
    0.unnamed column
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
    0.unnamed column
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
        dataset = pd.read_csv(path_name +'/' + file_names +'/' + 'training.csv', usecols = [1,4,9])# 2 = in_shelter
        dataset.iloc[-10:,-1] = 1
        end_exp = dataset.iloc[:,2].diff()[dataset.iloc[:,2].diff() == 1].index.values[-1]  # col = 9 -- value going from 0 to 1, difference < 0
        dataset_end = dataset.iloc[:int(end_exp - 1),:].reset_index(drop=True)      
        
        # to make the DF start after 3rd pellet is put in arena
        pellet3 = dataset_end.iloc[:,2].diff()[dataset_end.iloc[:,2].diff() == -1].index.values[-1]
        dataset_cut = dataset_end.iloc[int(pellet3):,:].reset_index(drop=True)  
        
        
        number =file_names.replace('MH',"")
        
        if int(number) <39: # 30 FPS --> everything brought back to 30 FPS
            dataset_sec = dataset_cut
        
        elif 39 < int(number) < 45: # 60 FPS videos
            dataset_sec = dataset_cut.groupby(np.arange(len(dataset_cut))//2).mean().reset_index(drop=True)  
         
        elif 45 < int(number) <71: #  90 FPS videos]
            dataset_sec = dataset_cut.groupby(np.arange(len(dataset_cut))//3).mean().reset_index(drop=True)  
            
        else: # 60 FPS videos
            dataset_sec = dataset_cut.groupby(np.arange(len(dataset_cut))//2).mean().reset_index(drop=True)  
        
        pellet = dataset_sec.iloc[:,1].diff()[dataset_sec.iloc[:,1].diff() > 0].index.values.astype(int)[0] # col = 1 -- alue going from 0 to 1, difference > 0        
        dataset_final = dataset_sec.iloc[(int(pellet)-300) :(int(pellet) + 300),:].reset_index(drop=True)  
        
        position = dataset_final.iloc[300, 0]
        
        dataset_final[file_names] = dataset_final.iloc[:,0] - int(position) 
        dataset_final[file_names] = (dataset_final[file_names]/ 6)+100
        sheltertime = dataset_final.loc[:,file_names]

       
        return(sheltertime)

        

#%%
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'
columns = ['xpos']
index = range(10)
shelter_loom = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = Trace(file_names)  
    shelter_loom = pd.concat([shelter_loom, animal], axis=1)
shelter_loom = shelter_loom.drop(columns = ['xpos'])
shelter_loom = shelter_loom.drop(index = 0)

shelter_loom['frames'] = range(len(shelter_loom))
shelter_loom['sec'] = (shelter_loom['frames']-300)/30


grabbing = shelter_loom.loc[shelter_loom.loc[:,'sec']==0].index.values[0]
shelter_loom['pellet'] = -100
shelter_loom.loc[grabbing,'pellet'] = 100
#%%
#sns.set_style("white")
sns.set_style("ticks")
fig = plt.figure(figsize=(5,4))

'''Loom'''
ax = sns.lineplot(x='sec', y='MH033', data =shelter_loom, color = 'black')#.set_title('Traing\nLoom')#Loom
ax = sns.lineplot(x='sec', y='MH034', data =shelter_loom, color = 'black')#Loom going on after pellet
ax = sns.lineplot(x='sec', y='MH046', data =shelter_loom, color = 'black')#Loom
ax = sns.lineplot(x='sec', y='MH059', data =shelter_loom, color = 'black')#Loom
ax = sns.lineplot(x='sec', y='MH065', data =shelter_loom, color = 'black')#Loom
ax = sns.lineplot(x='sec', y='MH066', data =shelter_loom, color = 'black')#Loom

##
##'''Loom'''
ax = sns.lineplot(x='sec', y='MH085', data =shelter_loom, color = 'black')# batch 2')#Loom
ax = sns.lineplot(x='sec', y='MH086', data =shelter_loom, color = 'black')#Loom
ax = sns.lineplot(x='sec', y='MH087', data =shelter_loom, color = 'black')#Loom
ax = sns.lineplot(x='sec', y='MH088', data =shelter_loom, color = 'black')#Loom going on after pellet
ax = sns.scatterplot(x='sec', y='pellet',data =shelter_loom, color ='orange', s = 150,zorder =3)

#ax2 = ax.twinx() # create a second y axis

ax.axhspan(2, -100, alpha=0.4, color='maroon')
ax.set_ylim([-20, 200])

ax.set_xlim([-10, 10])
ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.03)) 
ax.tick_params(labelsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.yaxis.set_tick_params(size=5, width=2)
ax.xaxis.set_tick_params(size=5, width=2)
ax.set_xlabel('', fontsize = 20)
ax.set_ylabel('', fontsize = 20)

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

plt.savefig('loom_pellet_aligned.jpeg',bbox_inches='tight', dpi=1800)

export_csv = shelter_loom.to_csv ('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1/before_after_pellet_traces.csv', header=True)  



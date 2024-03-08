#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:25:30 2020

@author: mirjamheinemans
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:35:29 2019

@author: mirjamheinemans
"""

"""
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
sns.set_style("whitegrid", {'axes.grid' : False})

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/analysis files') # to change directory Read csv files with Pandas
#%%

path_name = '/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/_Loom'


def ShelterTime(file_names):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name +'/' + file_names +'/' + 'test.csv', usecols = [1,4,9])
        stimulus = dataset.loc[dataset.iloc[:,2] == 1].index.values.astype(int)[0]  

        number =file_names.replace('MH',"")
               
        if int(number) <39: #630 FPS during test
           dataset_stim = dataset.iloc[int(stimulus - 600):int(stimulus + 5400),:].reset_index(drop=True)
           dataset_sec = dataset_stim.groupby(np.arange(len(dataset_stim))//2).mean()
    
        elif 39 < int(number) <71: #  90 FPS videos
           dataset_stim = dataset.iloc[int(stimulus - 900):int(stimulus + 8100),:].reset_index(drop=True)
           dataset_sec = dataset_stim.groupby(np.arange(len(dataset_stim))//3).mean()
            
        else: # 60 FPS videos
           dataset_stim = dataset.iloc[int(stimulus - 600):int(stimulus + 5400),:].reset_index(drop=True) 
           dataset_sec = dataset_stim.groupby(np.arange(len(dataset_stim))//2).mean()
  
    
        position = dataset_sec.iloc[300, 0]
        dataset_sec[file_names] = ((dataset_sec.iloc[:,0] - int(position)) /6) + 85
        x_value = dataset_sec.iloc[:,-1]
       

        dataset_sec['pellet'+ file_names] = -1000
        dataset_sec['after_pell' + file_names] = "NaN"
        if sum(dataset_sec.iloc[:,1]) > 0:
            pellet = dataset_sec.loc[dataset_sec.iloc[:,1] == 1].index.values.astype(int)[0]
            dataset_sec.iloc[pellet, -2] = ((dataset_sec.iloc[pellet,0] - int(position)) /6) + 85
            dataset_sec.iloc[pellet:, -1] = ((dataset_sec.iloc[pellet:,0] - int(position)) /6) + 85
        

        
        
        pellet_frame = dataset_sec.iloc[:,-2]
        after_pellet = dataset_sec['after_pell' + file_names]
        trace = pd.concat([x_value, pellet_frame, after_pellet], axis = 1)
        
        
        return(trace)

#%%
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'
columns = ['xpos']
index = range(10)
shelter_loom = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = ShelterTime(file_names)  
    shelter_loom = pd.concat([shelter_loom, animal], axis=1)
shelter_loom = shelter_loom.drop(columns = ['xpos'])
shelter_loom = shelter_loom.drop(index = 0)

shelter_loom['frames'] = range(len(shelter_loom))
shelter_loom['sec'] = (shelter_loom['frames']-300)/30


df = shelter_loom.apply(pd.to_numeric, errors='coerce')
#%%
'''Loom'''
ax = sns.lineplot(x='sec', y='MH033', data =df, color = 'black', zorder=1).set_title('Loom MH33')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH033', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH033', data =df, color = 'pink', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH034', data =df, color = 'black', zorder=1).set_title('Loom MH34')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH034', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH034', data =df, color = 'pink', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH046', data =df, color = 'black', zorder=1).set_title('Loom MH46')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH046', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH046', data =df, color = 'pink', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH059', data =df, color = 'black', zorder=1).set_title('Loom MH59')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH059', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH059', data =df, color = 'pink', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH065', data =df, color = 'black', zorder=1).set_title('Loom MH65')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH065', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH065', data =df, color = 'pink', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH066', data =df, color = 'black', zorder=1).set_title('Loom MH66')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH066', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH066', data =df, color = 'pink', zorder=2)#Loom


'''Loom'''
ax = sns.lineplot(x='sec', y='MH085', data =df, color = 'black', zorder = 1).set_title('Loom MH85')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH085', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH085', data =df, color = 'pink', zorder=2)#Loom
#
#
ax = sns.lineplot(x='sec', y='MH086', data =df, color = 'black', zorder = 1).set_title('Loom MH86')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH086', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH086', data =df, color = 'pink', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH087', data =df, color = 'black', zorder = 1).set_title('Loom MH87')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH087', data = df, color = 'pink',s =150,zorder = 2)
ax = sns.lineplot(x='sec', y='after_pellMH087', data =df, color = 'pink', zorder=1)#Loom


ax = sns.lineplot(x='sec', y='MH088', data =df, color = 'black', zorder = 1).set_title('Loom MH88')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH088', data = df, color = 'pink',s =150,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH088', data =df, color = 'pink', zorder=2)#Loom
ax.axvspan(0, 1, alpha=0.2, color='k')
ax.axvspan(2, 3, alpha=0.2, color='k')
ax.axvspan(4, 5, alpha=0.2, color='k')
ax.axvspan(6, 7, alpha=0.2, color='k')
ax.axvspan(8, 9, alpha=0.2, color='k')
ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.03)) 


ax2 = ax.twinx() # create a second y axis

ax.axhspan(20, -100, alpha=0.8, color='maroon')
ax.set_ylim([-20, 200])
ax2.set_ylim([-0, 200])
ax.set_xlim([-10, 30])
ax.set_title('Traces aligned to loom trigger\n', fontsize = 20)
ax2.set_ylabel('Position in arena (cm)', fontsize = 15)
ax.set_ylabel('', fontsize = 15)
ax.tick_params(labelsize=15)
ax2.tick_params(labelsize=15)
ax.set_xlabel('Seconds', fontsize = 15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
'''
Fast retrievers:
MH088
MH086
MH085
MH066
'''
#%%
sns.set_style("ticks")
#%%
'''Color coded'''
ax = sns.lineplot(x='sec', y='MH033', data =df, color = 'black', zorder=1)#.set_title('Loom MH33')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH033', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH033', data =df, color = 'lightgray', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH034', data =df, color = 'black', zorder=1)#.set_title('Loom MH34')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH034', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH034', data =df, color = 'lightgray', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH046', data =df, color = 'teal', zorder=1)#.set_title('Loom MH46')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH046', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH046', data =df, color = 'lightgray', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH059', data =df, color = 'darkslateblue', zorder=1)#.set_title('Loom MH59')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH059', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH059', data =df, color = 'lightgray', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH065', data =df, color = 'black', zorder=1)#.set_title('Loom MH65')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH065', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH065', data =df, color = 'lightgray', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH066', data =df, color = 'yellowgreen', zorder=1)#.set_title('Loom MH66')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH066', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH066', data =df, color = 'lightgray', zorder=2)#Loom


'''Loom'''
ax = sns.lineplot(x='sec', y='MH085', data =df, color = 'lightgreen', zorder = 1)#.set_title('Loom MH85')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH085', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH085', data =df, color = 'lightgray', zorder=2)#Loom
#
#
ax = sns.lineplot(x='sec', y='MH086', data =df, color = 'mediumaquamarine', zorder = 1)#.set_title('Loom MH86')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH086', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH086', data =df, color = 'lightgray', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH087', data =df, color = 'black', zorder = 1)#.set_title('Loom MH87')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH087', data = df, color = 'orange',s =150,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH087', data =df, color = 'lightgray', zorder=1)#Loom


ax = sns.lineplot(x='sec', y='MH088', data =df, color = 'cadetblue', zorder = 1)#.set_title('Loom MH88')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH088', data = df, color = 'orange',s =150,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH088', data =df, color = 'lightgray', zorder=2)#Loom

ax.axvline(0, alpha=0.9, ls='--', color='k')
ax.axvline(2, alpha=0.9, ls='--', color='k')
ax.axvline(4, alpha=0.9, ls='--', color='k')
ax.axvline(6, alpha=0.9, ls='--', color='k')
ax.axvline(8, alpha=0.9, ls='--', color='k')


ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.03)) 

#ax2 = ax.twinx() # create a second y axis

ax.axhspan(2, -100, alpha=0.4, color='maroon')
ax.set_ylim([-20, 200])
#ax2.set_ylim([-0, 200])
ax.set_xlim([-10, 30])
#ax.set_title('Traces aligned to loom trigger\n', fontsize = 20)
ax.set_ylabel('Position in arena (cm)', fontsize = 20)
#ax.set_ylabel('', fontsize = 20)
ax.tick_params(labelsize=20)
#ax2.tick_params(labelsize=15)
ax.set_xlabel('Seconds', fontsize = 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax2.spines['right'].set_visible(False)
#ax2.spines['top'].set_visible(False)

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 2
ax.yaxis.set_tick_params(size=5, width=2)
#ax2.xaxis.set_tick_params(size=5, width=2, color='k')
ax.xaxis.set_tick_params(size=5, width=2, color='k')

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

plt.savefig('loom_test_30sec.jpeg',bbox_inches='tight', dpi=1800)
#%%

'''
Fast retrievers:
MH088
MH086
MH085
MH066
'''
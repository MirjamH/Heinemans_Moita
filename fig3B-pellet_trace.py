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
    1.MH033,
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
import seaborn as sns
import matplotlib.pyplot as plt
import json
sns.set_style("whitegrid", {'axes.grid' : False})

os.chdir('/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/analysis files') # to change directory Read csv files with Pandas
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
           if stimulus <900:
               ind = 900 - stimulus
               baseline = pd.DataFrame(0, index = range(ind), columns = dataset.columns)
               dataset = pd.concat([baseline, dataset], axis = 0).reset_index(drop=True)
               stimulus = dataset.loc[dataset.iloc[:,2] == 1].index.values.astype(int)[0]
           dataset_stim = dataset.iloc[int(stimulus - 900):int(stimulus + 8100),:].reset_index(drop=True)
           dataset_sec = dataset_stim.groupby(np.arange(len(dataset_stim))//3).mean()
            
        else: # 60 FPS videos if stimulus <900:
            if stimulus <900:
                ind = 600 - stimulus
                baseline = pd.DataFrame(0, index = range(ind), columns = dataset.columns)
                dataset = pd.concat([baseline, dataset], axis = 0).reset_index(drop=True)
                stimulus = dataset.loc[dataset.iloc[:,2] == 1].index.values.astype(int)[0]
            dataset_stim = dataset.iloc[int(stimulus - 600):int(stimulus + 5400),:].reset_index(drop=True) 
            dataset_sec = dataset_stim.groupby(np.arange(len(dataset_stim))//2).mean()
  
    
        position = dataset_sec.iloc[300, 0]
        dataset_sec[file_names] = ((dataset_sec.iloc[:,0] - int(position)) /6) + 85
        x_value = dataset_sec.iloc[:,-1]
       

        dataset_sec['pellet'+ file_names] = -1000
        dataset_sec['after_pell' + file_names] = "NaN"
        if sum(dataset_sec.iloc[:,1]) > 0:
            pellet = dataset_sec.loc[dataset_sec.iloc[:,1] > 0].index.values.astype(int)[0]
            dataset_sec.iloc[pellet, -2] = ((dataset_sec.iloc[pellet,0] - int(position)) /6) + 85
            dataset_sec.iloc[pellet:, -1] = ((dataset_sec.iloc[pellet:,0] - int(position)) /6) + 85
        

        
        
        pellet_frame = dataset_sec.iloc[:,-2]
        after_pellet = dataset_sec['after_pell' + file_names]
        trace = pd.concat([x_value, pellet_frame, after_pellet], axis = 1)
        
        return(trace)
        
        

#%%
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/_Loom'
columns = ['xpos']
index = range(300)
shelter_loom = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = ShelterTime(file_names)  
    shelter_loom = pd.concat([shelter_loom, animal], axis=1)
shelter_loom = shelter_loom.drop(columns = ['xpos'])
shelter_loom = shelter_loom.drop(index = 0)


shelter_loom = shelter_loom.apply(pd.to_numeric, errors='coerce')


#%%
path_name = '/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/_T_Loom'
columns = ['xpos']
index = range(300)
shelter_t_loom = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = ShelterTime(file_names)  
    shelter_t_loom = pd.concat([shelter_t_loom, animal], axis=1)
shelter_t_loom = shelter_t_loom.drop(columns = ['xpos'])
shelter_t_loom = shelter_t_loom.drop(index = 0)
shelter_t_loom = shelter_t_loom.apply(pd.to_numeric, errors='coerce')


#%%
path_name = '/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/_T_Shock'
columns = ['xpos']
index = range(300)
shelter_t_shock = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = ShelterTime(file_names)  
    shelter_t_shock = pd.concat([shelter_t_shock, animal], axis=1)
shelter_t_shock = shelter_t_shock.drop(columns = ['xpos'])
shelter_t_shock = shelter_t_shock.drop(index = 0)
shelter_t_shock = shelter_t_shock.apply(pd.to_numeric, errors='coerce')

#%%
path_name = '/Users/mirjamheinemans/Dropbox/My Mac (Mirjam’s MacBook Pro)/Desktop/Annotator python tryout/_Tone'
columns = ['xpos']
index = range(300)
shelter_tone = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = ShelterTime(file_names)  
    shelter_tone = pd.concat([shelter_tone, animal], axis=1)
shelter_tone = shelter_tone.drop(columns = ['xpos'])
shelter_tone = shelter_tone.drop(index = 0)
shelter_tone = shelter_tone.apply(pd.to_numeric, errors='coerce')
        

#%%
df = pd.concat([shelter_loom, shelter_t_shock, shelter_t_loom, shelter_tone], axis = 1)
df['frames']=  range(len(df))
df['sec'] =(df['frames']/30) - 10
#%%
df.to_csv("traces_15_sec.csv")
#%%
'''Loom'''
ax = sns.lineplot(x='sec', y='MH033', data =df, color = 'black', zorder=1).set_title('Loom MH33')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH033', data = df, color = 'c',s =150,zorder = 2)

ax = sns.lineplot(x='sec', y='MH034', data =df, color = 'black', zorder=1).set_title('Loom MH34')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH034', data = df, color = 'c',s =150,zorder = 2)


ax = sns.lineplot(x='sec', y='MH046', data =df, color = 'black', zorder=1).set_title('Loom MH46')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH046', data = df, color = 'c',s =150,zorder = 2)


ax = sns.lineplot(x='sec', y='MH059', data =df, color = 'black', zorder=1).set_title('Loom MH59')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH059', data = df, color = 'c',s =150,zorder = 2)


ax = sns.lineplot(x='sec', y='MH065', data =df, color = 'black', zorder=1).set_title('Loom MH65')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH065', data = df, color = 'c',s =150,zorder = 2)
ax = sns.lineplot(x='sec', y='after_pellMH065', data =df, color = 'lightgray', zorder=1)#Loom


ax = sns.lineplot(x='sec', y='MH066', data =df, color = 'black', zorder=1).set_title('Loom MH66')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH066', data = df, color = 'c',s =150,zorder = 2)
ax = sns.lineplot(x='sec', y='after_pellMH066', data =df, color = 'lightgray', zorder=1)#Loom

'''Loom'''
ax = sns.lineplot(x='sec', y='MH085', data =df, color = 'black', zorder = 1).set_title('Loom MH85')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH085', data = df, color = 'g',s =150,zorder = 2)
ax = sns.lineplot(x='sec', y='after_pellMH085', data =df, color = 'lightgray', zorder=1)#Loom


ax = sns.lineplot(x='sec', y='MH086', data =df, color = 'black', zorder = 1).set_title('Loom MH86')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH086', data = df, color = 'g',s =150,zorder = 2)
ax = sns.lineplot(x='sec', y='after_pellMH086', data =df, color = 'lightgray', zorder=1)#Loom


ax = sns.lineplot(x='sec', y='MH087', data =df, color = 'black', zorder = 1).set_title('Loom MH87')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH087', data = df, color = 'g',s =150,zorder = 2)
#
#
ax = sns.lineplot(x='sec', y='MH088', data =df, color = 'black', zorder = 1).set_title('Loom MH88')#Loom
ax = sns.scatterplot(x = 'sec', y = 'pelletMH088', data = df, color = 'g',s =150,zorder = 2)
ax = sns.lineplot(x='sec', y='after_pellMH088', data =df, color = 'lightgray', zorder=1)#Loom

ax.axvline(0, alpha=0.9, ls='--', color='k')
ax.axvline(2, alpha=0.9, ls='--', color='k')
ax.axvline(4, alpha=0.9, ls='--', color='k')
ax.axvline(6, alpha=0.9, ls='--', color='k')
ax.axvline(8, alpha=0.9, ls='--', color='k')

ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.03)) 


ax.axhspan(-500, -700, alpha=0.3, color='green')
ax.set_title('Test\nAligned with trigger position test')
ax.set_ylabel('x-position (a.u.)')
ax.set_ylim([-20, 200])
ax.set_xlim([-10, 30])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#%%
'''Tone-Loom'''
#
#ax = sns.lineplot(x='sec', y='MH043', data =df, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH043', data = df, color = 'pink',s =100,zorder = 2).set_title('Tone-Loom MH43')
#ax = sns.lineplot(x='sec', y='after_pellMH043', data =df, color = 'pink', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH044', data =df, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH044', data = df, color = 'pink',s =100,zorder = 2).set_title('Tone-Loom MH44')
#ax = sns.lineplot(x='sec', y='after_pellMH044', data =df, color = 'pink', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH051', data =df, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH051', data = df, color = 'pink',s =100,zorder = 2).set_title('Tone-Loom MH51')
#ax = sns.lineplot(x='sec', y='after_pellMH051', data =df, color = 'pink', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH052', data =df, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH052', data = df, color = 'pink',s =100,zorder = 2).set_title('Tone-Loom MH52')
#ax = sns.lineplot(x='sec', y='after_pellMH052', data =df, color = 'pink', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH063', data =df, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH063', data = df, color = 'pink',s =100,zorder = 2).set_title('Tone-Loom MH63')
#ax = sns.lineplot(x='sec', y='after_pellMH063', data =df, color = 'pink', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH064', data =df, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH064', data = df, color = 'pink',s =100,zorder = 2).set_title('Tone-Loom MH64')
#ax = sns.lineplot(x='sec', y='after_pellMH064', data =df, color = 'pink', zorder=1)#Loom

#
#ax = sns.lineplot(x='sec', y='MH070', data =df, color = 'blue',zorder = 1)#T-Loom
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH070', data = df, color = 'pink',s =100,zorder = 2).set_title('Tone-Loom MH70')
#ax = sns.lineplot(x='sec', y='after_pellMH070', data =df, color = 'pink', zorder=1)#Loom
#
#ax.axvspan(0, 10, alpha=0.1, color='orange')


#ax.axvspan(0, 1, alpha=0.2, color='k')
#ax.axvspan(2, 3, alpha=0.2, color='k')
#ax.axvspan(4, 5, alpha=0.2, color='k')
#ax.axvspan(6, 7, alpha=0.2, color='k')
#ax.axvspan(8, 9, alpha=0.2, color='k')



#ax2 = ax.twinx() # create a second y axis

#ax.axhspan(20, -100, alpha=0.8, color='maroon')
#ax.set_ylim([-20, 210])
##ax2.set_ylim([-0, 200])
#ax.set_xlim([-10, 30])
#ax.set_title('Tone-Loom - aligned to tone onset\n', fontsize = 20)
#ax.set_ylabel('Position in arena (cm)', fontsize = 15)
#
#ax.tick_params(labelsize=15)
##ax2.tick_params(labelsize=15)
#ax.set_xlabel('Seconds', fontsize = 15)
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
#ax2.spines['right'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#%%
'''Tone-Loom'''

fig = plt.figure(figsize=(5,4))
with sns.axes_style("ticks"):
    ax = sns.lineplot(x='sec', y='MH071', data =df, color = 'royalblue',zorder = 1)#.set_title('Tone-Loom MH71')#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH071', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH071', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH072', data =df, color = 'royalblue',zorder = 1)#.set_title('Tone-Loom MH71')#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH072', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH072', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH077', data =df, color = 'royalblue',zorder = 1)#.set_title('Tone-Loom MH71')#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH077', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH077', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH078', data =df, color = 'royalblue',zorder = 1)#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH078', data = df, color = 'black',s =200,zorder = 2)#.set_title('Tone-Loom MH78')
    ax = sns.lineplot(x='sec', y='after_pellMH078', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH092', data =df, color = 'royalblue',zorder = 1)#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH092', data = df, color = 'black',s =200,zorder = 2)#.set_title('Tone-Loom MH92')
    ax = sns.lineplot(x='sec', y='after_pellMH092', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH096', data =df, color = 'royalblue',zorder = 1)#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH096', data = df, color = 'black',s =200,zorder = 2)#.set_title('Tone-Loom MH96')
    ax = sns.lineplot(x='sec', y='after_pellMH096', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH101',data =df, color = 'royalblue',zorder = 1)#.set_title('Tone-Loom MH101')
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH101', data = df, color = 'black',s =200,zorder = 2)#.set_ylabel('x-position (a.u.)')#T-Loom
    ax = sns.lineplot(x='sec', y='after_pellMH101', data =df, color = 'grey', zorder=1)#Loom
    
    ax.axvline(0, alpha=0.9, ls='--', color='k')
    ax.axvline(2, alpha=0.9, ls='--', color='k')
    ax.axvline(4, alpha=0.9, ls='--', color='k')
    ax.axvline(6, alpha=0.9, ls='--', color='k')
    ax.axvline(8, alpha=0.9, ls='--', color='k')
    
    ax.spines['left'].set_position(('axes', - 0.02))
    ax.spines['bottom'].set_position(('axes', -0.03)) 
    
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.yaxis.set_tick_params(size=5, width=2)
    ax.xaxis.set_tick_params(size=5, width=2, color='k')
    
    ax.axhspan(2, -100, alpha=0.4, color='maroon')
    ax.set_ylim([-20, 200])
    #ax2.set_ylim([-0, 200])
    ax.set_xlim([-10, 15])
    ax.xaxis.set_ticks(np.arange(-10, 16, 5))
    ax.set_ylabel('', fontsize = 20)
    
    ax.tick_params(labelsize=20)
    #ax2.tick_params(labelsize=15)
    ax.set_xlabel('', fontsize = 20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax2.spines['right'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    
os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

#plt.savefig('TL_test_15sec.jpeg',bbox_inches='tight', dpi=1800)
#%%
'''Tone-Shock'''
#ax = sns.lineplot(x='sec', y='MH039', data =df, color = 'red',zorder = 1).set_title('Tone-Shock MH39')#T-Shock
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH039', data = df, color = 'black',s =200,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH039', data =df, color = 'lightgray', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH040', data =df, color = 'red',zorder = 1).set_title('Tone-Shock MH40')#T-Shock
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH040', data = df, color = 'black',s =200,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH040', data =df, color = 'lightgray', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH049', data =df, color = 'red',zorder = 1).set_title('Tone-Shock MH49')#T-Shock
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH049', data = df, color = 'black',s =200,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH049', data =df, color = 'lightgray', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH050', data =df, color = 'red',zorder = 1).set_title('Tone-Shock MH50')#T-Shock
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH050', data = df, color = 'black',s =200,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH050', data =df, color = 'lightgray', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH055', data =df, color = 'red',zorder = 1).set_title('Tone-Shock MH55')#T-Shock
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH055', data = df, color = 'black',s =200,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH055', data =df, color = 'lightgray', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH056', data =df, color = 'red',zorder = 1).set_title('Tone-Shock MH56')#T-Shock
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH056', data = df, color = 'black',s =200,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH056', data =df, color = 'lightgray', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH067', data =df, color = 'red',zorder = 1).set_title('Tone-Shock MH67')#T-Shock
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH067', data = df, color = 'black',s =100,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH067', data =df, color = 'lightgray', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH068', data =df, color = 'red',zorder = 1).set_title('Tone-Shock MH68')#T-Shock
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH068', data = df, color = 'black',s =200,zorder = 2)
#ax = sns.lineplot(x='sec', y='after_pellMH068', data =df, color = 'lightgray', zorder=1)#Loom
#
#ax.axvspan(0, 10, alpha=0.1, color='orange')


#ax.axhspan(-500, -700, alpha=0.3, color='green')
#ax.set_title('Tone-Shock\n Traces after tone onset', fontsize = 20)
#ax.set_ylabel('Position in arena (a.u.)', fontsize = 15)
#ax.set_ylim([-700, 700])
#ax.set_xlim([-10, 11])
#ax.set_xlabel('Seconds', fontsize = 15)
#ax.tick_params(labelsize=15)
#
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
#%%

'''Tone-Shock'''

fig = plt.figure(figsize=(5,4))
with sns.axes_style("ticks"):
    ax = sns.lineplot(x='sec', y='MH073', data =df, color = 'crimson',zorder = 1)#.set_title('Tone-Shock MH73')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH073', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH073', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH074', data =df, color = 'crimson',zorder = 1)#.set_title('Tone-Shock MH74')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH074', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH074', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH079', data =df, color = 'crimson',zorder = 1)#.set_title('Tone-Shock MH79')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH079', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH079', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH080', data =df, color = 'crimson',zorder = 1)#.set_title('Tone-Shock MH80')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH080', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH080', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH084', data =df, color = 'crimson',zorder = 1)#.set_title('Tone-Shock MH84')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH084', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH084', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH089', data =df, color = 'crimson',zorder = 1)#.set_title('Tone-Shock MH89')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH089', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH089', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH090', data =df, color = 'crimson',zorder = 1)#.set_title('Tone-Shock MH90')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH090', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH090', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH100',data =df, color = 'crimson',zorder = 1)#.set_title('Tone-Shock MH100')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH100', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH100', data =df, color = 'grey', zorder=1)#Loom
    
    ax.axvline(0, alpha=0.9, ls='--', color='k')
    ax.axvline(2, alpha=0.9, ls='--', color='k')
    ax.axvline(4, alpha=0.9, ls='--', color='k')
    ax.axvline(6, alpha=0.9, ls='--', color='k')
    ax.axvline(8, alpha=0.9, ls='--', color='k')
    
    ax.spines['left'].set_position(('axes', - 0.02))
    ax.spines['bottom'].set_position(('axes', -0.03)) 

    ax.yaxis.set_tick_params(size=5, width=2)
    ax.xaxis.set_tick_params(size=5, width=2, color='k')
    
    ax.axhspan(2, -100, alpha=0.4, color='maroon')
    ax.set_ylim([-20, 200])
    #ax2.set_ylim([-0, 200])
    ax.set_xlim([-10, 15])
    ax.xaxis.set_ticks(np.arange(-10, 16, 5))
    ax.set_ylabel('', fontsize = 20)
    
    ax.tick_params(labelsize=20)
    #ax2.tick_params(labelsize=15)
    ax.set_xlabel('', fontsize = 20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

#plt.savefig('TS_test_15sec.jpeg',bbox_inches='tight', dpi=1800)
#%%
'''Tone'''
#ax = sns.lineplot(x='sec', y='MH037', data =df, color = 'green').set_title('Tone MH37')#Tone
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH037', data = df, color = 'black',s =100,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH037', data =df, color = 'lightgray', zorder=2)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH038', data =df, color = 'green').set_title('Tone MH38')#Tone
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH038', data = df, color = 'black',s =100,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH038', data =df, color = 'lightgray', zorder=2)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH042', data =df, color = 'green').set_title('Tone MH42')#Tone
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH042', data = df, color = 'black',s =100,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH042', data =df, color = 'lightgray', zorder=2)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH048', data =df, color = 'green').set_title('Tone MH48')#Tone
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH048', data = df, color = 'black',s =100,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH048', data =df, color = 'lightgray', zorder=1)#Loom
#
#
#ax = sns.lineplot(x='sec', y='MH058', data =df, color = 'green').set_title('Tone MH58')#Tone
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH058', data = df, color = 'black',s =100,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH058', data =df, color = 'lightgray', zorder=2)#Loom
#
#
#
#ax = sns.lineplot(x='sec', y='MH062', data =df, color = 'green').set_title('Tone MH62')#Tone
#ax = sns.scatterplot(x = 'sec', y = 'pelletMH062', data = df, color = 'black',s =100,zorder = 3)
#ax = sns.lineplot(x='sec', y='after_pellMH062', data =df, color = 'lightgray', zorder=2)#Loom
#
#ax.axvspan(0, 10, alpha=0.1, color='orange')
#
#
#ax.axhspan(-500, -700, alpha=0.3, color='green')
#ax.set_title('Tone\n Traces after tone onset', fontsize = 20)
#ax.set_ylabel('Position in arena (a.u.)', fontsize = 15)
#ax.set_ylim([-700, 700])
#ax.set_xlim([-10, 11])
#ax.set_xlabel('Seconds', fontsize = 15)
#ax.tick_params(labelsize=15)
#
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
#%%
'''Tone'''

fig = plt.figure(figsize=(5,4))
sns.set_style('ticks')
ax = sns.lineplot(x='sec', y='MH075', data =df, color = 'green')#.set_title('Tone MH75')#Tone
ax = sns.scatterplot(x = 'sec', y = 'pelletMH075', data = df, color = 'black',s =200,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH075', data =df, color = 'grey', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH076', data =df, color = 'green')#.set_title('Tone MH76')#Tone
ax = sns.scatterplot(x = 'sec', y = 'pelletMH076', data = df, color = 'black',s =200,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH076', data =df, color = 'grey', zorder=2)#Loom

ax = sns.lineplot(x='sec', y='MH081', data =df, color = 'green')#.set_title('Tone MH81')#Tone
ax = sns.scatterplot(x = 'sec', y = 'pelletMH081', data = df, color = 'black',s =200,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH081', data =df, color = 'grey', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH082', data =df, color = 'green')#.set_title('Tone MH82')#Tone
ax = sns.scatterplot(x = 'sec', y = 'pelletMH082', data = df, color = 'black',s =200,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH082', data =df, color = 'grey', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH093', data =df, color = 'green')#.set_title('Tone MH93')#Tone
ax = sns.scatterplot(x = 'sec', y = 'pelletMH093', data = df, color = 'black',s =200,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH093', data =df, color = 'grey', zorder=2)#Loom


ax = sns.lineplot(x='sec', y='MH098', data =df, color = 'green')#.set_title('Tone MH98')#Tone
ax = sns.scatterplot(x = 'sec', y = 'pelletMH098', data = df, color = 'black',s =200,zorder = 3)
ax = sns.lineplot(x='sec', y='after_pellMH098', data =df, color = 'grey', zorder=2)#Loom

ax.axvline(0, alpha=0.9, ls='--', color='k')
ax.axvline(2, alpha=0.9, ls='--', color='k')
ax.axvline(4, alpha=0.9, ls='--', color='k')
ax.axvline(6, alpha=0.9, ls='--', color='k')
ax.axvline(8, alpha=0.9, ls='--', color='k')

ax.spines['left'].set_position(('axes', - 0.02))
ax.spines['bottom'].set_position(('axes', -0.03)) 

ax.yaxis.set_tick_params(size=5, width=2)
ax.xaxis.set_tick_params(size=5, width=2, color='k')
ax.axhspan(2, -100, alpha=0.4, color='maroon')
ax.set_ylim([-20, 200])
#ax2.set_ylim([-0, 200])
ax.set_xlim([-10, 15])
ax.xaxis.set_ticks(np.arange(-10, 16, 5))
ax.set_ylabel('', fontsize = 20)

ax.tick_params(labelsize=20)
#ax2.tick_params(labelsize=15)
ax.set_xlabel('', fontsize = 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
    
os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

#plt.savefig('T_test_15sec.jpeg',bbox_inches='tight', dpi=1800)

#%%

''' Tone-Loom colorcoded for freezing: darker is high freezing'''

fig = plt.figure(figsize=(5,4))
with sns.axes_style("ticks"):
    ax = sns.lineplot(x='sec', y='MH071', data =df, color = 'teal',zorder = 1)#.set_title('Tone-Loom MH71')#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH071', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH071', data =df, color = 'teal', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH072', data =df, color = 'lightgreen',zorder = 1)#.set_title('Tone-Loom MH71')#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH072', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH072', data =df, color = 'lightgreen', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH077', data =df, color = 'cadetblue',zorder = 1)#.set_title('Tone-Loom MH71')#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH077', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH077', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH078', data =df, color = 'indigo',zorder = 1)#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH078', data = df, color = 'black',s =200,zorder = 2)#.set_title('Tone-Loom MH78')
    ax = sns.lineplot(x='sec', y='after_pellMH078', data =df, color = 'indigo', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH092', data =df, color = 'mediumaquamarine',zorder = 1)#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH092', data = df, color = 'black',s =200,zorder = 2)#.set_title('Tone-Loom MH92')
    ax = sns.lineplot(x='sec', y='after_pellMH092', data =df, color = 'mediumaquamarine', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH096', data =df, color = 'yellowgreen',zorder = 1)#T-Loom
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH096', data = df, color = 'black',s =200,zorder = 2)#.set_title('Tone-Loom MH96')
    ax = sns.lineplot(x='sec', y='after_pellMH096', data =df, color = 'yellowgreen', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH101',data =df, color = 'darkslateblue',zorder = 1)#.set_title('Tone-Loom MH101')
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH101', data = df, color = 'black',s =200,zorder = 2)#.set_ylabel('x-position (a.u.)')#T-Loom
    ax = sns.lineplot(x='sec', y='after_pellMH101', data =df, color = 'darkslateblue', zorder=1)#Loom
    
    
    ax.axvline(0, alpha=0.9, ls='--', color='k')
    ax.axvline(2, alpha=0.9, ls='--', color='k')
    ax.axvline(4, alpha=0.9, ls='--', color='k')
    ax.axvline(6, alpha=0.9, ls='--', color='k')
    ax.axvline(8, alpha=0.9, ls='--', color='k')
    
    ax.yaxis.set_tick_params(size=5, width=2)
    ax.xaxis.set_tick_params(size=5, width=2, color='k')
    
    ax.axhspan(2, -100, alpha=0.4, color='maroon')
    ax.set_ylim([-20, 210])
    #ax2.set_ylim([-0, 200])
    ax.set_xlim([-10, 15])
    #ax.set_title('Tone-Loom - aligned to tone onset (n = 7)\n', fontsize = 20)
    ax.set_ylabel('', fontsize = 20)
    
    ax.tick_params(labelsize=20)
    #ax2.tick_params(labelsize=15)
    ax.set_xlabel('', fontsize = 20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax2.spines['right'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    
os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

#plt.savefig('T_test_15sec.jpeg',bbox_inches='tight', dpi=1800)
    
#%%

''' Tone-Shock colorcoded for freezing: darker is high freezing'''

fig = plt.figure(figsize=(5,4))
with sns.axes_style("ticks"):
    ax = sns.lineplot(x='sec', y='MH073', data =df, color = 'lightgreen',zorder = 1).set_title('Tone-Shock MH73')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH073', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH073', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH074', data =df, color = 'mediumaquamarine',zorder = 1).set_title('Tone-Shock MH74')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH074', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH074', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH079', data =df, color = 'yellowgreen',zorder = 1).set_title('Tone-Shock MH79')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH079', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH079', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH080', data =df, color = 'indigo',zorder = 1).set_title('Tone-Shock MH80')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH080', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH080', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH084', data =df, color = 'darkslateblue',zorder = 1).set_title('Tone-Shock MH84')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH084', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH084', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH089', data =df, color = 'teal',zorder = 1).set_title('Tone-Shock MH89')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH089', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH089', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH090', data =df, color = 'cadetblue',zorder = 1).set_title('Tone-Shock MH90')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH090', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH090', data =df, color = 'grey', zorder=1)#Loom
    
    
    ax = sns.lineplot(x='sec', y='MH100',data =df, color = 'yellow',zorder = 1).set_title('Tone-Shock MH100')#T-Shock
    ax = sns.scatterplot(x = 'sec', y = 'pelletMH100', data = df, color = 'black',s =200,zorder = 2)
    ax = sns.lineplot(x='sec', y='after_pellMH100', data =df, color = 'grey', zorder=1)#Loom
    
    ax.axvline(0, alpha=0.9, ls='--', color='k')
    ax.axvline(2, alpha=0.9, ls='--', color='k')
    ax.axvline(4, alpha=0.9, ls='--', color='k')
    ax.axvline(6, alpha=0.9, ls='--', color='k')
    ax.axvline(8, alpha=0.9, ls='--', color='k')
    
    
    
    #ax2 = ax.twinx() # create a second y axis
    
    ax.axhspan(20, -100, alpha=0.4, color='maroon')
    ax.set_ylim([-20, 210])
    #ax2.set_ylim([-0, 200])
    ax.set_xlim([-10, 15])
    ax.set_title('Tone-Shock - aligned to tone onset (n = 8)\n', fontsize = 20)
    ax.set_ylabel('Position in arena (cm)', fontsize = 15)
    
    ax.tick_params(labelsize=15)
    #ax2.tick_params(labelsize=15)
    ax.set_xlabel('Seconds', fontsize = 15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('axes', - 0.02))
    ax.spines['bottom'].set_position(('axes', -0.03)) 


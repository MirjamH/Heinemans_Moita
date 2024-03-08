#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:09:01 2020

@author: mirjamheinemans

Conditioninng Columns:
0. ' '
1. baseline
2. stimulus
3. freezing

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

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout') # to change directory Read csv files with Pandas
#%%

def CondFreezTL(file_names):

    if file_names == '.DS_Store':
        next
    else:
        
        dataset = pd.read_csv(path_name + '/' + file_names +'/conditioning.csv', usecols = [1,2,3]) 
        base = dataset.iloc[:,].diff()[dataset.iloc[:,0].diff() != 0].index.values
        baseline = base[1]
        freeze_base = (sum(dataset.iloc[:baseline,2]))/baseline*100
        
        number =file_names.replace('MH',"")
        
        if int(number) <71: #TL animals <71 have 30 FPS videos
            three_min = baseline + 48600 #48600 frames = 30 FPS * 60 sec * 27 min
            freeze_stim = (sum(dataset.iloc[baseline:three_min,2]))/(48600)*100
            
        else: # TL animals >71 have 60 FPS videos
            three_min = baseline + 97200 #97200 frames = 60 FPS * 60 sec * 27 min
            freeze_stim = (sum(dataset.iloc[baseline:three_min,2]))/(97200)*100
        
        freezing_percent = pd.DataFrame(data =[freeze_base,freeze_stim], index =['base','stim'],columns = [file_names])
       
        return(freezing_percent)

#%%
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_T_Loom'
columns = ['xpos']
index = range(1)
freeze_loom = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = CondFreezTL(file_names)
    freeze_loom = pd.concat([freeze_loom, animal], axis=1, sort = True)

freeze_loom = freeze_loom.drop(columns = ['xpos'])
freeze_loom = freeze_loom.drop(index = 0)
loom = freeze_loom.T
loom['difference'] = loom['stim'] - loom['base']

#%%

def CondFreezShock(file_names):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name + '/' + file_names +'/conditioning.csv', usecols = [1,2,3])
        base = dataset.iloc[:,].diff()[dataset.iloc[:,0].diff() != 0].index.values
        baseline = base[1]
        freeze_base = (sum(dataset.iloc[:baseline,2]))/baseline*100
        

        three_min = baseline + 40500 #97200 frames = 25 FPS * 60 sec * 27 min
        freeze_stim = (sum(dataset.iloc[baseline:three_min,2]))/(40500)*100
        
        freezing_percent = pd.DataFrame(data =[freeze_base,freeze_stim], index =['base','stim'],columns = [file_names])
       
        return(freezing_percent)    
 #%%
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_T_Shock'
columns = ['xpos']
index = range(1)
freeze_shock = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = CondFreezShock(file_names)
    freeze_shock = pd.concat([freeze_shock, animal], axis=1, sort = True)

freeze_shock = freeze_shock.drop(columns = ['xpos'])
freeze_shock = freeze_shock.drop(index = 0)     
shock = freeze_shock.T

shock['difference'] = shock['stim'] - shock['base']

#%%

def CondFreezTone(file_names):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name + '/' + file_names +'/conditioning.csv', usecols = [1,2,3])
        base = dataset.iloc[:,].diff()[dataset.iloc[:,0].diff() != 0].index.values
        baseline = base[1]
        freeze_base = (sum(dataset.iloc[:baseline,2]))/baseline*100
        

        three_min = baseline + 40500 #97200 frames = 25 FPS * 60 sec * 27 min
        freeze_stim = (sum(dataset.iloc[baseline:three_min,2]))/(40500)*100
        
        freezing_percent = pd.DataFrame(data =[freeze_base,freeze_stim], index =['base','stim'],columns = [file_names])
       
        return(freezing_percent)    
 #%%
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Tone'
columns = ['xpos']
index = range(1)
freeze_tone = pd.DataFrame(index = index, columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = CondFreezTone(file_names)
    freeze_tone = pd.concat([freeze_tone, animal], axis=1, sort = True)

freeze_tone = freeze_tone.drop(columns = ['xpos'])
freeze_tone = freeze_tone.drop(index = 0)     
tone = freeze_tone.T
tone['difference'] = tone['stim'] - tone['base']

#%%

# your input data:
before_s = shock['base']
after_s = shock['stim']

before_l = loom['base']
after_l = loom['stim']

before_t = tone['base']
after_t = tone['stim']
#%%
fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):
    plt.scatter(np.zeros(len(before_s)), before_s, color = 'orange', s = 100, zorder = 2)
    plt.scatter(np.ones(len(after_s)), after_s, color = 'orange',s =100, zorder = 2)
    
#    
#    plt.scatter(np.ones(len(before_l))*2, before_l, color = 'royalblue', s = 50, zorder = 2)
#    plt.scatter(np.ones(len(after_l))*3, after_l, color = 'royalblue',s =50, zorder = 2)
#    
#    plt.scatter(np.ones(len(before_t))*4, before_t, color = 'green', s = 50, zorder = 2)
#    plt.scatter(np.ones(len(after_t))*5, after_t, color = 'green',s =50, zorder = 2)
    # plotting the lines
    for i in range(len(before_s)):
        plt.plot( [0,1], [before_s[i], after_s[i]], c='red', zorder =1)
    
#    
#    for i in range(len(before_l)):
#        plt.plot( [2,3], [before_l[i], after_l[i]], c='royalblue', zorder =1)
#        
#    for i in range(len(before_t)):
#        plt.plot( [4,5], [before_t[i], after_t[i]], c='green', zorder =1)
#    
        
    plt.xticks([0,1], ['baseline', 'stimulus'])
    plt.ylim(-5, 100)
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    
    plt.tick_params(labelsize=20)
    sns.despine()
    os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas
#    plt.savefig('escape_shock_conditoining.jpeg',bbox_inches='tight', dpi=1800)
    

#%%

fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):
#    plt.scatter(np.zeros(len(before_s)), before_s, color = 'crimson', s = 50, zorder = 2)
#    plt.scatter(np.ones(len(after_s)), after_s, color = 'crimson',s =50, zorder = 2)
    
    
    plt.scatter(np.ones(len(before_l))*2, before_l, color = 'orange', s = 100, zorder = 2)
    plt.scatter(np.ones(len(after_l))*3, after_l, color = 'orange',s =100, zorder = 2)
    
#    plt.scatter(np.ones(len(before_t))*4, before_t, color = 'green', s = 50, zorder = 2)
#    plt.scatter(np.ones(len(after_t))*5, after_t, color = 'green',s =50, zorder = 2)
    # plotting the lines
#    for i in range(len(before_s)):
#        plt.plot( [0,1], [before_s[i], after_s[i]], c='crimson', zorder =1)
    
    
    for i in range(len(before_l)):
        plt.plot( [2,3], [before_l[i], after_l[i]], c='royalblue', zorder =1)
#        
#    for i in range(len(before_t)):
#        plt.plot( [4,5], [before_t[i], after_t[i]], c='green', zorder =1)
#    
        
    plt.xticks([2,3], ['baseline', 'stimulus'])
    plt.ylim(-5, 100)
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    #plt.spines['left'].set_position(('axes', - 0.02))
    plt.tick_params(labelsize=20)
    sns.despine()

    os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas
#    plt.savefig('escape_loom_conditoining.jpeg',bbox_inches='tight', dpi=1800)
    
#%%
fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):
#    plt.scatter(np.zeros(len(before_s)), before_s, color = 'crimson', s = 50, zorder = 2)
#    plt.scatter(np.ones(len(after_s)), after_s, color = 'crimson',s =50, zorder = 2)
    
#    
#    plt.scatter(np.ones(len(before_l))*2, before_l, color = 'orange', s = 100, zorder = 2)
#    plt.scatter(np.ones(len(after_l))*3, after_l, color = 'orange',s =100, zorder = 2)
    
    plt.scatter(np.ones(len(before_t))*4, before_t, color = 'orange', s = 100, zorder = 2)
    plt.scatter(np.ones(len(after_t))*5, after_t, color = 'orange',s =100, zorder = 2)
    # plotting the lines

    for i in range(len(before_t)):
        plt.plot( [4,5], [before_t[i], after_t[i]], c='green', zorder =1)
#    
        
    plt.xticks([4,5], ['baseline', 'stimulus'])
    plt.ylim(-5, 100)
    plt.title('', fontsize =20)
    plt.ylabel('', fontsize = 20)
    #plt.spines['left'].set_position(('axes', - 0.02))
    plt.tick_params(labelsize=20)
    sns.despine()

    os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

#    plt.savefig('escape_tone_conditoining.jpeg',bbox_inches='tight', dpi=1800)

#%%

''' 
The Tone-Loom condition is split into a group with freezing similar to the Tone-Shock condition and 
a group with freezing lower than the Tone-Shock condition. I take the minimum freezer in TS, and then divide the 
TL dataframe into two: tl_low_freeze and tl_high_freeze. 
'''
#shock_min = shock.loc[shock['stim'] ==shock['stim'].min()]
#shock_minimum = shock_min['stim'].astype(int)
#
#tl_low =  loom.loc[loom['stim'] < shock_minimum[0]] # MH43 and MH64 are low, freeze 19 and 17 percent. 
##It would make sense to include them? Maybe say "Does not freeze >5% less than lowest Tone-Shock?)
#tl_high = loom.loc[loom['stim'] > shock_minimum[0]]
#
##%%
#before_s = shock['base']
#after_s = shock['stim']
#
#before_l_low = tl_low['base']
#after_l_low = tl_low['stim']
#
#before_l_high = tl_high['base']
#after_l_high = tl_high['stim']
#
#before_t = tone['base']
#after_t = tone['stim']
## plotting the points
#
#plt.scatter(np.zeros(len(before_s)), before_s, color = 'red', s = 50, zorder = 2)
#plt.scatter(np.ones(len(after_s)), after_s, color = 'red',s =50, zorder = 2)
#
#plt.scatter(np.ones(len(before_l_low))*2, before_l_low, color = 'cyan', s = 50, zorder = 2)
#plt.scatter(np.ones(len(after_l_low))*3, after_l_low, color = 'cyan',s =50, zorder = 2)
#
#plt.scatter(np.ones(len(before_l_high))*2, before_l_high, color = 'blue', s = 50, zorder = 2)
#plt.scatter(np.ones(len(after_l_high))*3, after_l_high, color = 'blue',s =50, zorder = 2)
#
#plt.scatter(np.ones(len(before_t))*4, before_t, color = 'green', s = 50, zorder = 2)
#plt.scatter(np.ones(len(after_t))*5, after_t, color = 'green',s =50, zorder = 2)
## plotting the lines
#for i in range(len(before_s)):
#    plt.plot( [0,1], [before_s[i], after_s[i]], c='red', zorder =1)
#
#for i in range(len(before_l_low)):
#    plt.plot( [2,3], [before_l_low[i], after_l_low[i]], c='lightblue', zorder =1)
#    
#for i in range(len(before_l_high)):
#    plt.plot( [2,3], [before_l_high[i], after_l_high[i]], c='blue', zorder =1)
#    
#for i in range(len(before_t)):
#    plt.plot( [4,5], [before_t[i], after_t[i]], c='green', zorder =1)
#
#    
#plt.xticks([0,1,2,3,4,5], ['Shock\nbaseline', 'Shock\nstimulus','Loom\nbaseline', 'Loom\nstimulus','Tone\nbaseline', 'Tone\nstimulus'])
#plt.ylim(-5, 105)
#plt.title('Exp1 Conditioning\n Freezing before and after stimulus')
#plt.ylabel('% Freezing')
#
#plt.show()
#
#%%

shock['condition'] = 'shock'
tone['condition'] = 'tone'
loom['condition'] = 'loom'

#
#
##%%
#tl_low_index =['MH072', 'MH096']
#
#for animal in tl_low_index:
#    loom.loc[animal,'condition'] = 't_loom_low'
#    
df_all = pd.concat([shock, loom, tone], axis =0)
df_all['difference'] = df_all['stim'] - df_all['base']


#%%
'''freezing during stimulus'''
fig = plt.figure(figsize=(3,4))
sns.set_style('ticks')

condition = list(df_all['condition'])
total = list(df_all['difference'])
ax = sns.boxplot(x=condition, y=total, color = 'lightgray', showfliers = False)
ax.set_ylim([-3,100])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=20)
ax = sns.swarmplot(x="condition", y="difference", 
                 data=df_all, palette = ['crimson', 'royalblue', 'green'], s=8)
ax.set_xlabel('', fontsize=20)
ax.set_ylabel('', fontsize=20)
ax.set_title('', fontsize = 20)
ax.spines['left'].set_position(('axes', - 0.02))

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas
#plt.savefig('escape_conditoining_new.jpeg',bbox_inches='tight', dpi=1800)

    #%%
    
'''colorcoded freezing Tone-Loom'''
loom['rank']= 0

for row in loom.iterrows():
    if row[0] == 'MH078':
        loom.loc[row[0], 'rank'] = 0    
    elif row[0] == 'MH101':
        loom.loc[row[0], 'rank'] = 1
    elif row[0] == 'MH071':
        loom.loc[row[0], 'rank'] = 2
    elif row[0] == 'MH077':
        loom.loc[row[0], 'rank'] = 3
    elif row[0] == 'MH092':
        loom.loc[row[0], 'rank'] = 4
    elif row[0] == 'MH072':
        loom.loc[row[0], 'rank'] = 5
    elif row[0] == 'MH096':
        loom.loc[row[0], 'rank'] = 6

loom['condition']='Tone-Loom'



fig = plt.figure(figsize=(3.5,4.5))
with sns.axes_style("ticks"):

    ax = sns.swarmplot(x="condition", y="difference", 
                    data=loom, hue = "rank", palette = ['indigo','darkslateblue', 'teal', 'cadetblue',
                                                                'mediumaquamarine', 'lightgreen', 'yellowgreen'], s = 10)
    ax.set_title('Time until retrieving pellet \nafter looms\n', fontsize =25)
    ax.set_ylim([0,100])
    ax.set_ylabel('Seconds', fontsize = 20)
    ax.set_xlabel('')
    ax.tick_params(labelsize=20)
    ax.legend(title = 'Pellet retrieval',title_fontsize=20, loc=[-1.5,0],markerscale=2., fontsize = 15)#
    
    ax.tick_params(labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#%%
    
'''colorcoded freezing Tone-Shock'''
shock['rank']= 0

for row in shock.iterrows():
    if row[0] == 'MH080':
        shock.loc[row[0], 'rank'] = 0    
    elif row[0] == 'MH084':
        shock.loc[row[0], 'rank'] = 1
    elif row[0] == 'MH089':
        shock.loc[row[0], 'rank'] = 2
    elif row[0] == 'MH090':
        shock.loc[row[0], 'rank'] = 3
    elif row[0] == 'MH074':
        shock.loc[row[0], 'rank'] = 4
    elif row[0] == 'MH073':
        shock.loc[row[0], 'rank'] = 5
    elif row[0] == 'MH079':
        shock.loc[row[0], 'rank'] = 6
    elif row[0] == 'MH100':
        shock.loc[row[0], 'rank'] = 7
shock['condition']='Tone-Shock'



fig = plt.figure(figsize=(3.5,4.5))
with sns.axes_style("ticks"):

    ax = sns.swarmplot(x="condition", y="stim", 
                    data=shock, hue = "rank", palette = ['indigo','darkslateblue', 'teal', 'cadetblue',
                                                                'mediumaquamarine', 'lightgreen', 'yellowgreen', 'yellow'], s = 10)
    ax.set_title('Time until retrieving pellet \nafter looms\n', fontsize =25)
    ax.set_ylim([0,100])
    ax.set_ylabel('Seconds', fontsize = 20)
    ax.set_xlabel('')
    ax.tick_params(labelsize=20)
    ax.legend(title = 'Pellet retrieval',title_fontsize=20, loc=[-1.5,0],markerscale=2., fontsize = 15)#
    
    ax.tick_params(labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)





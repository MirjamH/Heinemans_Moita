#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:17:36 2020

@author: mirjamheinemans
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

os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/analysis files') # to change directory Read csv files with Pandas
#%%

path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'


def PelletTrain(file):

    if file_names == '.DS_Store':
        next
    else:
        dataset = pd.read_csv(path_name +'/' + file_names +'/' + 'training.csv', usecols = [2,4,9])# 2 = in_shelter
        dataset.iloc[-10:,-1] = 1
        end_exp = dataset.iloc[:,2].diff()[dataset.iloc[:,2].diff() == 1].index.values[-1] 
        dataset_end = dataset.iloc[:int(end_exp - 1),:].reset_index(drop=True)      
        
        pellet3 = dataset_end.iloc[:,2].diff()[dataset_end.iloc[:,2].diff() == -1].index.values[-1]
        dataset_cut = dataset.iloc[int(pellet3):,:].reset_index(drop=True)  
        
        out_shelter = dataset_cut.iloc[:,0].diff()[dataset_cut.iloc[:,0].diff() == -1].index.values[0] 
        df_final = dataset_cut.iloc[int(out_shelter):,:].reset_index(drop=True)  
        

        pellet = df_final.loc[df_final.iloc[:,1] ==1].index.values[0]  
        
        
        number =file_names.replace('MH',"")
        if int(number) <39:
            pellet = pellet/30
            batch = [pellet, 1]
            df_pellet = pd.DataFrame(batch, index = ['train', 'batch'], columns = [file_names])
         
        elif 39 < int(number) < 45: # animals <45 have 30 FPS videos
 
            pellet = pellet/60
            batch = [pellet, 1]
            df_pellet = pd.DataFrame(batch, index = ['train', 'batch'], columns = [file_names])
            
        elif 45 < int(number) <71: #  animals >45 have 60 FPS videos
    
            pellet = pellet/90
            batch = [pellet, 1]
            df_pellet = pd.DataFrame(batch, index = ['train', 'batch'], columns = [file_names])
        
        else: #  animals >45 have 60 FPS videos
    
            pellet = pellet/60
            batch = [pellet, 2]
            df_pellet = pd.DataFrame(batch, index = ['train', 'batch'], columns = [file_names])
            
        return(df_pellet)
  
'''in this for-loop i create a list of lists of lists with each animal on one line.'''
path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'
columns = ['xpos']

train_loom = pd.DataFrame(index = ['train'], columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = PelletTrain(file_names)  
    train_loom = pd.concat([train_loom, animal], axis=1)
train_loom = train_loom.drop(columns = ['xpos'])
train_loom.loc['condition',:] = 'Loom'


#%%
"""
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
def PelletTest(file):

    if file_names == '.DS_Store':
        next
    else:
        
        dataset = pd.read_csv(path_name +'/' + file_names +'/' + 'test.csv', usecols = [2,4,9])# 2 = in_shelter
        '''here clock starts ticking when stimulus is triggered'''
        stimulus = dataset.loc[dataset.iloc[:,2] == 1].index.values.astype(int)[0]  
        dataset_stim = dataset.iloc[int(stimulus):,:].reset_index(drop=True) #18000 is 5 minutes
        
        
        number =file_names.replace('MH',"")
        
        if int(number) <39: # 60 FPS
            dataset_stim.iloc[54000,1] = 1 # set last row to 1 in case animal did not grab pellet
            pellet = dataset_stim.loc[dataset_stim.iloc[:,1] == 1].index.values[0] 
            pellet = pellet/60
            batch = [pellet, 1]
            df_pellet = pd.DataFrame(batch, index = ['test', 'batch'], columns = [file_names])

        elif 39 < int(number) <71: #  90 FPS videos
            dataset_stim.iloc[81000,1] = 1
            pellet = dataset_stim.loc[dataset_stim.iloc[:,1] == 1].index.values[0] 
            pellet = pellet/90
            batch = [pellet, 1]
            df_pellet = pd.DataFrame(batch, index = ['test', 'batch'], columns = [file_names])
         
        else: # 60 FPS videos
            dataset_stim.iloc[54000,1] = 1
            pellet = dataset_stim.loc[dataset_stim.iloc[:,1] == 1].index.values[0] 
            pellet = pellet/60
            batch = [pellet, 2]
            df_pellet = pd.DataFrame(batch, index = ['test', 'batch'], columns = [file_names])
         


            
        return(df_pellet)


path_name = '/Users/mirjamheinemans/Desktop/Annotator python tryout/_Loom'
columns = ['xpos']

test_loom = pd.DataFrame(index = ['test'], columns = columns)

for file_names in sorted(os.listdir(path_name)): 
    print(file_names)
    animal = PelletTest(file_names)  
    test_loom = pd.concat([test_loom, animal], axis=1)
test_loom = test_loom.drop(columns = ['xpos'])
test_loom.loc['condition',:] = 'Loom'
#all_loom = pd.concat([test_loom, train_loom])
#%%
loom = test_loom.T
loom['takes_pellet'] = 1
#%%
for row in loom.iterrows():
#    print(row)
#    #%%
    if row[0] == 'MH033':
        loom.loc[row[0], 'takes_pellet'] = 0    
    elif row[0] == 'MH034':
        loom.loc[row[0], 'takes_pellet'] = 0
    elif row[0] == 'MH046':
        loom.loc[row[0], 'takes_pellet'] = 2
    elif row[0] == 'MH059':
        loom.loc[row[0], 'takes_pellet'] = 1
    elif row[0] == 'MH065':
        loom.loc[row[0], 'takes_pellet'] = 0
    elif row[0] == 'MH066':
        loom.loc[row[0], 'takes_pellet'] = 6
    elif row[0] == 'MH085':
        loom.loc[row[0], 'takes_pellet'] = 5
    elif row[0] == 'MH086':
        loom.loc[row[0], 'takes_pellet'] = 4
    elif row[0] == 'MH087':
        loom.loc[row[0], 'takes_pellet'] = 0
    elif row[0] == 'MH088':
        loom.loc[row[0], 'takes_pellet'] = 3
    elif row[0] == 'MH089':
        loom.loc[row[0], 'takes_pellet'] = 1
        
        
    
    
#%%

#before_loom = loom['train']
#after_loom = loom['test']
#
#
#plt.scatter(np.zeros(len(before_loom)), before_loom, color = 'black', s = 50, zorder = 2)
#plt.scatter(np.ones(len(after_loom)), after_loom, color = 'black',s =50, zorder = 2)
#
#for i in range(len(before_loom)):
#    plt.plot( [0,1], [before_loom[i], after_loom[i]], c='black', zorder =1)
#
#
#plt.xticks([0,1], ['Training', 'Test'])
#plt.ylim(-5, 900)
#plt.title('Loom\n Pellet retrieval time from shelter exit')
#plt.ylabel('seconds')
#
#plt.show()

condition = list(loom['condition'])
total = list(loom['test'])

fig = plt.figure(figsize=(2,4))
with sns.axes_style("ticks"):

    ax = sns.swarmplot(x="condition", y="test", 
                     data=loom, hue = "takes_pellet", palette = ['black','darkslateblue', 'teal', 'cadetblue',
                                                                 'mediumaquamarine', 'lightgreen', 'yellowgreen'], s = 17)
    #ax.set_title('Time until retrieving pellet \nafter looms\n', fontsize =25)
    #ax = sns.boxplot(x=condition, y=total, palette="Set2", showfliers = False)
    ax.set_ylabel('Seconds', fontsize = 20)
    ax.set_xlabel('')
    ax.tick_params(labelsize=20)
    ax.legend(title = '',title_fontsize=2, loc=[-0.5,0],markerscale=1., fontsize = 5)#
    
    ax.tick_params(labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Seconds', fontsize = 20)
    ax.set_ylim([-30,950])
    ax.set_yticks([0,300,600,900])
    
    ax.tick_params(labelsize=20)
    #ax.set_xlabel('Condition', fontsize = 20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_tick_params(size=5, width=2)
    ax.xaxis.set_tick_params(size=5, width=2, color='k')
#    ax.margins(2,2)

    ax.spines['left'].set_position(('axes', - 0.02))
    ax.spines['bottom'].set_position(('axes', -0.05)) 
os.chdir('/Users/mirjamheinemans/Desktop/Annotator python tryout/xExp 1') # to change directory Read csv files with Pandas

plt.savefig('loom_test_pell_time.jpeg',bbox_inches='tight', dpi=1800)


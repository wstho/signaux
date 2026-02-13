#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 13:58:57 2026

@author: wst
"""

import numpy as np

from neo import SpikeTrain
from elephant import statistics as ele_stats

from elephant.spike_train_correlation import correlation_coefficient as cc
from elephant.conversion import BinnedSpikeTrain as BST
from quantities import s, ms

import matplotlib.pyplot as plt
import seaborn as sns



def plot_spikes(sg, postsynaptic, ax = None, palette = 'Set2'):
    """Plot spike rasters for visualization"""
    pal = sns.color_palette(palette, n_colors = len(sg.inputs.keys())+1)
    
    if ax is None:
        fig, ax = plt.subplots(figsize = [20, 12])
    y = 0
    for idx, trainset in enumerate(sg.total_spikes[str(postsynaptic)].values()):
        for train in trainset:
            ax.scatter(train, [y]*len(train), c = pal[idx], marker = '|', s = 5)
            y +=  1
        
    try:
        start = sg.start
    except:
        start = 0 
    try:
        end = sg.end
    except:
        end = 5
        
    ax.set_xlim(start, end)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Synapse')
    sns.despine()
    plt.show()
    
def plot_ISI_distribution(sg, postsynaptic = 0, ax = None, separate_plots = False, palette = 'Set2'):
    """Plot inter-spike interval distributions of unique spike trains"""
    pal = sns.color_palette(palette, n_colors = len(sg.inputs.keys())+1)
    
    unique_trains  = sg.get_unique_trains(str(postsynaptic))
        
    if ax is None:
        if separate_plots:
            fig, axes = plt.subplots(nrows = len(unique_trains))
            
            for i, spiketrain in enumerate(unique_trains):
                isis = ele_stats.isi(spiketrain)*1000      #s --> ms  
                sns.kdeplot(x =  isis, ax =  axes[i], clip = (0, None), cut = 0, color = pal[i])
            
            for ax in axes[:-1]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[-1].set_xlabel('ISI (ms)')
            axes[-1].set_yticks([])
            
        else:
            fig, ax = plt.subplots(nrows = 1)
            for i, spiketrain in enumerate(unique_trains):
                isis = ele_stats.isi(spiketrain)*1000         #s --> ms        
                sns.kdeplot(x = isis, ax = ax, clip = (0, None), cut = 0, color = pal[i])
                
            ax.set_xlabel('ISI (ms)')
            ax.set_yticks([])
            ax.set_xlim(0, None)
                
    else:
        for i, spiketrain in enumerate(unique_trains):
            isis = ele_stats.isi(spiketrain)*1000         #s --> ms        
            sns.kdeplot(x = isis, ax = ax, clip = (0, None), cut = 0, color = pal[i])
            
        ax.set_xlabel('ISI (ms)')
        ax.set_yticks([])
        ax.set_xlim(0, None)
    sns.despine()
    plt.show()
    
    
def plot_input_correlation(sg, ax = None, binsize = 10, cmap = 'viridis'):
    """"Plot correlation matrix of spike trains """
    if ax ==  None:
        fig, ax = plt.subplots()
    trains = []
    for st in sorted(sg.csv_spikes, key = lambda x: x[0]):
        st = [x for x in st if (x < sg.end) & (x> sg.start)]
        trains.append(SpikeTrain(times = st*s, t_start = sg.start*s, t_stop = sg.end*s))

    cc_matrix = cc(BST(trains, binsize = binsize*ms))
    im = ax.imshow(cc_matrix, clim = (-1, 1), cmap = cmap) 
    fig.colorbar(im, orientation = 'vertical', label = 'Correlation Coefficient')

    ax.set_title('Spike Time Correlation')
    ax.set_xlabel('Input ID')
    ax.set_ylabel('Input ID')
    plt.show()
    sns.despine()
    plt.show()
       
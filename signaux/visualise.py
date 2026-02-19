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



def plot_spike_raster(sg, postsynaptic = 0, input_type = None, ax = None, palette = 'Set2', size = 5, label = True):
    """Plot spike rasters for visualization"""
    pal = sns.color_palette(palette, n_colors = len(sg.total_trains[str(postsynaptic)].keys())+1)
    
    if ax is None:
        fig, ax = plt.subplots(figsize = [20, 12])
        
    y = 0
    y_tick_labels = []
    for input_name, spiketrain in sg.total_trains[str(postsynaptic)].items():
        if input_type:
            if input_type not in input_name:
                continue
        ax.scatter(spiketrain, [y]*len(spiketrain), c = pal[y], marker = '|', s = size)
        y_tick_labels.append(input_name)
        y +=  1
        
    
        
    start = sg.inputs[input_name]['start']
    end = sg.inputs[input_name]['end']  
    ax.set_xlim(start, end)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Input')
    if label:
        ax.set_yticks(range(0, y))
        ax.set_yticklabels(y_tick_labels)
    sns.despine()
    plt.show()
    
def plot_ISI_distribution(sg, postsynaptic = 0, input_type = None, ax = None, separate_plots = False, palette = 'Set2'):
    """Plot inter-spike interval distributions of unique spike trains"""
   
    if isinstance(palette, str):
        pal = sns.color_palette(palette, n_colors = len(sg.inputs.keys())+1)
        color_coded = False
    elif isinstance(palette, dict):
        pal = palette
        color_coded = True
    
    
    isis = isi_distribution_helper(sg, postsynaptic, input_type)

    if ax is None:
        if separate_plots:
            fig, axes = plt.subplots(nrows = len(isis))
            for i, (input_name, isi) in enumerate(isis.items()):
                if color_coded:
                    sns.kdeplot(x = isi, ax = axes[i], clip = (0, None), cut = 0, color = pal[input_name.split('_')[0]])
                else:
                    sns.kdeplot(x = isi, ax = axes[i], clip = (0, None), cut = 0, color = pal[i])
            
            for ax in axes[:-1]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[-1].set_xlabel('ISI (ms)')
            axes[-1].set_yticks([])
            
        else:
            fig, ax = plt.subplots(nrows = 1)
            for i, (input_name, isi) in enumerate(isis.items()):
                if color_coded:
                    sns.kdeplot(x = isi, ax = ax, clip = (0, None), cut = 0, color = pal[input_name.split('_')[0]])
                else:
                    sns.kdeplot(x = isi, ax = ax, clip = (0, None), cut = 0, color = pal[i])

            ax.set_xlabel('ISI (ms)')
            ax.set_yticks([])
            ax.set_xlim(0, None)
                
    else:
        for i, (input_name, isi) in enumerate(isis.items()):
            if color_coded:
                sns.kdeplot(x = isi, ax = ax, clip = (0, None), cut = 0, color = pal[input_name.split('_')[0]])
            else:
                sns.kdeplot(x = isi, ax = ax, clip = (0, None), cut = 0, color = pal[i])

            
        ax.set_xlabel('ISI (ms)')
        ax.set_yticks([])
        ax.set_xlim(0, None)
    sns.despine()
    plt.show()
    
    
def isi_distribution_helper(sg, postsynaptic, input_type = None ):
    isis = {}
    for i, (input_name, spiketrain) in enumerate(sg.total_trains[str(postsynaptic)].items()):
        if input_type:
            if input_type not in input_name:
                continue
        isis[input_name] = ele_stats.isi(spiketrain)*1000       #s --> ms        

    return isis
    
def plot_input_correlation(sg, postsynaptic = 0, input_type = None, ax = None, binsize = 10, cmap = 'viridis', label = False):
    """"plot correlation matrix of spike trains """
    
    if ax ==  None:
        fig, ax = plt.subplots()
    trains = []
    tick_labels = []
    for input_name, spiketrain in sg.total_trains[str(postsynaptic)].items():
        if input_type:
            if input_type not in input_name:
                continue
        trains.append(SpikeTrain(times = spiketrain*s, t_start = sg.default_start*s, t_stop = sg.default_end*s))
        tick_labels.append(input_name)

    cc_matrix = cc(BST(trains, bin_size = binsize*ms))
    im = ax.imshow(cc_matrix, clim = (-1, 1), cmap = cmap) 
    fig.colorbar(im, orientation = 'vertical', label = 'Correlation Coefficient')

    ax.set_title('Spike Time Correlation')
    ax.set_xlabel('Input ID')
    ax.set_ylabel('Input ID')
    if label:
        ax.set_xticks(range(0, len(tick_labels)))
        ax.set_yticks(range(0, len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation = 90)
        ax.set_yticklabels(tick_labels)
    plt.show()
    sns.despine()
    plt.show()
       
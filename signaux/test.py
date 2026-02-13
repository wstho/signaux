#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:05:18 2025

@author: wst
"""
import os
import numpy as np
os.chdir('/Users/wst/Desktop/Karolinska/Simulation/Neuron/signaux/signaux')
from signaux import Signaux

net_name = 'test'
os.chdir('/Users/wst/Desktop/Karolinska/Simulation/Neuron/')
network_path = os.path.join(os.getcwd(), "networks", net_name)
try:
    os.mkdir(network_path)
except FileExistsError:
    pass 
rc = None

dummy = list(np.arange(0, 100))
sg = Signaux(network_path, rc = rc, postsynaptic = dummy, input_defs = [{'dSPN': {'generator': 'csv','n_presynaptic':10,"num_inputs" : 50, 'frequency': 1.4, 'variability' : 1,'end':5, 'bursts': [[1.25,1.5, 50, 100]]}}])
sg.check_for_csv()
sg.write_json()
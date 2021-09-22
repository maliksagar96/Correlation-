# MT matrix format:
    # 1 row per bead per frame, sorted by frame number then x position (roughly)
    # columns:
    # 1:2 - X and Y positions (in pixels)
    # 3   - Integrated intensity(mass)
    # 4   - Rg squared of feature
    # 5   - eccentricity
    # 6   - frame #
    # 7   - time of frame
#dataframe format used in trackpy    
# df = pd.DataFrame(data, columns = ['y', 'x', 'mass', 'size', 'ecc', signal , raw_mass, 'ep', 'frame'])     

#For linking if we have to plot the S4, xi4 and correlation length we have to make sure that number of frames and 
#goodenough are equal or else the xi.py file will have complications.
        
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import plot, draw, show
from pandas import DataFrame, Series
import pims
import trackpy as tp
import time
import threading
import multiprocessing
import concurrent.futures

ps  = 7
goodEnough = 450
dispDistance = (ps+1)/2                 # Distance a particle is allowed to move between 2 consecutive frames
mem = 3                               # memory 

if __name__ == "__main__":
    print("Starting linking particles with ps = ",ps," goodenough = ", goodEnough, "and memory = ", mem)
    fname  = '/home/sagar/Documents/codes/finalCodes/correlation/featureFinding/fov0/MT_featsize_7.npy'
    data = np.load(fname)
        
    ep = np.zeros(len(data[:,0]), dtype = type(data[0,0]))
    signal  = np.zeros(len(data[:,0]), dtype = type(data[0,0]))
    raw_mass = np.zeros(len(data[:,0]), dtype = type(data[0,0]))

    df = pd.DataFrame({'y':data[:,1], 'x':data[:,0], 'mass':data[:,2], 'size':data[:,3], 'ecc':data[:,4],'signal':signal, 'raw_mass':raw_mass, 'ep':ep, 'frame':data[:,5] })

    t = tp.link(df, dispDistance, memory = mem)
    t = tp.filter_stubs(t, goodEnough)
   
    column0 = t['x'].tolist()
    column1 = t['y'].tolist()  
    column2 = t['frame'].tolist()
    column3 = t['particle'].tolist()
    
    finalData = [column0, column1, column2, column3]    
    finalData = np.array(finalData)

    finalData = np.transpose(finalData)

    np.save('link057_450', finalData)